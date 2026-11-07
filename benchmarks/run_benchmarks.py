"""Isolated memory + latency measurement harness for Kshipra.

Includes:
- Isolated spawned subprocess for true OS RSS measurement.
- Poisson traffic generation.
- Latency percentile calculation (P50, P90, P99).
- Op-level profiling of the engine step.

The engine is now yours to extend. You could plug in a real transformers model, 
replace the mock attention with the fused C++ kernel, or 
deploy it behind a FastAPI server.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from benchmarks.traffic import generate_poisson_traffic, Request
from benchmarks.profile import profile_function

# ---------------------------------------------------------------------- #
# RSS reading
# ---------------------------------------------------------------------- #

def _read_vmhwm_bytes() -> int:
    """Return the kernel's peak-RSS high-water mark for the current process."""
    with open("/proc/self/status", "r") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                parts = line.split()
                return int(parts[1]) * 1024
    raise RuntimeError("VmHWM not present in /proc/self/status")


def read_vmrss_bytes() -> int:
    """Return the *current* resident set size of the calling process."""
    with open("/proc/self/status", "r") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                parts = line.split()
                return int(parts[1]) * 1024
    raise RuntimeError("VmRSS not present in /proc/self/status")


# ---------------------------------------------------------------------- #
# Child entry point
# ---------------------------------------------------------------------- #

def _child_entry(
    fn: Callable[..., Any],
    kwargs: Dict[str, Any],
    conn: Any,
    num_threads: int,
) -> None:
    """Runs in the spawned child. Reports baseline and peak RSS to parent."""
    torch.set_num_threads(num_threads)
    _ = torch.zeros(1, dtype=torch.float32)

    baseline_bytes = _read_vmhwm_bytes()
    t0 = time.perf_counter()
    payload: Dict[str, Any]
    try:
        result = fn(**kwargs)
        elapsed_s = time.perf_counter() - t0
        payload = {
            "status": "ok",
            "result": result,
            "baseline_bytes": baseline_bytes,
            "peak_bytes": _read_vmhwm_bytes(),
            "elapsed_s": elapsed_s,
        }
    except BaseException:
        payload = {
            "status": "error",
            "traceback": traceback.format_exc(),
            "baseline_bytes": baseline_bytes,
            "peak_bytes": _read_vmhwm_bytes(),
            "elapsed_s": time.perf_counter() - t0,
        }
    try:
        conn.send(payload)
    finally:
        conn.close()


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #

@dataclass(frozen=True)
class IsolatedRunResult:
    """Result of a single :func:`isolated_run` invocation."""
    result: Any
    baseline_bytes: int
    peak_bytes: int
    elapsed_s: float

    @property
    def delta_bytes(self) -> int:
        return max(0, self.peak_bytes - self.baseline_bytes)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / (1024 ** 3)

    @property
    def delta_mib(self) -> float:
        return self.delta_bytes / (1024 ** 2)


def isolated_run(
    fn: Callable[..., Any],
    kwargs: Optional[Dict[str, Any]] = None,
    timeout_s: float = 600.0,
    num_threads: int = 1,
) -> IsolatedRunResult:
    """Run ``fn(**kwargs)`` in a spawned subprocess and measure peak RSS."""
    if sys.platform != "linux":
        raise RuntimeError(
            "isolated_run requires Linux (/proc/self/status VmHWM); "
            f"got platform={sys.platform!r}"
        )
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")

    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_child_entry,
        args=(fn, dict(kwargs or {}), child_conn, num_threads),
        daemon=False,
    )
    proc.start()
    child_conn.close()

    if not parent_conn.poll(timeout_s):
        proc.terminate()
        proc.join(timeout=5.0)
        raise TimeoutError(
            f"isolated workload exceeded {timeout_s}s (pid={proc.pid}, "
            f"exitcode={proc.exitcode})"
        )

    try:
        payload = parent_conn.recv()
    except EOFError as exc:
        proc.join(timeout=5.0)
        raise RuntimeError(
            f"isolated child died before reporting (exitcode={proc.exitcode})."
        ) from exc
    finally:
        parent_conn.close()

    proc.join(timeout=10.0)
    if payload["status"] == "error":
        raise RuntimeError(
            "isolated workload failed:\n" + payload["traceback"]
        )
    if proc.exitcode not in (0, None):
        raise RuntimeError(
            f"isolated child exited with code {proc.exitcode}"
        )

    return IsolatedRunResult(
        result=payload["result"],
        baseline_bytes=payload["baseline_bytes"],
        peak_bytes=payload["peak_bytes"],
        elapsed_s=payload["elapsed_s"],
    )


def latency_percentiles(latencies_s: Sequence[float]) -> Dict[str, float]:
    """Return p50/p90/p99 for a sequence of per-request latencies."""
    if not latencies_s:
        raise ValueError("latencies_s must be non-empty")
    t = torch.tensor(list(latencies_s), dtype=torch.float64)
    qs = torch.tensor([0.50, 0.90, 0.99], dtype=torch.float64)
    vals = torch.quantile(t, qs, interpolation="linear")
    return {
        "p50": float(vals[0]),
        "p90": float(vals[1]),
        "p99": float(vals[2]),
    }


# ---------------------------------------------------------------------- #
# Executable Benchmark Runner
# ---------------------------------------------------------------------- #

def _benchmark_workload() -> dict:
    """A representative workload measuring the full engine."""
    import time
    from engine.continuous import ContinuousEngine, SeqState
    from engine.prefix_cache import PrefixCache
    from engine.fusion import FusedMLP
    import torch

    # 1. Setup engine
    cache = PrefixCache(
        num_blocks=100, block_size=64, num_layers=2, num_kv_heads=2, head_dim=8
    )
    engine = ContinuousEngine(
        prefix_cache=cache,
        block_size=64,
        max_batch_tokens=1024,
        itl_budget_s=0.05,
        t_tok_s=0.005,
    )

    # 2. Generate Poisson traffic
    from benchmarks.traffic import generate_poisson_traffic
    traffic = generate_poisson_traffic(
        num_requests=30,
        arrival_rate_rps=2.0,
        prompt_len_range=(64, 256),
        new_tokens_range=(5, 15),
    )

    latencies = []
    all_requests = []
    start_time = time.perf_counter()
    req_idx = 0
    
    # 3. Main simulation loop
    while req_idx < len(traffic) or engine.active_sequences or engine.waiting_queue:
        sim_time = time.perf_counter() - start_time
        while req_idx < len(traffic) and sim_time >= traffic[req_idx].arrival_time_s:
            req = traffic[req_idx]
            r = engine.add_request([req.req_id % 100] * req.prompt_len, req.max_new_tokens)
            all_requests.append(r)
            req_idx += 1
            
        if engine.active_sequences or engine.waiting_queue:
            engine.step()
        else:
            time.sleep(0.001)
            
        for req in all_requests:
            if req.state == SeqState.FINISHED and req.finish_time is not None:
                latencies.append(req.finish_time - req.arrival_time)
                req.finish_time = None

    total_time = time.perf_counter() - start_time
    
    # 4. Profile a REAL tensor workload (FusedMLP)
    # We use a tiny MLP so profiling is fast but still generates aten ops.
    mlp = FusedMLP(hidden_dim=512, intermediate_dim=2048).eval()
    x = torch.randn(4, 512)
    prof_breakdown = profile_function(mlp, x)
    
    return {
        "latencies": latencies,
        "total_time": total_time,
        "profile": prof_breakdown,
    }


if __name__ == "__main__":
    print("Starting Kshipra Benchmark Suite...\n")
    
    res = isolated_run(_benchmark_workload, timeout_s=300.0)
    
    print(f"=== Memory Profile ===")
    print(f"Baseline RSS: {res.baseline_bytes / 1024**2:.1f} MiB")
    print(f"Peak RSS:     {res.peak_bytes / 1024**2:.1f} MiB")
    print(f"Delta:        {res.delta_mib:.1f} MiB")
    
    if res.peak_bytes > 4 * 1024**3:
        print("❌ FAILED: Peak memory exceeded 4 GiB")
    else:
        print("✅ PASSED: Peak memory under 4 GiB")

    print(f"\n=== Latency Profile ===")
    latencies = res.result["latencies"]
    if latencies:
        percentiles = latency_percentiles(latencies)
        print(f"Total Requests: {len(latencies)}")
        print(f"Total Wall Time: {res.result['total_time']:.2f} s")
        print(f"P50 Latency: {percentiles['p50'] * 1000:.2f} ms")
        print(f"P90 Latency: {percentiles['p90'] * 1000:.2f} ms")
        print(f"P99 Latency: {percentiles['p99'] * 1000:.2f} ms")
    
    print(f"\n=== Top Operators (CPU) ===")
    profile = res.result["profile"]
    if not profile:
        print("No profiling data captured.")
    else:
        for op, pct in list(profile.items())[:5]:  # Top 5
            print(f"{pct:5.1f}%  {op}")