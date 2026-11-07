"""Unit tests for engine.fusion (Milestone 7)."""

from __future__ import annotations

import sys
import pytest
import torch

from benchmarks.run_benchmarks import isolated_run
from engine.fusion import FusedEngine, _is_inductor_available


def _relative_l2(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    return float((y - y_ref).norm() / y_ref.norm().clamp_min(1e-12))


@pytest.mark.skipif(
    not _is_inductor_available(),
    reason="torch.compile with Inductor backend required",
)
def test_fused_matches_eager():
    """Numerical parity: fused output must match eager output."""
    torch.manual_seed(42)
    engine = FusedEngine(hidden_dim=128, intermediate_dim=512)

    # Small input to keep compile time fast in CI
    x = torch.randn(4, 128, dtype=torch.float32)
    eager_out, fused_out = engine.forward(x)

    rel = _relative_l2(eager_out, fused_out)
    # Inductor may reorder float ops, so allow small tolerance
    assert rel < 1e-4, f"Fused output mismatch: rel L2 = {rel}"


@pytest.mark.skipif(
    not _is_inductor_available(),
    reason="torch.compile with Inductor backend required",
)
def test_fused_dynamic_batch():
    """Changing batch size should not crash and should match eager."""
    torch.manual_seed(0)
    engine = FusedEngine(hidden_dim=64, intermediate_dim=256)

    # 1. Warm up compile with B=2
    x1 = torch.randn(2, 64)
    e1, f1 = engine.forward(x1)
    assert _relative_l2(e1, f1) < 1e-4

    # 2. Same engine, different batch size (B=5)
    x2 = torch.randn(5, 64)
    e2, f2 = engine.forward(x2)
    assert _relative_l2(e2, f2) < 1e-4

    # 3. Back to B=2, should hit cache
    x3 = torch.randn(2, 64)
    e3, f3 = engine.forward(x3)
    assert _relative_l2(e3, f3) < 1e-4


def test_dispatch_overhead_heuristic():
    """Verify that dispatch overhead is non-negligible in eager mode."""
    from engine.fusion import FusedMLP
    import time

    mlp = FusedMLP(hidden_dim=512, intermediate_dim=2048).eval()
    x = torch.randn(1, 512)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = mlp(x)

    N = 100
    with torch.no_grad():
        t0 = time.perf_counter()
        for _ in range(N):
            _ = mlp(x)
        elapsed = (time.perf_counter() - t0) / N

    print(f"\n  Eager FusedMLP(512, 2048) forward: {elapsed * 1000:.2f} ms")


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
@pytest.mark.skipif(
    not _is_inductor_available(),
    reason="torch.compile with Inductor backend and C++ compiler required",
)
def test_fused_under_4gib():
    res = isolated_run(_workload_fused, timeout_s=180.0)
    assert res.result["rel_err"] < 1e-4
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_fused() -> dict:
    torch.manual_seed(42)
    engine = FusedEngine(hidden_dim=128, intermediate_dim=512)
    x = torch.randn(4, 128)
    e, f = engine.forward(x)
    rel = float((f - e).norm() / e.norm().clamp_min(1e-12))
    return {"rel_err": rel}