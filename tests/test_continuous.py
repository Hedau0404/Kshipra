"""Unit tests for engine.continuous (Milestone 5)."""

from __future__ import annotations

import sys

import pytest

from benchmarks.run_benchmarks import isolated_run
from engine.continuous import ContinuousEngine, SeqState
from engine.prefix_cache import PrefixCache


def _make_engine(pool_blocks: int = 100) -> ContinuousEngine:
    cache = PrefixCache(
        num_blocks=pool_blocks,
        block_size=64,
        num_layers=2,
        num_kv_heads=2,
        head_dim=8,
    )
    return ContinuousEngine(
        prefix_cache=cache,
        block_size=64,
        max_batch_tokens=1024,
        itl_budget_s=0.1,
        t_tok_s=0.01,
    )


def test_admission_strict_reservation():
    engine = _make_engine(pool_blocks=10)
    req = engine.add_request([0] * 600, 40)  # ceil(640/64) = 10 blocks
    assert engine._admission_check(req) is True
    engine.schedule()
    assert req.state == SeqState.PREFILLING
    assert len(req.block_table) == 10

    req2 = engine.add_request([0] * 64, 10)
    assert engine._admission_check(req2) is False


def test_scheduler_interleaves_prefill_and_decode():
    engine = _make_engine(pool_blocks=100)

    # Admit one sequence and manually push it into RUNNING (prefill done).
    req_dec = engine.add_request([0] * 128, 10)
    engine.schedule()
    req_dec.prefill_pos = len(req_dec.prompt_tokens)
    req_dec.state = SeqState.RUNNING
    req_dec.generated_tokens = [0]

    # Add a long prefill request.
    req_pre = engine.add_request([1] * 512, 10)

    sched = engine.schedule()

    assert len(sched.decodes) == 1
    assert sched.decodes[0] is req_dec

    assert len(sched.prefills) == 1
    pre_req, pre_chunk = sched.prefills[0]
    assert pre_req is req_pre
    # Budget = int(0.1/0.01) = 10; n_d = 1 → c_max = 9.
    assert pre_chunk == 9
    assert sched.total_tokens == 10


def test_preemption_swap():
    # 640 prompt + 10 max = 650 → 11 blocks. Need pool >= 11.
    engine = _make_engine(pool_blocks=12)
    req = engine.add_request([0] * 640, 10)
    engine.schedule()

    assert req.state == SeqState.PREFILLING
    assert len(req.block_table) == 11
    assert len(engine.cache.allocator.free_blocks) == 1

    # Force to RUNNING to satisfy the preemption contract (victim is an
    # active generator).
    req.prefill_pos = len(req.prompt_tokens)
    req.state = SeqState.RUNNING

    engine._preempt(req)

    assert req.state == SeqState.WAITING
    assert len(req.block_table) == 0
    assert req.swapped_kv is not None
    assert req.swapped_kv.shape[0] == 11
    assert len(engine.cache.allocator.free_blocks) == 12


def test_itl_budget_bound():
    engine = _make_engine()
    for _ in range(5):
        r = engine.add_request([0] * 64, 10)
        engine.schedule()
        r.prefill_pos = len(r.prompt_tokens)
        r.state = SeqState.RUNNING
        r.generated_tokens = [0]

    rp = engine.add_request([1] * 256, 10)
    sched = engine.schedule()

    # Budget = 10; n_d = 5 → c_max = 5.
    assert len(sched.prefills) == 1
    assert sched.prefills[0][1] == 5
    assert sched.total_tokens == 10


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_continuous_under_4gib():
    res = isolated_run(_workload_continuous, timeout_s=180.0)
    assert res.result["finished"] == 50
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_continuous() -> dict:
    engine = _make_engine(pool_blocks=100)
    for i in range(50):
        engine.add_request([i % 100] * 64, 5)

    for _ in range(5000):  # iteration bound
        engine.step()
        if not engine.active_sequences and not engine.waiting_queue:
            break

    return {"finished": engine._total_finished}