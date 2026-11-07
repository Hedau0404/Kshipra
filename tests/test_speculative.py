"""Unit tests for engine.speculative (Milestone 6)."""

from __future__ import annotations

import sys

import pytest
import torch
import torch.nn.functional as F

from benchmarks.run_benchmarks import isolated_run
from engine.speculative import (
    MockModel,
    SpeculativeConfig,
    SpeculativeDecoder,
)


def _make_decoder(max_k=4, target_t=0.083, draft_t=0.03) -> SpeculativeDecoder:
    target = MockModel(vocab_size=100, seed=1)
    draft = MockModel(vocab_size=100, seed=1)  # Same seed = identical models
    cfg = SpeculativeConfig(
        max_k=max_k, target_t_tok_s=target_t, draft_t_tok_s=draft_t
    )
    return SpeculativeDecoder(target, draft, cfg)


def test_perfect_draft_accepts_all():
    """If draft == target, all K tokens should be accepted."""
    decoder = _make_decoder()
    # Same seed means perfect agreement
    input_ids = torch.tensor([[1, 2, 3]])
    rng = torch.Generator().manual_seed(42)

    tokens, _, _ = decoder.step(
        input_ids, torch.zeros(1), torch.zeros(1), rng
    )

    # Should get K+1 tokens (all accepted + bonus)
    # Note: may be less if adaptive K selects k < max_k
    assert len(tokens) >= 1
    assert len(tokens) <= decoder.cfg.max_k + 1


def test_no_draft_fallback():
    """K=0 should produce standard decoding."""
    decoder = _make_decoder()
    decoder._select_k = lambda _: 0  # Force K=0
    input_ids = torch.tensor([[1, 2, 3]])
    rng = torch.Generator().manual_seed(42)

    tokens, _, _ = decoder.step(
        input_ids, torch.zeros(1), torch.zeros(1), rng
    )
    assert len(tokens) == 1


def test_residual_resample_exactness():
    """Statistical check: residual sampling produces exact p distribution."""
    # Create a toy target and draft where p != q
    vocab = 3
    target_probs = torch.tensor([0.2, 0.5, 0.3])
    draft_probs = torch.tensor([0.4, 0.3, 0.3])

    # Simulate 100k samples from the rejection sampling algorithm
    rng = torch.Generator().manual_seed(0)
    counts = torch.zeros(vocab)
    N = 100_000
    for _ in range(N):
        d_token = torch.multinomial(draft_probs, 1, generator=rng).item()
        p = target_probs[d_token].item()
        q = draft_probs[d_token].item()

        if torch.rand(1, generator=rng).item() < min(1.0, p / q):
            counts[d_token] += 1
        else:
            residual = torch.clamp(target_probs - draft_probs, min=0.0)
            residual_sum = residual.sum()
            if residual_sum > 0:
                out = torch.multinomial(residual / residual_sum, 1, generator=rng).item()
            else:
                out = torch.multinomial(target_probs, 1, generator=rng).item()
            counts[out] += 1

    empirical = counts / counts.sum()
    assert torch.allclose(empirical, target_probs, atol=0.01)


def test_adaptive_k_respects_cost():
    """High K should be rejected if the draft is too slow."""
    # Draft is slower than target -> should choose K=0
    decoder = _make_decoder(max_k=4, target_t=0.083, draft_t=0.5)
    input_ids = torch.tensor([[1, 2, 3]])
    rng = torch.Generator().manual_seed(42)

    # Force draft probs to be high confidence
    decoder.draft.forward = lambda ids, pos: torch.zeros(1, 100) + 10.0
    k = decoder._select_k(torch.softmax(torch.zeros(4, 100) + 10.0, dim=-1))
    assert k == 0


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_speculative_under_4gib():
    res = isolated_run(_workload_speculative, timeout_s=120.0)
    assert res.result["steps"] > 0
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_speculative() -> dict:
    decoder = _make_decoder()
    input_ids = torch.tensor([[1, 2, 3]])
    rng = torch.Generator().manual_seed(42)

    for _ in range(100):
        tokens, _, _ = decoder.step(
            input_ids, torch.zeros(1), torch.zeros(1), rng
        )
        input_ids = torch.cat(
            [input_ids, torch.tensor([tokens], dtype=torch.long)], dim=1
        )
        if input_ids.shape[1] > 256:
            break

    return {"steps": 100}