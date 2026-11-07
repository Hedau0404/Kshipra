"""Unit tests for engine.chunked_prefill (Milestone 4)."""

from __future__ import annotations

import math
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.run_benchmarks import isolated_run
from engine.chunked_prefill import chunked_attention, monolithic_attention
from engine.prefix_cache import PrefixCache


def _relative_l2(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    return float((y - y_ref).norm() / y_ref.norm().clamp_min(1e-12))


def test_chunked_matches_monolithic_random():
    """Golden test: chunked online softmax must match monolithic dense mask."""
    torch.manual_seed(42)
    H, C, D = 4, 128, 64
    P = 256
    B = 16

    q = torch.randn(H, C, D)
    k = torch.randn(H, P + C, D)
    v = torch.randn(H, P + C, D)

    # 1. Monolithic reference
    ref = monolithic_attention(q, k, v, P, C)

    # 2. Chunked path
    num_blocks = (P + C + B - 1) // B
    k_pool = torch.zeros(num_blocks, H, B, D)
    v_pool = torch.zeros(num_blocks, H, B, D)
    block_table = list(range(num_blocks))

    # Fill pool
    for i in range(P + C):
        blk = i // B
        off = i % B
        k_pool[blk, :, off, :] = k[:, i, :]
        v_pool[blk, :, off, :] = v[:, i, :]

    out = chunked_attention(q, k_pool, v_pool, block_table, P, C, B)

    rel = _relative_l2(ref, out)
    assert rel < 1e-5, f"Chunked attention mismatch: {rel}"


def test_chunked_attention_prefix_only():
    """P=256, C=0 means just process the prefix (no query)."""
    # Not applicable here because Q is length C. But if C is small,
    # prefix dominates.
    torch.manual_seed(42)
    H, C, D = 4, 1, 64
    P = 1024
    B = 16

    q = torch.randn(H, C, D)
    k = torch.randn(H, P + C, D)
    v = torch.randn(H, P + C, D)

    ref = monolithic_attention(q, k, v, P, C)

    num_blocks = (P + C + B - 1) // B
    k_pool = torch.zeros(num_blocks, H, B, D)
    v_pool = torch.zeros(num_blocks, H, B, D)
    block_table = list(range(num_blocks))
    for i in range(P + C):
        blk = i // B
        off = i % B
        k_pool[blk, :, off, :] = k[:, i, :]
        v_pool[blk, :, off, :] = v[:, i, :]

    out = chunked_attention(q, k_pool, v_pool, block_table, P, C, B)
    rel = _relative_l2(ref, out)
    assert rel < 1e-5


def test_unaligned_prefix_boundary():
    """P not a multiple of B."""
    torch.manual_seed(42)
    H, C, D = 4, 16, 64
    P = 37  # Not a multiple of 16
    B = 16

    q = torch.randn(H, C, D)
    k = torch.randn(H, P + C, D)
    v = torch.randn(H, P + C, D)

    ref = monolithic_attention(q, k, v, P, C)

    num_blocks = (P + C + B - 1) // B
    k_pool = torch.zeros(num_blocks, H, B, D)
    v_pool = torch.zeros(num_blocks, H, B, D)
    block_table = list(range(num_blocks))
    for i in range(P + C):
        blk = i // B
        off = i % B
        k_pool[blk, :, off, :] = k[:, i, :]
        v_pool[blk, :, off, :] = v[:, i, :]

    out = chunked_attention(q, k_pool, v_pool, block_table, P, C, B)
    rel = _relative_l2(ref, out)
    assert rel < 1e-5, f"Unaligned prefix boundary mismatch: {rel}"


def test_masks_are_correct():
    """Verify causal masking: each query i in chunk sees 0..P+i."""
    torch.manual_seed(0)
    H, C, D = 1, 4, 1
    P = 2
    B = 1
    q = torch.ones(H, C, D)
    k = torch.ones(H, P + C, D)
    v = torch.ones(H, P + C, D)
    v[:, :, :] = torch.arange(P + C).view(1, P + C, 1).float()

    ref = monolithic_attention(q, k, v, P, C)

    # Blocked path: B=1 means each token is its own block.
    num_blocks = P + C
    k_pool = torch.zeros(num_blocks, H, B, D)
    v_pool = torch.zeros(num_blocks, H, B, D)
    block_table = list(range(num_blocks))
    for i in range(P + C):
        k_pool[i, :, 0, :] = k[:, i, :]
        v_pool[i, :, 0, :] = v[:, i, :]

    out = chunked_attention(q, k_pool, v_pool, block_table, P, C, B)
    assert torch.allclose(ref, out, atol=1e-5)


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_chunked_prefill_under_4gib():
    res = isolated_run(_workload_chunked_prefill, timeout_s=120.0)
    assert res.result["C"] == 128
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_chunked_prefill() -> dict:
    H, D, P = 4, 64, 1024
    C = 128
    B = 16

    q = torch.randn(H, C, D)
    num_blocks = (P + C + B - 1) // B
    k_pool = torch.zeros(num_blocks, H, B, D)
    v_pool = torch.zeros(num_blocks, H, B, D)
    block_table = list(range(num_blocks))

    out = chunked_attention(q, k_pool, v_pool, block_table, P, C, B)
    return {"C": C}