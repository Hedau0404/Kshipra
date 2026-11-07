"""Unit tests for engine.kv_cache (Milestone 1).

Covers:
* GQA byte arithmetic (Llama-3-8B reference).
* The 4 GiB capacity crossover.
* Zero-length initial state, zero-length append no-op.
* Chunked-append equivalence.
* Overflow, dtype, shape, and head_dim validation.
* View semantics of :meth:`KVCache.get`.
* Isolated peak-RSS bound under 4 GiB.
"""

from __future__ import annotations

import sys

import pytest
import torch

from benchmarks.run_benchmarks import isolated_run
from engine.kv_cache import KVCache, KVCacheOverflowError, KVConfig


# Llama-3-8B GQA shape: 32 query heads, 8 KV heads, head_dim 128, 32 layers.
LLAMA3_8B = dict(num_layers=32, num_kv_heads=8, head_dim=128)
FP16_BYTES = 2


def _llama3_cache(
    max_context_len: int = 1024, dtype: torch.dtype = torch.float16
) -> KVCache:
    return KVCache(
        KVConfig(
            num_layers=LLAMA3_8B["num_layers"],
            num_kv_heads=LLAMA3_8B["num_kv_heads"],
            head_dim=LLAMA3_8B["head_dim"],
            max_context_len=max_context_len,
            dtype=dtype,
            device=torch.device("cpu"),
        )
    )


def _rand_kv(L, H, T, D, dtype=torch.float16, seed=0):
    g = torch.Generator().manual_seed(seed)
    k = torch.randn(L, H, T, D, generator=g, dtype=torch.float32).to(dtype)
    v = torch.randn(L, H, T, D, generator=g, dtype=torch.float32).to(dtype)
    return k, v


# ---------------------------------------------------------------------- #
# Arithmetic
# ---------------------------------------------------------------------- #


def test_bytes_per_token_matches_llama3_gqa():
    cfg = KVConfig(**LLAMA3_8B, max_context_len=1024, dtype=torch.float16)
    # 2 (K,V) * 32 layers * 8 kv_heads * 128 head_dim * 2 bytes/FP16
    assert cfg.bytes_per_token() == 2 * 32 * 8 * 128 * FP16_BYTES
    assert cfg.bytes_per_token() == 131_072  # 128 KiB per token
    assert cfg.total_bytes() == 128 * 1024 * 1024  # 128 MiB at S=1024


def test_4gib_crossover_arithmetic():
    per_tok = 2 * 32 * 8 * 128 * FP16_BYTES  # 131,072 bytes
    cap = (4 * 1024 ** 3) // per_tok
    assert cap == 32_768

    cfg_below = KVConfig(**LLAMA3_8B, max_context_len=cap - 1, dtype=torch.float16)
    cfg_equal = KVConfig(**LLAMA3_8B, max_context_len=cap, dtype=torch.float16)
    cfg_above = KVConfig(**LLAMA3_8B, max_context_len=cap + 1, dtype=torch.float16)

    assert cfg_below.total_bytes() < 4 * 1024 ** 3
    assert cfg_equal.total_bytes() == 4 * 1024 ** 3
    assert cfg_above.total_bytes() > 4 * 1024 ** 3


def test_config_rejects_bad_shapes():
    with pytest.raises(ValueError):
        KVConfig(num_layers=0, num_kv_heads=8, head_dim=128, max_context_len=1)
    with pytest.raises(ValueError):
        KVConfig(num_layers=1, num_kv_heads=-1, head_dim=128, max_context_len=1)
    with pytest.raises(ValueError):
        KVConfig(num_layers=1, num_kv_heads=1, head_dim=0, max_context_len=1)
    with pytest.raises(ValueError):
        KVConfig(num_layers=1, num_kv_heads=1, head_dim=1, max_context_len=-1)


# ---------------------------------------------------------------------- #
# Zero-length and initial-state edge cases
# ---------------------------------------------------------------------- #


def test_initial_state_is_zero_length():
    cache = _llama3_cache(max_context_len=64)
    assert cache.length == 0
    assert len(cache) == 0
    assert cache.remaining() == 64
    k, v = cache.get()
    assert k.shape == (32, 8, 0, 128)
    assert v.shape == (32, 8, 0, 128)


def test_zero_length_append_is_noop():
    cache = _llama3_cache(max_context_len=8)
    k0 = torch.zeros(32, 8, 0, 128, dtype=torch.float16)
    v0 = torch.zeros(32, 8, 0, 128, dtype=torch.float16)
    assert cache.append(k0, v0) == 0
    assert cache.length == 0
    assert cache.remaining() == 8


# ---------------------------------------------------------------------- #
# Round-trip and chunking
# ---------------------------------------------------------------------- #


def test_append_and_get_roundtrip():
    cache = _llama3_cache(max_context_len=16)
    k1, v1 = _rand_kv(32, 8, 5, 128, seed=1)
    cache.append(k1, v1)
    assert cache.length == 5

    k_out, v_out = cache.get()
    assert k_out.shape == (32, 8, 5, 128)
    assert torch.equal(k_out, k1)
    assert torch.equal(v_out, v1)

    k2, v2 = _rand_kv(32, 8, 3, 128, seed=2)
    cache.append(k2, v2)
    assert cache.length == 8

    k_out, v_out = cache.get()
    assert torch.equal(k_out[:, :, 5:8, :], k2)
    assert torch.equal(v_out[:, :, 5:8, :], v2)


def test_chunked_append_matches_single_append():
    full_k, full_v = _rand_kv(32, 8, 10, 128, seed=3)

    ref = _llama3_cache(max_context_len=10)
    ref.append(full_k, full_v)

    chunked = _llama3_cache(max_context_len=10)
    chunked.append(full_k[:, :, 0:3, :], full_v[:, :, 0:3, :])
    # Explicit zero-length append in the middle must be a no-op.
    chunked.append(full_k[:, :, 3:3, :], full_v[:, :, 3:3, :])
    chunked.append(full_k[:, :, 3:7, :], full_v[:, :, 3:7, :])
    chunked.append(full_k[:, :, 7:10, :], full_v[:, :, 7:10, :])

    rk, rv = ref.get()
    ck, cv = chunked.get()
    assert torch.equal(rk, ck)
    assert torch.equal(rv, cv)
    assert chunked.length == ref.length == 10


# ---------------------------------------------------------------------- #
# Overflow and slot reuse
# ---------------------------------------------------------------------- #


def test_overflow_raises_cleanly():
    cache = _llama3_cache(max_context_len=4)
    k, v = _rand_kv(32, 8, 4, 128)
    cache.append(k, v)
    assert cache.length == 4
    assert cache.remaining() == 0

    extra_k, extra_v = _rand_kv(32, 8, 1, 128, seed=99)
    with pytest.raises(KVCacheOverflowError):
        cache.append(extra_k, extra_v)
    # Cache is unchanged after the rejected append.
    assert cache.length == 4


def test_reset_allows_slot_reuse():
    cache = _llama3_cache(max_context_len=4)
    k1, v1 = _rand_kv(32, 8, 4, 128, seed=10)
    cache.append(k1, v1)

    cache.reset()
    assert cache.length == 0
    assert cache.remaining() == 4

    k2, v2 = _rand_kv(32, 8, 2, 128, seed=11)
    cache.append(k2, v2)
    k_out, v_out = cache.get()
    assert k_out.shape == (32, 8, 2, 128)
    assert torch.equal(k_out, k2)
    assert torch.equal(v_out, v2)


# ---------------------------------------------------------------------- #
# Validation
# ---------------------------------------------------------------------- #


def test_shape_mismatch_raises():
    cache = _llama3_cache(max_context_len=8)
    k = torch.zeros(32, 8, 2, 128, dtype=torch.float16)
    v = torch.zeros(32, 8, 2, 64, dtype=torch.float16)
    with pytest.raises(ValueError):
        cache.append(k, v)


def test_dtype_mismatch_raises():
    cache = _llama3_cache(max_context_len=8)
    k = torch.zeros(32, 8, 2, 128, dtype=torch.float32)
    v = torch.zeros(32, 8, 2, 128, dtype=torch.float32)
    with pytest.raises(TypeError):
        cache.append(k, v)


def test_head_dim_mismatch_raises():
    cache = _llama3_cache(max_context_len=8)
    k = torch.zeros(32, 8, 2, 64, dtype=torch.float16)
    v = torch.zeros(32, 8, 2, 64, dtype=torch.float16)
    with pytest.raises(ValueError):
        cache.append(k, v)


def test_rank_mismatch_raises():
    cache = _llama3_cache(max_context_len=8)
    k = torch.zeros(32, 8, 128, dtype=torch.float16)
    v = torch.zeros(32, 8, 128, dtype=torch.float16)
    with pytest.raises(ValueError):
        cache.append(k, v)


# ---------------------------------------------------------------------- #
# View semantics
# ---------------------------------------------------------------------- #


def test_get_returns_views_not_copies():
    cache = _llama3_cache(max_context_len=8)
    k, v = _rand_kv(32, 8, 4, 128, seed=20)
    cache.append(k, v)

    k_view, _ = cache.get()
    k_view[:, :, 0, 0] = 42.0
    k_view2, _ = cache.get()
    assert torch.all(k_view2[:, :, 0, 0] == 42.0)


def test_mem_bytes_matches_config_total():
    cache = _llama3_cache(max_context_len=1024)
    assert cache.mem_bytes() == cache.config.total_bytes()
    assert cache.mem_bytes() == 128 * 1024 * 1024


# ---------------------------------------------------------------------- #
# Isolated memory bound
# ---------------------------------------------------------------------- #


def _workload_kv_cache_512mib() -> dict:
    """Allocate a 512 MiB KV cache and force every page resident.

    ``torch.zeros`` for large buffers is mmap'd and lazily zero-filled by
    the kernel, so RSS would not reflect the nominal allocation until
    touched. We append one full 4096-token slab to force residency.
    """
    cfg = KVConfig(
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        max_context_len=4096,
        dtype=torch.float16,
        device=torch.device("cpu"),
    )
    cache = KVCache(cfg)
    # 32 * 8 * 4096 * 128 * 2 = 256 MiB per tensor; cache holds both K and V.
    k = torch.empty(32, 8, 4096, 128, dtype=torch.float16)
    v = torch.empty_like(k)
    cache.append(k, v)
    del k, v  # tempts are munmap'd by glibc; cache retains the resident pages
    return {
        "bytes": cache.mem_bytes(),
        "length": cache.length,
        "capacity": cache.capacity,
    }


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_kv_cache_stays_under_4gib_peak_rss():
    res = isolated_run(_workload_kv_cache_512mib, timeout_s=120.0)
    assert res.result["bytes"] == 512 * 1024 * 1024
    assert res.result["length"] == 4096
    assert res.result["capacity"] == 4096
    assert res.peak_bytes < 4 * 1024 ** 3, (
        f"peak RSS {res.peak_gib:.3f} GiB exceeded 4 GiB ceiling; "
        f"baseline was {res.baseline_bytes / 1024**2:.1f} MiB, "
        f"delta {res.delta_mib:.1f} MiB"
    )