"""Milestone 1 — Baseline KV Cache.

Tensor layout policy
--------------------
The cache is stored as two tensors (K and V), each shaped::

    [num_layers, num_kv_heads, max_context_len, head_dim]

This is the separated form of layout ``[2, L, H_kv, S, D]``; K and V live in
distinct buffers for ergonomic indexing in the decode kernel. Separation has
identical byte footprint to the stacked form — the choice is purely about
indexing ergonomics.

Why this layout
~~~~~~~~~~~~~~~
For each KV head, consecutive tokens are separated by ``head_dim`` elements
(stride = ``head_dim`` in the second-to-last axis). With ``head_dim = 128``
and FP16, that is a 256-byte row per token — one cache-line-pair per step of
the per-head GEMV decode loop, which the hardware prefetcher streams cleanly.
The innermost axis (``head_dim``) has stride 1 element (2 bytes in FP16),
which is the precondition for AVX2/AVX-512/NEON vectorization.

Compare to layout ``[L, 2, S, H_kv, D]`` where consecutive tokens for a fixed
head are ``H_kv * D`` elements apart (2048 B in the Llama-3-8B example). Both
layouts have innermost stride 1; the *policy* choice depends on the traversal
order of the decode kernel. We pick per-head-major because our decode kernel
(in Milestones 5-7) is a per-head GEMV. Note: vLLM's paged KV blocks use the
opposite (position-major) convention because their kernel batches over heads
per position. This is a policy decision contingent on the kernel.

Memory complexity
~~~~~~~~~~~~~~~~~
Footprint is preallocated and fixed at construction::

    bytes = 2 * L * H_kv * max_context_len * D * itemsize(dtype)

There is no per-step allocation: :meth:`KVCache.append` copies into
preallocated slices, and :meth:`KVCache.get` returns *views* into those
slices. This is deliberate — per-step allocation would thrash glibc's
per-thread arenas (M_TRIM_THRESHOLD defaults to 128 KiB, so freed
medium-sized chunks are retained, not returned to the kernel) and destroy
the L3 residency we are trying to preserve across decode steps.

Hardware trade-off
~~~~~~~~~~~~~~~~~~
On an 8 GB CPU host, DDR bandwidth (40-60 GB/s typical) is the binding
constraint, not FLOPs. At S=1024 for a Llama-3-8B GQA config, the KV cache is
128 MiB; scanning it once per decode step at ~50 GB/s costs ~2.7 ms of pure
memory movement before any compute. Every design choice here is subordinated
to making that scan a clean, prefetchable, vectorizable stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


class KVCacheOverflowError(RuntimeError):
    """Raised when an :meth:`KVCache.append` would exceed ``max_context_len``."""


def _dtype_itemsize(dtype: torch.dtype) -> int:
    """Return the byte width of ``dtype``.

    ``torch.dtype.itemsize`` exists in modern PyTorch but constructing an
    empty tensor is unconditionally portable across 1.x/2.x.
    """
    return torch.empty(0, dtype=dtype).element_size()


@dataclass(frozen=True)
class KVConfig:
    """Static description of a KV cache allocation.

    Attributes
    ----------
    num_layers:
        Number of transformer blocks whose K/V tensors this cache holds.
    num_kv_heads:
        Number of *key/value* heads (GQA). For MHA this equals
        ``num_attention_heads``; for GQA (Llama-3-8B: 32 query heads, 8 KV
        heads) it is strictly smaller. Using the query-head count here
        silently inflates the cache by ``num_attention_heads /
        num_kv_heads`` — 4x for Llama-3-8B — and quadruples per-decode-step
        KV read traffic.
    head_dim:
        Per-head feature width. For Llama-family models this is
        ``hidden_size // num_attention_heads`` and is generally *not*
        equal to ``hidden_size // num_kv_heads``.
    max_context_len:
        Hard ceiling on the number of tokens the cache will ever hold,
        including the prompt. This is what bounds peak RSS: the allocation
        is ``max_context_len``-shaped from construction, not grown on
        demand.
    dtype:
        Element type. FP16 halves DDR traffic relative to FP32 and maps to
        the AVX-512-FP16 / NEON-FP16 vector units. FP32 is useful for
        numerical-equivalence tests.
    device:
        Target device. Defaults to CPU. All ops are device-agnostic so a
        future CUDA build can pass ``torch.device("cuda")`` unchanged.
    """

    num_layers: int
    num_kv_heads: int
    head_dim: int
    max_context_len: int
    dtype: torch.dtype = torch.float16
    device: torch.device = torch.device("cpu")

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {self.num_layers}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {self.head_dim}")
        if self.max_context_len < 0:
            raise ValueError(
                f"max_context_len must be >= 0, got {self.max_context_len}"
            )

    def bytes_per_token(self) -> int:
        """Bytes of KV storage consumed by one additional token.

        ``2`` accounts for both K and V; ``itemsize`` converts elements to
        bytes. This is the *unique* payload scanned by a decode step, and
        therefore the per-step memory-traffic floor for the cache alone.
        """
        return (
            2
            * self.num_layers
            * self.num_kv_heads
            * self.head_dim
            * _dtype_itemsize(self.dtype)
        )

    def total_bytes(self) -> int:
        """Preallocated footprint of the cache at full ``max_context_len``."""
        return self.bytes_per_token() * self.max_context_len


class KVCache:
    """Preallocated, single-sequence KV cache.

    Storage layout (per buffer)::

        k_cache : [num_layers, num_kv_heads, max_context_len, head_dim]
        v_cache : [num_layers, num_kv_heads, max_context_len, head_dim]

    Memory complexity: ``O(L * H_kv * max_context_len * D)`` bytes,
    allocated once at construction. No per-append allocation, no
    fragmentation, no glibc arena churn.

    Thread-safety: not thread-safe. The continuous-batching scheduler in
    Milestone 5 is single-threaded per engine instance by design.
    """

    __slots__ = ("config", "_k", "_v", "_length")

    def __init__(self, config: KVConfig) -> None:
        self.config = config
        shape = (
            config.num_layers,
            config.num_kv_heads,
            config.max_context_len,
            config.head_dim,
        )
        self._k = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self._v = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self._length = 0

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    @property
    def length(self) -> int:
        """Number of token positions currently occupied."""
        return self._length

    @property
    def capacity(self) -> int:
        """Maximum number of token positions (``config.max_context_len``)."""
        return self.config.max_context_len

    def remaining(self) -> int:
        """Slots still available before :class:`KVCacheOverflowError`."""
        return self.capacity - self._length

    def mem_bytes(self) -> int:
        """Actual resident bytes of the two backing buffers."""
        return (
            self._k.numel() * self._k.element_size()
            + self._v.numel() * self._v.element_size()
        )

    def device(self) -> torch.device:
        return self._k.device

    def dtype(self) -> torch.dtype:
        return self._k.dtype

    # ------------------------------------------------------------------ #
    # Mutation
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Drop all cached tokens without freeing memory.

        Intended for slot reuse in continuous batching (Milestone 5): the
        backing buffers are retained so that the allocator is never touched
        mid-serve, which is what preserves the L3-resident working set.
        """
        self._length = 0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> int:
        """Copy ``T`` new K/V vectors into the cache.

        Parameters
        ----------
        k, v:
            Tensors of shape ``[num_layers, num_kv_heads, T, head_dim]``
            with the same dtype and device as the cache. ``T`` may be zero,
            in which case this call is a no-op.

        Returns
        -------
        int
            The new ``length`` after the append.

        Raises
        ------
        KVCacheOverflowError
            If ``length + T > max_context_len``.
        TypeError / ValueError
            If shapes, dtype, or device do not match the cache config.
        """
        if k.shape != v.shape:
            raise ValueError(
                f"k and v must have identical shape, got "
                f"{tuple(k.shape)} vs {tuple(v.shape)}"
            )
        if k.dim() != 4:
            raise ValueError(
                f"k/v must be 4-D [L, H_kv, T, D], got {k.dim()}-D"
            )
        exp_l, exp_h = self.config.num_layers, self.config.num_kv_heads
        if k.shape[0] != exp_l or k.shape[1] != exp_h:
            raise ValueError(
                f"k/v leading dims must be ({exp_l}, {exp_h}), got "
                f"{(k.shape[0], k.shape[1])}"
            )
        if k.shape[3] != self.config.head_dim:
            raise ValueError(
                f"k/v head_dim must be {self.config.head_dim}, got {k.shape[3]}"
            )
        if k.dtype != self._k.dtype:
            raise TypeError(
                f"k dtype {k.dtype} does not match cache dtype {self._k.dtype}"
            )
        if v.dtype != self._k.dtype:
            raise TypeError(
                f"v dtype {v.dtype} does not match cache dtype {self._k.dtype}"
            )
        if k.device != self._k.device:
            raise ValueError(
                f"k device {k.device} does not match cache device {self._k.device}"
            )
        if v.device != self._k.device:
            raise ValueError(
                f"v device {v.device} does not match cache device {self._k.device}"
            )

        t = k.shape[2]
        if t == 0:
            return self._length
        if self._length + t > self.capacity:
            raise KVCacheOverflowError(
                f"append of {t} tokens would exceed capacity "
                f"({self._length} + {t} > {self.capacity})"
            )

        lo = self._length
        hi = lo + t
        # In-place copy into preallocated slices: no allocator traffic, no
        # extra resident pages. `copy_` is a contiguous memcpy on the inner
        # two axes because head_dim is stride 1.
        self._k[:, :, lo:hi, :].copy_(k)
        self._v[:, :, lo:hi, :].copy_(v)
        self._length = hi
        return self._length

    # ------------------------------------------------------------------ #
    # Access
    # ------------------------------------------------------------------ #

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return *views* of the populated prefix.

        Returns
        -------
        (k, v) : tuple of tensors
            Each of shape ``[num_layers, num_kv_heads, length, head_dim]``.

        Notes
        -----
        The returned tensors are views into the cache's backing buffers,
        not copies. Subsequent :meth:`append` calls will be visible through
        previously returned views, and callers must not mutate them unless
        they intend to mutate the cache. This is intentional: it avoids a
        per-step ``L * H_kv * S * D`` copy that would dominate the decode
        bandwidth budget.
        """
        return self._k[:, :, : self._length, :], self._v[:, :, : self._length, :]

    def __len__(self) -> int:
        return self._length

    def __repr__(self) -> str:
        return (
            f"KVCache(length={self._length}/{self.capacity}, "
            f"shape={tuple(self._k.shape)}, dtype={self._k.dtype}, "
            f"device={self._k.device}, bytes={self.mem_bytes()})"
        )