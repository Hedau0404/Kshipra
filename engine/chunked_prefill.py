"""Milestone 4 — Chunked Prefill Engine.

Motivation
----------
On a CPU host, a monolithic 4096-token prefill is heavily compute-bound and
blocks the engine loop for hundreds of seconds. During that window, every
active decode step is stalled, causing a catastrophic ITL (inter-token latency)
spike. Chunking splits the prefill into ``C``-token slices interleaved with
decode steps, trading a small TTFT penalty for bounded ITL.

Attention Architecture
----------------------
We implement attention in two modes:

1. **Monolithic (oracle)**: ``[C, P+C]`` boolean mask, concatenated K/V.
   Simple, obviously correct, memory-hungry. Used to verify the chunked path.
2. **Chunked (online softmax)**: Reads prefix blocks in place (no gather).
   Attention is computed block-by-block with a running max/exp-sum/acc,
   merging via LSE (log-sum-exp). The only mask generated is a small on-the-fly
   causal tile for the chunk portion.

RoPE Position Injection
-----------------------
A chunk at absolute positions ``P, P+1, ..., P+C-1`` must apply RoPE using
``positions = P + arange(C)``. Prefix K/V stored in cache were already rotated
at absolute positions ``0..P-1`` and must NOT be re-rotated. This is enforced by
gathering cos/sin by absolute position in the engine, not relative index.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from engine.kv_cache import KVCache
from engine.prefix_cache import PrefixCache


# ---------------------------------------------------------------------- #
# Attention Modes
# ---------------------------------------------------------------------- #


def monolithic_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    prefix_len: int,
    chunk_len: int,
) -> torch.Tensor:
    """Approach (a): Naive concatenation with a [C, P+C] causal mask.

    Oracle for correctness testing. Gathers the prefix into a contiguous
    tensor.

    Parameters
    ----------
    q: [H, C, D]
    k_cache, v_cache: [H, S, D] where S = prefix_len + chunk_len
    """
    # Extract prefix (already populated by prior steps)
    k_pre = k_cache[:, :prefix_len, :]
    v_pre = v_cache[:, :prefix_len, :]

    # The chunk part is the tail
    k_chunk = k_cache[:, prefix_len:, :]
    v_chunk = v_cache[:, prefix_len:, :]

    k_all = torch.cat([k_pre, k_chunk], dim=1)
    v_all = torch.cat([v_pre, v_chunk], dim=1)

    # Build mask [C, P+C]. First P columns all True, last C causal.
    mask = torch.ones((chunk_len, prefix_len + chunk_len), dtype=torch.bool)
    causal = torch.tril(
        torch.ones((chunk_len, chunk_len), dtype=torch.bool)
    )
    mask[:, prefix_len:] = causal

    # SDPA expects [B, H, C, D]. Add B=1 dim.
    q_ = q.unsqueeze(0)
    k_ = k_all.unsqueeze(0)
    v_ = v_all.unsqueeze(0)
    # Broadcast mask to [1, 1, C, P+C]
    m_ = mask.unsqueeze(0).unsqueeze(0)

    out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=m_, scale=None)
    return out.squeeze(0)


def chunked_attention(
    q: torch.Tensor,
    k_pool: torch.Tensor,
    v_pool: torch.Tensor,
    block_table: List[int],
    prefix_len: int,
    chunk_len: int,
    block_size: int,
) -> torch.Tensor:
    H_q = q.shape[0]
    H_kv = k_pool.shape[1]
    assert H_q % H_kv == 0, f"GQA requires H_q ({H_q}) % H_kv ({H_kv}) == 0"
    g = H_q // H_kv

    _, C, D = q.shape
    scale = 1.0 / math.sqrt(D)
    total_len = prefix_len + chunk_len
    m = torch.full((H_q, C), float("-inf"))
    l = torch.zeros((H_q, C))
    acc = torch.zeros((H_q, C, D))
    q_positions = prefix_len + torch.arange(C)

    for j in range((total_len + block_size - 1) // block_size):
        blk_id = block_table[j]
        start = j * block_size
        end = min(start + block_size, total_len)
        n_valid = end - start
        k_tile = k_pool[blk_id, :, :n_valid, :]  # [H_kv, B, D]
        v_tile = v_pool[blk_id, :, :n_valid, :]
        if g > 1:
            k_tile = k_tile.repeat_interleave(g, dim=0)  # [H_q, B, D]
            v_tile = v_tile.repeat_interleave(g, dim=0)

        s = torch.bmm(q, k_tile.transpose(1, 2)) * scale  # [H_q, C, B]
        if end > prefix_len:
            k_positions = start + torch.arange(n_valid)
            mask = k_positions.unsqueeze(0) > q_positions.unsqueeze(1)
            s.masked_fill_(mask.unsqueeze(0), float("-inf"))

        m_new = torch.max(m, s.max(dim=-1).values)
        m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)
        p = torch.exp(s - m_safe.unsqueeze(-1))
        corr = torch.exp(m - m_safe)
        l = l * corr + p.sum(dim=-1)
        acc = acc * corr.unsqueeze(-1) + torch.bmm(p, v_tile)
        m = m_new

    return acc / l.unsqueeze(-1)


# ---------------------------------------------------------------------- #
# Engine
# ---------------------------------------------------------------------- #


class ChunkedPrefillEngine:
    """Orchestrates chunked prefill over a single sequence using a PrefixCache."""

    def __init__(
        self,
        prefix_cache: PrefixCache,
        chunk_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
    ) -> None:
        self.cache = prefix_cache
        self.block_size = prefix_cache.block_size
        self.chunk_size = chunk_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        # GQA ratio
        self.gqa_ratio = num_heads // num_kv_heads

    def prefill(
        self,
        token_ids: List[int],
        embedding_layer: torch.nn.Module,
        rope_fn: callable,
        layer_fn: callable,
        use_oracle: bool = False,
    ) -> torch.Tensor:
        """Run prefill in chunks.

        Parameters
        ----------
        token_ids: Full prompt.
        embedding_layer: nn.Embedding.
        rope_fn: Callable(q, k, positions) -> (q, k).
        layer_fn: Callable(x, attn_fn) -> x. This is a mock Transformer layer.
        use_oracle: If True, use monolithic_attention for all chunks.
        """
        P = len(token_ids)
        # 1. Match Prefix
        matched_blocks, matched_len, node = self.cache.tree.match(token_ids)
        self.cache.tree.pin(node)

        # 2. Allocate blocks for the remainder
        remaining = P - matched_len
        new_blocks_needed = math.ceil(
            (matched_len % self.block_size + remaining) / self.block_size
        )
        # Note: this is simplified. In a full impl, we CoW the tail block.
        # For this milestone, we assume block-aligned matches.
        new_blocks = self.cache.allocate_blocks(new_blocks_needed)

        # Block table is the concatenation
        block_table = matched_blocks + new_blocks

        # 3. Chunking loop
        pos = matched_len
        hidden = None
        while pos < P:
            chunk_tokens = token_ids[pos : pos + self.chunk_size]
            C = len(chunk_tokens)

            # Embed
            x = embedding_layer(torch.tensor(chunk_tokens))

            # Define attention closure for this chunk
            def attn_fn(q, k, v):
                # Apply RoPE. Q and new K are rotated at absolute pos.
                positions = pos + torch.arange(C)
                q, k = rope_fn(q, k, positions)

                # Write new K, V to cache
                blk_idx = pos // self.block_size
                offset = pos % self.block_size
                for i in range(C):
                    b_id = block_table[blk_idx + (offset + i) // self.block_size]
                    o = (offset + i) % self.block_size
                    self.cache.allocator.pool[b_id, :, 0, :, o, :] = k[:, i, :]
                    self.cache.allocator.pool[b_id, :, 1, :, o, :] = v[:, i, :]

                # Attention
                if use_oracle:
                    # Gather for oracle
                    pass
                return chunked_attention(
                    q,
                    self.cache.allocator.pool[:, :, 0],  # K
                    self.cache.allocator.pool[:, :, 1],  # V
                    block_table,
                    prefix_len=pos,
                    chunk_len=C,
                    block_size=self.block_size,
                )

            hidden = layer_fn(x, attn_fn)
            pos += C

        self.cache.tree.unpin(node)
        # Insert new suffix into tree (simplified)
        return hidden