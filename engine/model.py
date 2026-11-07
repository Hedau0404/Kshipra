"""Kshipra model binding: real transformer forward through our KV pool.

This is the missing integration layer. It takes a real HuggingFace Llama or
Qwen2 model and routes every forward pass through Kshipra primitives:

    embed -> [layer: (LN -> QKV -> RoPE -> WRITE-to-pool -> chunked_attention
              -> O-proj) -> (LN -> MLP) ] -> final_norm -> lm_head

The KV cache lives in a BlockAllocator pool shared with PrefixCache /
ContinuousEngine. This class owns the per-layer compute; the caller owns
sequence management.

Optionally replaces all nn.Linear projections with QuantizedLinear (INT4
per-group G=128, or INT8 per-channel) to test numerical behavior and
measure real weight-footprint reduction.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from engine.chunked_prefill import chunked_attention
from engine.quantization import QuantizedLinear


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _compute_rope(
    head_dim: int, position_ids: torch.Tensor, theta: float, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute cos, sin for RoPE. Returns [B, L, head_dim] each."""
    inv_freq = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    )
    freqs = torch.einsum("bi,j->bij", position_ids.float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """q, k: [B, H, L, D]. cos, sin: [B, L, D] -> broadcast to [B, 1, L, D]."""
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


class KshipraModel:
    """Runs a real HF Llama/Qwen2 checkpoint through the Kshipra pipeline."""

    def __init__(
        self,
        hf_model: nn.Module,
        device: torch.device = torch.device("cpu"),
        quantize: Optional[str] = None,  # None | "int8" | "int4"
    ) -> None:
        self.model = hf_model.to(device).eval()
        self.device = device
        cfg = hf_model.config
        self.n_heads = int(cfg.num_attention_heads)
        self.n_kv_heads = int(getattr(cfg, "num_key_value_heads", self.n_heads))
        self.head_dim = int(getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads))
        self.n_layers = int(cfg.num_hidden_layers)
        self.rope_theta = float(getattr(cfg, "rope_theta", 10000.0))
        self.vocab_size = int(cfg.vocab_size)

        if quantize is not None:
            self._quantize_all_linears(quantize)

    def _quantize_all_linears(self, scheme: str) -> None:
        """Swap nn.Linear -> QuantizedLinear in every transformer block."""
        for layer in self.model.model.layers:
            attn = layer.self_attn
            attn.q_proj = QuantizedLinear(attn.q_proj, scheme=scheme, group_size=128)
            attn.k_proj = QuantizedLinear(attn.k_proj, scheme=scheme, group_size=128)
            attn.v_proj = QuantizedLinear(attn.v_proj, scheme=scheme, group_size=128)
            attn.o_proj = QuantizedLinear(attn.o_proj, scheme=scheme, group_size=128)
            mlp = layer.mlp
            mlp.gate_proj = QuantizedLinear(mlp.gate_proj, scheme=scheme, group_size=128)
            mlp.up_proj = QuantizedLinear(mlp.up_proj, scheme=scheme, group_size=128)
            mlp.down_proj = QuantizedLinear(mlp.down_proj, scheme=scheme, group_size=128)

    def forward(
        self,
        input_ids: torch.Tensor,           # [1, L]
        position_ids: torch.Tensor,        # [1, L]
        pool: torch.Tensor,                # [num_blocks, L_layers, 2, H_kv, B, D]
        block_table: List[int],
        slot_mapping: List[int],           # [L] physical slot per token
        prefix_len: int,                   # tokens already in pool
    ) -> torch.Tensor:
        """Returns logits [1, L, vocab]. Writes K/V for the chunk to pool."""
        assert input_ids.shape[0] == 1, "KshipraModel expects batch=1"
        B, L = input_ids.shape
        block_size = pool.shape[4]

        hidden = self.model.model.embed_tokens(input_ids)  # [1, L, D]
        cos, sin = _compute_rope(self.head_dim, position_ids, self.rope_theta, self.device)
        # cos, sin: [1, L, D]

        for i, layer in enumerate(self.model.model.layers):
            residual = hidden
            hidden = layer.input_layernorm(hidden)

            q = layer.self_attn.q_proj(hidden)  # [1, L, H_q * D]
            k = layer.self_attn.k_proj(hidden)  # [1, L, H_kv * D]
            v = layer.self_attn.v_proj(hidden)  # [1, L, H_kv * D]
            q = q.view(1, L, self.n_heads, self.head_dim).transpose(1, 2)   # [1, H_q, L, D]
            k = k.view(1, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(1, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos, sin)

            # Write K, V into the pool at the mapped physical slots.
            for t in range(L):
                slot = slot_mapping[t]
                blk = slot // block_size
                off = slot % block_size
                pool[blk, i, 0, :, off, :] = k[0, :, t, :]
                pool[blk, i, 1, :, off, :] = v[0, :, t, :]

            # Chunked attention reads prefix + this chunk (post-RoPE) from the pool.
            attn = chunked_attention(
                q[0],              # [H_q, L, D]
                pool[:, i, 0],     # [num_blocks, H_kv, B, D]
                pool[:, i, 1],
                block_table,
                prefix_len=prefix_len,
                chunk_len=L,
                block_size=block_size,
            )  # [H_q, L, D]
            attn = attn.transpose(0, 1).reshape(1, L, self.n_heads * self.head_dim)
            hidden = layer.self_attn.o_proj(attn)
            hidden = residual + hidden

            residual = hidden
            hidden = layer.post_attention_layernorm(hidden)
            hidden = layer.mlp(hidden)
            hidden = residual + hidden

        hidden = self.model.model.norm(hidden)
        return self.model.lm_head(hidden)

    @torch.inference_mode()
    def generate(
        self,
        prompt_ids: List[int],
        max_new_tokens: int = 20,
        eos_token_id: Optional[int] = None,
        block_size: int = 64,
    ) -> List[int]:
        """Greedy generation through the Kshipra pipeline. Returns new token ids only."""
        from engine.prefix_cache import BlockAllocator

        L_prompt = len(prompt_ids)
        total_capacity = L_prompt + max_new_tokens
        num_blocks = (total_capacity + block_size - 1) // block_size
        alloc = BlockAllocator(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=self.n_layers,
            num_kv_heads=self.n_kv_heads,
            head_dim=self.head_dim,
            dtype=next(self.model.parameters()).dtype,
            device=self.device,
        )
        block_table = list(range(num_blocks))
        pool = alloc.pool

        # Prefill: process prompt in one forward.
        input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        positions = torch.arange(L_prompt, device=self.device).unsqueeze(0)
        slots = [(p // block_size) * block_size + (p % block_size) for p in range(L_prompt)]
        # Note: block_table[i] is the block id; physical slot in pool is block_table[i]*block_size + offset.
        slot_mapping = [block_table[s // block_size] * block_size + (s % block_size) for s in slots]

        logits = self.forward(input_ids, positions, pool, block_table, slot_mapping, prefix_len=0)
        next_tok = int(logits[0, -1].argmax().item())
        generated = [next_tok]

        if eos_token_id is not None and next_tok == eos_token_id:
            return generated

        # Decode: one token at a time.
        for step in range(1, max_new_tokens):
            pos = L_prompt + step - 1
            slot = block_table[pos // block_size] * block_size + (pos % block_size)
            cur_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            cur_pos = torch.tensor([[pos]], dtype=torch.long, device=self.device)
            logits = self.forward(cur_ids, cur_pos, pool, block_table, [slot], prefix_len=pos)
            next_tok = int(logits[0, -1].argmax().item())
            generated.append(next_tok)
            if eos_token_id is not None and next_tok == eos_token_id:
                break

        return generated