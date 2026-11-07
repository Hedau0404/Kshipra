"""Milestone 6 — Adaptive Speculative Decoder (CPU-optimized).

This module implements exact speculative decoding for linear draft chains.
Tree speculation is omitted because the compute-bound verification cost
on a CPU (15 rows crossing the 2.6-row ridge) makes it strictly slower
than a linear chain.

Key Mechanics
-------------
1. **Adaptive K**: K is selected dynamically (0 to max_K) based on a
   survival probability estimate. If the expected acceptance rate drops,
   K shrinks to avoid wasted compute.
2. **Exact Sampling**: Uses rejection sampling with residual resampling.
   Accepts draft token d_i with probability min(1, p(d_i)/q(d_i)).
   On rejection at index i, samples from max(0, p - q) normalized.
3. **Draft KV as Best-Effort**: The draft model has a separate, smaller
   block arena. If it runs out of blocks, we degrade to K=0 (standard
   decoding) rather than blocking the request.
4. **Verification**: Verification reuses the M4 chunked prefill kernel,
   processing `[y, d_1, ..., d_K]` as a chunk of `K+1` tokens.

Shape contract
--------------
``model.forward(input_ids[B, L])`` returns logits ``[B, L, vocab]``: row ``i``
is the distribution over the token that FOLLOWS position ``i``. Every
consumer therefore takes the row it needs:

* sampling the next token after a sequence  -> ``logits[:, -1, :]``
* verifying ``d_1..d_K`` appended to a prompt of length ``P`` -> the LAST
  ``K + 1`` rows (positions ``P-1 .. P+K-1``): row ``i`` predicts ``d_{i+1}``
  and row ``K`` is the bonus-token distribution.

Change log for this revision (search for [FIX n]):
* [FIX 1]  MockModel.forward returns [B, L, vocab] (causal, per position).
* [FIX 2a] Draft loop samples from the LAST position's logits.
* [FIX 2b] Verification takes the LAST K+1 rows, not the first K.
* [FIX 3]  k == 0 fallback also samples from the LAST position's logits.
           (Not in the original fix write-up: FIX 1 broke this path.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


class MockModel:
    """Mock model for testing. Generates deterministic logits."""

    def __init__(self, vocab_size: int = 100, seed: int = 42) -> None:
        self.vocab_size = vocab_size
        self.generator = torch.Generator().manual_seed(seed)

    # [FIX 1] was: logits = torch.zeros(B, self.vocab_size)   # [B, vocab]
    #
    # The old forward returned one row per *batch element* and ignored the
    # sequence dimension, but step() (like any real transformer consumer)
    # expects one row per *token position*. Indexing a [B, vocab] result as
    # if it were [B, L, vocab] is what produced the IndexError. This version
    # returns [B, L, vocab]. (Already present in the file as uploaded.)
    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor = None) -> torch.Tensor:
        """Mock forward returning per-token logits ``[B, L, vocab]``.

        Logits at position ``i`` depend only on ``input_ids[:i+1]`` so the
        mock is *causal* — adding future tokens cannot change earlier
        predictions. This is required for the perfect-draft test to be
        meaningful: two models with the same seed must agree at every
        prefix, not just at the very last position.
        """
        B, L = input_ids.shape
        logits = torch.zeros(B, L, self.vocab_size)
        for b in range(B):
            for i in range(L):
                h = hash(tuple(input_ids[b, : i + 1].tolist()))
                g = torch.Generator().manual_seed(h % (2 ** 31))
                logits[b, i] = torch.randn(self.vocab_size, generator=g)
        return logits


@dataclass
class SpeculativeConfig:
    max_k: int = 4
    target_t_tok_s: float = 0.083
    draft_t_tok_s: float = 0.03
    t_ovh_s: float = 0.05
    ridge_rows: float = 2.6


class SpeculativeDecoder:
    def __init__(
        self,
        target_model: MockModel,
        draft_model: MockModel,
        config: SpeculativeConfig,
        target_block_size: int = 64,
        draft_block_size: int = 64,
    ) -> None:
        self.target = target_model
        self.draft = draft_model
        self.cfg = config
        self.target_block_size = target_block_size
        self.draft_block_size = draft_block_size

        # Survival estimator (EMA of acceptance rate per position)
        self.survival_ema = [1.0] * (config.max_k + 1)

    def _estimate_target_time(self, num_rows: int) -> float:
        """Estimate target forward time based on the ridge model."""
        compute_time = num_rows * self.cfg.target_t_tok_s
        bandwidth_time = 0.215  # W / BW
        return self.cfg.t_ovh_s + max(bandwidth_time, compute_time)

    def _select_k(self, draft_probs: torch.Tensor) -> int:
        """Select K adaptively based on survival estimates and cost.

        draft_probs: [K_max, vocab] probabilities from the draft model.
        Returns the optimal K (0 to max_k).
        """
        # 1. Update survival estimate from the draft's top-1 confidence.
        # In a real engine, this is updated post-verification. Here we
        # use the draft's own confidence as a proxy for the current step.
        top1_probs = draft_probs.max(dim=-1).values  # [K_max]

        # 2. Marginal rule: extend K if expected tokens / time increases.
        # Base cost: 1 target step (1 row).
        best_k = 0
        best_rate = 1.0 / self._estimate_target_time(1)

        # Simulate cumulative survival
        s = 1.0
        for k in range(1, self.cfg.max_k + 1):
            s *= float(top1_probs[k - 1].item())
            # Expected tokens for this K
            e_tokens = sum(self.survival_ema[: k + 1])
            # Cost = draft forward (k rows) + target verify (k+1 rows)
            draft_cost = self.cfg.t_ovh_s + k * self.cfg.draft_t_tok_s
            target_cost = self._estimate_target_time(k + 1)
            total_cost = draft_cost + target_cost

            rate = e_tokens / total_cost
            if rate > best_rate:
                best_rate = rate
                best_k = k
            else:
                break

        return best_k

    @torch.no_grad()
    def step(
        self,
        input_ids: torch.Tensor,
        target_kv_cache: torch.Tensor,
        draft_kv_cache: torch.Tensor,
        rng: torch.Generator,
    ) -> Tuple[List[int], torch.Tensor, torch.Tensor]:
        """Run one speculative step.

        Returns (accepted_tokens, updated_target_kv, updated_draft_kv).
        """
        # 1. Draft phase: autoregressive generation of K tokens
        # We mock this by generating K tokens using the draft model.
        k_max = self.cfg.max_k
        draft_tokens = []
        draft_probs_list = []
        current_ids = input_ids.clone()

        for _ in range(k_max):
            logits = self.draft.forward(current_ids, None)       # [1, L, vocab]
            # [FIX 2a] was: probs = F.softmax(logits, dim=-1)    # [1, L, vocab]
            #
            # torch.multinomial only accepts 1-D/2-D probabilities, and
            # draft_probs_list needs one [vocab] row per draft token. The
            # distribution over the NEXT token is the last position's row.
            next_logits = logits[:, -1, :]                       # [1, vocab]
            probs = F.softmax(next_logits, dim=-1)               # [1, vocab]
            token = torch.multinomial(probs, 1, generator=rng)   # [1, 1]
            draft_tokens.append(int(token.item()))
            draft_probs_list.append(probs[0])                    # [vocab]
            current_ids = torch.cat([current_ids, token], dim=1)

        draft_probs = torch.stack(draft_probs_list)  # [K_max, vocab]

        # 2. Adaptive K selection
        k = self._select_k(draft_probs)
        if k == 0:
            # Fallback to standard decoding
            logits = self.target.forward(input_ids, None)        # [1, L, vocab]
            # [FIX 3] was: probs = F.softmax(logits, dim=-1)     # [1, L, vocab]
            #
            # Not in the original fix write-up. Once forward() returns
            # [B, L, vocab] (FIX 1), the unsliced softmax is 3-D and
            # torch.multinomial raises on it. This path is reachable whenever
            # survival_ema has decayed enough that k == 0 wins in _select_k,
            # i.e. exactly the "adaptive K shrinks" case.
            probs = F.softmax(logits[:, -1, :], dim=-1)          # [1, vocab]
            token = torch.multinomial(probs, 1, generator=rng)
            return [int(token.item())], target_kv_cache, draft_kv_cache

        draft_tokens = draft_tokens[:k]
        draft_probs = draft_probs[:k]

        # 3. Verification phase
        # Construct verification batch: [y, d_1, ..., d_K]
        # In a real engine, we run the target model on these K+1 tokens.
        verify_ids = torch.cat(
            [input_ids, torch.tensor([draft_tokens], dtype=torch.long)], dim=1
        )

        # [FIX 2b] was: p_all = F.softmax(target_logits, dim=-1)[0]  # [P+k, vocab]
        #              ... and then p = p_all[i] for i in range(k)
        #
        # verify_ids has P + k tokens (P = prompt length), so target_logits
        # has P + k rows. Row j is the distribution over the token FOLLOWING
        # position j:
        #     row P-1      -> distribution for d_1   (given the prompt)
        #     row P-1+i    -> distribution for d_{i+1}
        #     row P-1+k    -> bonus-token distribution (all k accepted)
        # i.e. exactly the LAST k+1 rows. The old p_all[i] (i = 0..k-1) read
        # rows 0..k-1, which only coincides with this when P == 1; for any
        # longer prompt it verified against the wrong distributions.
        target_logits = self.target.forward(verify_ids, None)    # [1, P+k, vocab]
        L_total = target_logits.shape[1]
        p_all = F.softmax(target_logits, dim=-1)[0][L_total - (k + 1):]  # [k+1, vocab]

        # 4. Rejection sampling loop
        accepted = []
        final_token = None
        n_accepted = 0

        for i in range(k):
            p = p_all[i]  # target probs for draft token i (d_{i+1})
            q = draft_probs[i]  # draft probs for token i
            d_token = draft_tokens[i]

            p_prob = float(p[d_token].item())
            q_prob = float(q[d_token].item())

            # Accept with probability min(1, p/q)
            accept_prob = min(1.0, p_prob / q_prob)
            if torch.rand(1, generator=rng).item() < accept_prob:
                accepted.append(d_token)
                n_accepted += 1
            else:
                # Reject: resample from residual
                residual = torch.clamp(p - q, min=0.0)
                residual_sum = residual.sum()
                if residual_sum > 0:
                    final_token = int(torch.multinomial(residual / residual_sum, 1, generator=rng).item())
                else:
                    # Numerical fallback: sample from p
                    final_token = int(torch.multinomial(p, 1, generator=rng).item())
                break

        # If all K accepted, sample bonus token from p_all[K]
        if n_accepted == k:
            bonus_p = p_all[k]
            final_token = int(torch.multinomial(bonus_p, 1, generator=rng).item())

        # Update survival estimate EMA (simplified)
        # In a real engine, we'd track the exact position of rejection.
        actual_survival = (n_accepted + 1) / (k + 1)
        for i in range(k + 1):
            self.survival_ema[i] = 0.9 * self.survival_ema[i] + 0.1 * actual_survival

        output_tokens = accepted + [final_token]
        return output_tokens, target_kv_cache, draft_kv_cache