"""Milestone 5 — Continuous Batching (Iteration Scheduler).

State machine
-------------
Each request transitions through::

    WAITING --admit--> PREFILLING --done--> RUNNING --done--> FINISHED
                                    \\-preempt-/
                                     \\
                                      WAITING (swap_restore path)

Only ``RUNNING`` requests are scheduled as decoders. Only ``PREFILLING``
requests (with ``prefill_pos < len(prompt)``) are scheduled as prefills.
This separation is critical: a freshly admitted request has blocks
allocated but has not yet executed a single forward pass, so treating it
as a decoder would feed garbage to the KV cache.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch

from engine.prefix_cache import OutOfBlocksError, PrefixCache


class SeqState(Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class Request:
    req_id: int
    prompt_tokens: List[int]
    max_new_tokens: int
    arrival_time: float = field(default_factory=time.monotonic)

    state: SeqState = SeqState.WAITING
    # How many prompt tokens have been prefilled so far. Reaches
    # ``len(prompt_tokens)`` exactly when prefill completes; at that moment
    # state transitions to RUNNING.
    prefill_pos: int = 0
    generated_tokens: List[int] = field(default_factory=list)
    block_table: List[int] = field(default_factory=list)
    swapped_kv: Optional[torch.Tensor] = None

    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    itl_times: List[float] = field(default_factory=list)

    @property
    def seq_len(self) -> int:
        # Prompt + generated. Prefill progress does NOT enter this sum:
        # block allocation is prompt+max_new regardless of how much of
        # the prompt has been processed.
        return len(self.prompt_tokens) + len(self.generated_tokens)


@dataclass
class SchedulerOutput:
    decodes: List[Request]
    prefills: List[Tuple[Request, int]]
    total_tokens: int


class ContinuousEngine:
    def __init__(
        self,
        prefix_cache: PrefixCache,
        block_size: int,
        max_batch_tokens: int,
        itl_budget_s: float,
        t_tok_s: float,
    ) -> None:
        self.cache = prefix_cache
        self.block_size = block_size
        self.max_batch_tokens = max_batch_tokens
        self.itl_budget_s = itl_budget_s
        self.t_tok_s = t_tok_s

        self.waiting_queue: List[Request] = []
        self.active_sequences: Dict[int, Request] = {}
        self._next_req_id = 0
        # Total sequences finished (not iterations). Read by benchmarks
        # and the memory test.
        self._total_finished = 0

    # ------------------------------------------------------------------ #
    # Admission
    # ------------------------------------------------------------------ #

    def add_request(self, prompt_tokens: List[int], max_new_tokens: int) -> Request:
        req = Request(
            req_id=self._next_req_id,
            prompt_tokens=list(prompt_tokens),
            max_new_tokens=max_new_tokens,
        )
        self._next_req_id += 1
        self.waiting_queue.append(req)
        return req

    def _required_blocks(self, req: Request) -> int:
        return math.ceil((len(req.prompt_tokens) + req.max_new_tokens) / self.block_size)

    def _active_reserved(self) -> int:
        """Sum over active sequences of blocks promised but not yet allocated."""
        total = 0
        for active in self.active_sequences.values():
            max_blocks = self._required_blocks(active)
            total += max(0, max_blocks - len(active.block_table))
        return total

    def _admission_check(self, req: Request) -> bool:
        blocks_needed = self._required_blocks(req)
        matched_blocks, matched_len, node = self.cache.tree.match(req.prompt_tokens)
        matched_full = len(matched_blocks)

        # h_evict: matched blocks that are currently evictable (ref=0)
        # and will be pinned by this admission.
        h_evict = 0
        curr = node
        while curr and curr.parent:
            if curr.ref_count == 0:
                h_evict += len(curr.block_ids)
            curr = curr.parent

        new_need = blocks_needed - matched_full
        active_reserved = self._active_reserved()
        free = len(self.cache.allocator.free_blocks)
        # Reclaimable estimate. In a production engine this is tracked
        # incrementally in the radix tree; here we over-estimate 0.
        evictable = 0

        return new_need + active_reserved <= free + evictable - h_evict

    # ------------------------------------------------------------------ #
    # Scheduling
    # ------------------------------------------------------------------ #

    def schedule(self) -> SchedulerOutput:
        # 1. Admit waiting requests.
        for req in list(self.waiting_queue):
            if self._admission_check(req):
                self.waiting_queue.remove(req)
                self._init_sequence(req)
                self.active_sequences[req.req_id] = req

        # 2. Decoders: state == RUNNING.
        decodes = [r for r in self.active_sequences.values() if r.state == SeqState.RUNNING]
        n_d = len(decodes)

        # 3. Compute budget for prefill.
        max_tokens_budget = int(self.itl_budget_s / self.t_tok_s + 1e-9)
        c_max = max(0, max_tokens_budget - n_d)

        # 4. Prefills: state == PREFILLING with prefill in progress.
        prefills: List[Tuple[Request, int]] = []
        budget_remaining = c_max
        for req in self.active_sequences.values():
            if budget_remaining <= 0:
                break
            if req.state != SeqState.PREFILLING:
                continue
            if req.prefill_pos >= len(req.prompt_tokens):
                # Should not happen: state machine transitions to RUNNING
                # in step() when prefill completes. Defensive skip.
                continue
            remaining = len(req.prompt_tokens) - req.prefill_pos
            c = min(remaining, budget_remaining, self.max_batch_tokens)
            if c > 0:
                prefills.append((req, c))
                budget_remaining -= c

        prefill_tokens = c_max - budget_remaining
        return SchedulerOutput(decodes, prefills, n_d + prefill_tokens)

    def _init_sequence(self, req: Request) -> None:
        matched_blocks, matched_len, node = self.cache.tree.match(req.prompt_tokens)
        self.cache.tree.pin(node)

        total_blocks = self._required_blocks(req)
        new_blocks_needed = max(0, total_blocks - len(matched_blocks))
        new_blocks = self.cache.allocate_blocks(new_blocks_needed)
        req.block_table = matched_blocks + new_blocks

        # State, not generated_tokens, tracks prefill progress.
        req.state = SeqState.PREFILLING
        # If we resume from swap: keep prefill_pos at len(prompt) so the
        # first scheduler step transitions us immediately to RUNNING.
        if req.swapped_kv is not None:
            req.prefill_pos = len(req.prompt_tokens)
        # else: prefill_pos stays 0 (its dataclass default)

    # ------------------------------------------------------------------ #
    # Preemption
    # ------------------------------------------------------------------ #

    def _preempt(self, victim: Request) -> None:
        victim.state = SeqState.WAITING
        if victim.block_table:
            num_blocks = len(victim.block_table)
            block_shape = self.cache.allocator.pool.shape[1:]
            swapped = torch.empty(
                (num_blocks, *block_shape),
                dtype=self.cache.allocator.pool.dtype,
            )
            for i, blk_id in enumerate(victim.block_table):
                swapped[i] = self.cache.allocator.pool[blk_id]
                self.cache.allocator.free(blk_id)
            victim.swapped_kv = swapped
            victim.block_table = []
        # Re-queue at head; retained prefill_pos lets us skip recompute
        # once blocks are restored from the swap.
        self.waiting_queue.insert(0, victim)
        self.active_sequences.pop(victim.req_id, None)

    # ------------------------------------------------------------------ #
    # Step
    # ------------------------------------------------------------------ #

    def step(self) -> bool:
        sched = self.schedule()
        if not sched.decodes and not sched.prefills:
            return False

        # Simulated forward pass.
        time.sleep(sched.total_tokens * self.t_tok_s)

        finished_any = False

        # --- Prefill ---
        for req, chunk_size in sched.prefills:
            req.prefill_pos += chunk_size
            if req.prefill_pos >= len(req.prompt_tokens):
                req.state = SeqState.RUNNING

        # --- Decode ---
        now = time.monotonic()
        for req in sched.decodes:
            req.generated_tokens.append(0)
            if req.first_token_time is None:
                req.first_token_time = now
            # ITL bookkeeping
            if req.itl_times:
                req.itl_times.append(now - req.itl_times[-1])
            else:
                req.itl_times.append(now - req.arrival_time)

            if len(req.generated_tokens) >= req.max_new_tokens:
                self._finish(req)
                finished_any = True

        return finished_any

    def _finish(self, req: Request) -> None:
        req.state = SeqState.FINISHED
        req.finish_time = time.monotonic()
        for blk in req.block_table:
            self.cache.allocator.free(blk)
        req.block_table = []
        self.active_sequences.pop(req.req_id, None)
        self._total_finished += 1