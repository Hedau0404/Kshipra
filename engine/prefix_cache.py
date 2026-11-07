"""Milestone 3 — Radix Tree Prefix Cache.

This module implements a block-based prefix cache to share KV memory across
requests with common prefixes (e.g., a system prompt).

Architecture
------------
1. **BlockAllocator**: A single preallocated tensor pool `[num_blocks, L, 2, H, B, D]`.
   Allocations return int32 block IDs from a LIFO free list. This avoids glibc
   heap fragmentation and guarantees O(1) allocation.
2. **TokenArena**: A flat int32 tensor that stores the actual token IDs for
   radix nodes. This prevents Python object overhead (boxed ints) from bloating
   tree metadata.
3. **RadixTree**: A tree of `RadixNode` objects. Because we use `__slots__`,
   the tree is memory-efficient. Each node represents a block-aligned chunk of
   tokens and holds a list of physical block IDs.
4. **PrefixCache**: The public API combining the allocator and tree.

Memory and Cache Behavior
-------------------------
When a request arrives, we traverse the tree once at admission (control plane).
We materialize a flat `block_table` (a list of physical block IDs) for the
sequence. The decode kernel only reads this block table, never the tree. This
avoids pointer-chasing and cache misses during the memory-bound decode step.

Eviction
--------
Nodes are reference-counted (`ref_count`). A sequence pins the entire path
from the matched leaf to the root at admission. A node is evictable if and only
if `ref_count == 0` AND it has no children. Evicting a leaf cascades upward,
allowing interior nodes to become evictable once their subtrees are removed.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import torch


class OutOfBlocksError(RuntimeError):
    """Raised when the allocator cannot satisfy a request without eviction."""


# ---------------------------------------------------------------------- #
# Allocator
# ---------------------------------------------------------------------- #


class BlockAllocator:
    """Preallocated pool of KV blocks.

    Layout: `[num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim]`.
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.pool = torch.zeros(
            (num_blocks, num_layers, 2, num_kv_heads, block_size, head_dim),
            dtype=dtype,
            device=device,
        )
        # LIFO stack of free block IDs. Reusing recently freed blocks
        # improves cache/TLB hit rates.
        self.free_blocks = list(range(num_blocks))

    def alloc(self) -> int:
        if not self.free_blocks:
            raise OutOfBlocksError("No free blocks available")
        return self.free_blocks.pop()

    def free(self, block_id: int) -> None:
        self.free_blocks.append(block_id)


# ---------------------------------------------------------------------- #
# Token Arena
# ---------------------------------------------------------------------- #


class TokenArena:
    """Flat storage for radix node tokens to avoid boxed Python ints."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.storage = torch.zeros(capacity, dtype=torch.int32)
        self.ptr = 0

    def alloc(self, tokens: List[int]) -> Tuple[int, int]:
        if self.ptr + len(tokens) > self.capacity:
            raise RuntimeError("TokenArena out of memory")
        start = self.ptr
        self.storage[start : start + len(tokens)] = torch.tensor(
            tokens, dtype=torch.int32
        )
        self.ptr += len(tokens)
        return start, len(tokens)


# ---------------------------------------------------------------------- #
# Radix Tree
# ---------------------------------------------------------------------- #


class RadixNode:
    __slots__ = (
        "parent",
        "children",
        "ref_count",
        "last_access",
        "block_ids",
        "token_start",
        "token_len",
    )

    def __init__(
        self,
        parent: Optional["RadixNode"],
        token_start: int,
        token_len: int,
        block_ids: List[int],
    ) -> None:
        self.parent = parent
        self.children: Dict[int, "RadixNode"] = {}
        self.ref_count = 0
        self.last_access = time.monotonic()
        self.block_ids = block_ids
        self.token_start = token_start
        self.token_len = token_len


class RadixTree:
    def __init__(
        self, allocator: BlockAllocator, token_arena: TokenArena, block_size: int
    ) -> None:
        self.allocator = allocator
        self.token_arena = token_arena
        self.block_size = block_size
        self.root = RadixNode(None, 0, 0, [])

    def match(
        self, token_ids: List[int]
    ) -> Tuple[List[int], int, Optional[RadixNode]]:
        """Match a prompt against the tree.

        Returns (matched_block_ids, matched_len, last_matched_node).
        If a partial block match occurs, we only take the full blocks and
        recompute the remainder (vLLM style).
        """
        curr = self.root
        matched_blocks = []
        matched_len = 0

        while matched_len < len(token_ids):
            next_tok = token_ids[matched_len]
            if next_tok not in curr.children:
                break
            child = curr.children[next_tok]

            child_tokens = self.token_arena.storage[
                child.token_start : child.token_start + child.token_len
            ].tolist()

            match_len = 0
            max_check = min(len(child_tokens), len(token_ids) - matched_len)
            for i in range(max_check):
                if child_tokens[i] == token_ids[matched_len + i]:
                    match_len += 1
                else:
                    break

            if match_len == len(child_tokens):
                # Full node match
                matched_blocks.extend(child.block_ids)
                matched_len += match_len
                curr = child
            elif match_len > 0:
                # Partial match: split at the nearest block boundary
                aligned_len = (match_len // self.block_size) * self.block_size
                if aligned_len == 0:
                    break
                self._split_node(child, aligned_len)
                matched_blocks.extend(child.block_ids)
                matched_len += aligned_len
                curr = child
                break
            else:
                break

        return matched_blocks, matched_len, curr

    def _split_node(self, node: RadixNode, split_len: int) -> None:
        """Split a node at ``split_len`` (must be a multiple of block_size)."""
        n_blocks_head = split_len // self.block_size
        head_block_ids = node.block_ids[:n_blocks_head]
        tail_block_ids = node.block_ids[n_blocks_head:]

        tail_len = node.token_len - split_len
        tail_tokens = self.token_arena.storage[
            node.token_start + split_len : node.token_start + node.token_len
        ].tolist()
        tail_start, _ = self.token_arena.alloc(tail_tokens)

        # Truncate current node
        node.token_len = split_len
        node.block_ids = head_block_ids

        # Create tail node
        tail = RadixNode(node, tail_start, tail_len, tail_block_ids)
        tail.ref_count = node.ref_count
        tail.children = node.children
        for child in tail.children.values():
            child.parent = tail

        first_tail_tok = int(self.token_arena.storage[tail_start].item())
        node.children = {first_tail_tok: tail}

    def insert(
        self,
        token_ids: List[int],
        block_ids: List[int],
        matched_len: int,
        curr: RadixNode,
    ) -> RadixNode:
        """Insert the unmatched suffix of a prompt under ``curr``."""
        if len(token_ids) == matched_len:
            return curr

        new_tokens = token_ids[matched_len:]
        start, length = self.token_arena.alloc(new_tokens)
        new_node = RadixNode(curr, start, length, block_ids)

        first_tok = new_tokens[0]
        curr.children[first_tok] = new_node
        return new_node

    def pin(self, node: Optional[RadixNode]) -> None:
        """Increment ref counts along the path to the root."""
        curr = node
        while curr is not None:
            curr.ref_count += 1
            curr.last_access = time.monotonic()
            curr = curr.parent

    def unpin(self, node: Optional[RadixNode]) -> None:
        """Decrement ref counts along the path to the root."""
        curr = node
        while curr is not None:
            curr.ref_count -= 1
            curr = curr.parent

    def evict(self, num_blocks_needed: int) -> int:
        """Evict least-recently-used leaf nodes until enough blocks are freed.

        Re-scans the tree after each eviction because evicting a leaf
        promotes its parent to leaf status. A static snapshot of leaves
        (the naive implementation) would fail to cascade: if n2 is
        evicted and n1 becomes a leaf, n1 is invisible to the original
        list and survives the eviction pass. This violates the pool
        accounting: evict(N) must free exactly N blocks, or as many as
        the tree permits, whichever comes first.

        Returns the number of blocks actually freed. Returns fewer than
        num_blocks_needed only if the tree is fully exhausted of
        evictable leaves (all remaining nodes pinned or have children).
        """
        evicted = 0
        while evicted < num_blocks_needed:
            # Re-collect current leaves every iteration.
            leaves = []
            stack = [self.root]
            while stack:
                node = stack.pop()
                if (
                    node is not self.root
                    and not node.children
                    and node.ref_count == 0
                ):
                    leaves.append(node)
                stack.extend(node.children.values())

            if not leaves:
                break

            leaves.sort(key=lambda n: n.last_access)
            node = leaves[0]  # LRU leaf
            first_tok = int(self.token_arena.storage[node.token_start].item())
            del node.parent.children[first_tok]
            for blk in node.block_ids:
                self.allocator.free(blk)
            evicted += len(node.block_ids)

        return evicted


# ---------------------------------------------------------------------- #
# Public API
# ---------------------------------------------------------------------- #


class PrefixCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(
            num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype, device
        )
        self.token_arena = TokenArena(capacity=1024 * 1024)
        self.tree = RadixTree(self.allocator, self.token_arena, block_size)

    def allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate ``num_blocks``, evicting if necessary."""
        try:
            return [self.allocator.alloc() for _ in range(num_blocks)]
        except OutOfBlocksError:
            evicted = self.tree.evict(num_blocks)
            if evicted < num_blocks:
                raise OutOfBlocksError("Not enough evictable memory to satisfy request")
            return [self.allocator.alloc() for _ in range(num_blocks)]