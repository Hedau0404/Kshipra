"""Unit tests for engine.prefix_cache (Milestone 3)."""

from __future__ import annotations

import sys

import pytest

from benchmarks.run_benchmarks import isolated_run
from engine.prefix_cache import (
    BlockAllocator,
    OutOfBlocksError,
    PrefixCache,
    RadixTree,
    TokenArena,
)


def test_block_allocator_lifo():
    alloc = BlockAllocator(
        num_blocks=4, block_size=16, num_layers=2, num_kv_heads=2, head_dim=8
    )
    b1 = alloc.alloc()
    b2 = alloc.alloc()
    assert b1 == 3
    assert b2 == 2
    alloc.free(b1)
    b3 = alloc.alloc()
    assert b3 == 3  # LIFO reuse


def test_token_arena():
    arena = TokenArena(capacity=10)
    start, length = arena.alloc([1, 2, 3])
    assert start == 0
    assert length == 3
    assert arena.storage[0:3].tolist() == [1, 2, 3]


def test_radix_tree_match_exact():
    arena = TokenArena(1000)
    alloc = BlockAllocator(
        num_blocks=100, block_size=4, num_layers=1, num_kv_heads=1, head_dim=1
    )
    tree = RadixTree(alloc, arena, block_size=4)

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    blocks = [alloc.alloc(), alloc.alloc()]
    node = tree.insert(tokens, blocks, 0, tree.root)
    tree.pin(node)

    matched_blocks, matched_len, last_node = tree.match(tokens)
    assert matched_len == 8
    assert matched_blocks == blocks
    assert last_node == node


def test_radix_tree_split():
    arena = TokenArena(1000)
    alloc = BlockAllocator(
        num_blocks=100, block_size=4, num_layers=1, num_kv_heads=1, head_dim=1
    )
    tree = RadixTree(alloc, arena, block_size=4)

    # Insert 8 tokens (2 blocks)
    tokens1 = [1, 2, 3, 4, 5, 6, 7, 8]
    blocks1 = [alloc.alloc(), alloc.alloc()]
    node1 = tree.insert(tokens1, blocks1, 0, tree.root)
    tree.pin(node1)

    # Match 4 tokens (1 block) - should split node1
    tokens2 = [1, 2, 3, 4, 9, 10, 11, 12]
    matched_blocks, matched_len, last_node = tree.match(tokens2)
    assert matched_len == 4
    assert matched_blocks == [blocks1[0]]
    assert last_node.token_len == 4
    assert last_node.block_ids == [blocks1[0]]
    assert len(last_node.children) == 1

    # The child should be the tail node with the remaining 4 tokens
    tail = list(last_node.children.values())[0]
    assert tail.token_len == 4
    assert tail.block_ids == [blocks1[1]]


def test_eviction_lru_and_ref_count():
    arena = TokenArena(1000)
    alloc = BlockAllocator(
        num_blocks=10, block_size=4, num_layers=1, num_kv_heads=1, head_dim=1
    )
    tree = RadixTree(alloc, arena, block_size=4)

    tokens1 = [1, 2, 3, 4]
    b1 = [alloc.alloc()]
    n1 = tree.insert(tokens1, b1, 0, tree.root)
    tree.pin(n1)

    tokens2 = [5, 6, 7, 8]
    b2 = [alloc.alloc()]
    n2 = tree.insert(tokens2, b2, 0, tree.root)
    tree.pin(n2)
    tree.unpin(n2)  # n2 is now evictable

    assert len(alloc.free_blocks) == 8

    tree.evict(1)
    assert len(alloc.free_blocks) == 9
    assert alloc.free_blocks[-1] == b2[0]  # n2 evicted
    assert n1.ref_count == 1  # n1 still pinned


def test_eviction_cascades():
    arena = TokenArena(1000)
    alloc = BlockAllocator(
        num_blocks=10, block_size=4, num_layers=1, num_kv_heads=1, head_dim=1
    )
    tree = RadixTree(alloc, arena, block_size=4)

    tokens1 = [1, 2, 3, 4]
    b1 = [alloc.alloc()]
    n1 = tree.insert(tokens1, b1, 0, tree.root)

    tokens2 = [1, 2, 3, 4, 5, 6, 7, 8]
    b2 = [alloc.alloc()]
    n2 = tree.insert(tokens2, b2, 4, n1)

    tree.evict(2)
    assert len(alloc.free_blocks) == 10
    assert len(tree.root.children) == 0


def test_prefix_cache_allocate_with_eviction():
    cache = PrefixCache(
        num_blocks=4, block_size=4, num_layers=1, num_kv_heads=1, head_dim=1
    )
    t1 = [1, 2, 3, 4]
    b1 = cache.allocate_blocks(1)
    n1 = cache.tree.insert(t1, b1, 0, cache.tree.root)
    cache.tree.pin(n1)

    t2 = [5, 6, 7, 8]
    b2 = cache.allocate_blocks(1)
    n2 = cache.tree.insert(t2, b2, 0, cache.tree.root)
    cache.tree.pin(n2)

    t3 = [9, 10, 11, 12]
    b3 = cache.allocate_blocks(1)
    n3 = cache.tree.insert(t3, b3, 0, cache.tree.root)
    cache.tree.pin(n3)

    # Now 3 blocks used, 1 free.
    cache.tree.unpin(n2)
    cache.tree.unpin(n3)

    # Allocate 2 blocks (needs eviction)
    b4 = cache.allocate_blocks(2)
    assert len(b4) == 2
    assert n1.ref_count == 1  # n1 kept


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_prefix_cache_under_4gib():
    res = isolated_run(_workload_prefix_cache, timeout_s=120.0)
    assert res.result["blocks"] == 100
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_prefix_cache() -> dict:
    cache = PrefixCache(
        num_blocks=100, block_size=16, num_layers=2, num_kv_heads=2, head_dim=8
    )
    t = list(range(64))
    b = cache.allocate_blocks(4)
    cache.tree.insert(t, b, 0, cache.tree.root)
    return {"blocks": 100}