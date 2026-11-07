"""Torch.profiler Trace Analyzer.

Extracts the top bottleneck operations from a profiling trace, computing
the percentage of total time spent in each operator (e.g., aten::mm,
aten::add, custom paged_attention).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple

import torch


def profile_function(
    fn: callable,
    *args,
    warmup: int = 3,
    active: int = 5,
    **kwargs,
) -> Dict[str, float]:
    """Profile a function and return the top time-consuming operations.
    
    Returns a dictionary mapping operator name to percentage of total time.
    """
    # 1. Warmup (compilation, memory alloc, etc.)
    for _ in range(warmup):
        fn(*args, **kwargs)
        
    # 2. Profile
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        record_shapes=False,
    ) as prof:
        for _ in range(active):
            fn(*args, **kwargs)
            
    # 3. Analyze
    # We use the key_averages() to aggregate across the active steps.
    key_averages = prof.key_averages()
    total_time_us = sum(evt.self_cpu_time_total for evt in key_averages)
    
    if total_time_us == 0:
        return {}
        
    breakdown = {}
    for evt in key_averages:
        name = evt.key
        # Filter out trivial overhead ops for readability
        if name in ("aten::empty", "aten::empty_like", "cudaDeviceSynchronize"):
            continue
        time_pct = (evt.self_cpu_time_total / total_time_us) * 100.0
        if time_pct > 0.1:  # Only report ops taking >0.1% of time
            breakdown[name] = time_pct
            
    # Sort by percentage descending
    sorted_breakdown = dict(
        sorted(breakdown.items(), key=lambda item: item[1], reverse=True)
    )
    
    return sorted_breakdown