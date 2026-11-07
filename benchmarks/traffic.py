"""Async Poisson Workload Generator.

Simulates realistic API traffic using a Poisson arrival process.
A Poisson process models independent, memoryless arrivals, which is the
standard assumption for web request traffic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List


@dataclass
class Request:
    req_id: int
    prompt_len: int
    max_new_tokens: int
    arrival_time_s: float


def generate_poisson_traffic(
    num_requests: int,
    arrival_rate_rps: float,    # Requests per second (lambda)
    prompt_len_range: tuple[int, int],
    new_tokens_range: tuple[int, int],
    seed: int = 42,
) -> List[Request]:
    """Generate a sequence of requests with Poisson inter-arrival times.
    
    The inter-arrival time between requests in a Poisson process follows
    an exponential distribution with mean 1/lambda.
    """
    rng = random.Random(seed)
    requests = []
    
    current_time = 0.0
    for i in range(num_requests):
        # Exponential distribution: -ln(U) / lambda
        inter_arrival = rng.expovariate(arrival_rate_rps)
        current_time += inter_arrival
        
        prompt_len = rng.randint(*prompt_len_range)
        max_new_tokens = rng.randint(*new_tokens_range)
        
        requests.append(
            Request(
                req_id=i,
                prompt_len=prompt_len,
                max_new_tokens=max_new_tokens,
                arrival_time_s=current_time,
            )
        )
        
    return requests