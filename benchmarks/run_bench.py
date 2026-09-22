import csv
import gc
import statistics
import time
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List

from engine.baseline import BaselineGenerator
from engine.kv_cache import KVCacheGenerator
from engine.batching import BatchGenerator

class BenchmarkRunner:
    """
    Unified benchmark harness for evaluating LLM inference techniques.
    Supports single-prompt generators and batched generators.
    """
    def __init__(self, max_new_tokens: int = 20, num_runs: int = 5):
        self.max_new_tokens = max_new_tokens
        self.num_runs = num_runs

    def run_single_engine(self, engine_name: str, engine: Any, prompt: str) -> Dict[str, Any]:
        """Executes warm-up and runs single-prompt engines (Baseline, KV-Cache)."""
        print("=" * 65)
        print(f"RUNNING BENCHMARK: {engine_name.upper()}")
        print("=" * 65)

        # Warm-up pass
        _ = engine.generate(prompt, max_new_tokens=5)
        gc.collect()

        all_results = []
        for run in range(self.num_runs):
            gc.collect()
            metrics = engine.generate(prompt, max_new_tokens=self.max_new_tokens)
            all_results.append(metrics)
            print(
                f"Run {run + 1}/{self.num_runs} | "
                f"TTFT: {metrics['time_to_first_token_s']:.4f}s | "
                f"Throughput: {metrics['tokens_per_sec']:.2f} tok/s | "
                f"Peak RAM: {metrics['peak_ram_mb']:.2f} MB"
            )

        aggregated = self._aggregate_single(engine_name, all_results)
        self._print_summary(aggregated)
        return aggregated

    def run_sequential_vs_batched(
        self, 
        kv_engine: KVCacheGenerator, 
        batch_engine: BatchGenerator, 
        prompts: List[str]
    ) -> Dict[str, Any]:
        """
        Runs N prompts sequentially through KV-Cache, then runs them as a batch.
        Compares total latency and aggregate system throughput.
        """
        batch_size = len(prompts)
        print("=" * 65)
        print(f"RUNNING BATCHING BENCHMARK (Batch Size = {batch_size})")
        print("=" * 65)

        # -------------------------------------------------------------
        # 1. Sequential Execution (Processing prompts one after another)
        # -------------------------------------------------------------
        print("\n--- 1. Sequential KV-Cache Execution ---")
        seq_start = time.perf_counter()
        seq_texts = []
        for p in prompts:
            res = kv_engine.generate(p, max_new_tokens=self.max_new_tokens)
            seq_texts.append(res["generated_text"])
        seq_total_time = time.perf_counter() - seq_start
        total_tokens = batch_size * self.max_new_tokens
        seq_throughput = total_tokens / seq_total_time

        print(f"Sequential Total Time: {seq_total_time:.4f} s")
        print(f"Sequential System Throughput: {seq_throughput:.2f} tok/s")

        # -------------------------------------------------------------
        # 2. Batched Execution (Processing prompts in parallel)
        # -------------------------------------------------------------
        print("\n--- 2. Batched KV-Cache Execution ---")
        # Warm-up pass
        _ = batch_engine.generate_batch(prompts, max_new_tokens=5)
        gc.collect()

        batch_results = []
        for run in range(self.num_runs):
            gc.collect()
            metrics = batch_engine.generate_batch(prompts, max_new_tokens=self.max_new_tokens)
            batch_results.append(metrics)
            print(
                f"Run {run + 1}/{self.num_runs} | "
                f"TTFT: {metrics['time_to_first_token_s']:.4f}s | "
                f"System Throughput: {metrics['tokens_per_sec']:.2f} tok/s | "
                f"Peak RAM: {metrics['peak_ram_mb']:.2f} MB"
            )

        batch_throughput_mean = statistics.mean([r["tokens_per_sec"] for r in batch_results])
        batch_ram_mean = statistics.mean([r["peak_ram_mb"] for r in batch_results])
        batch_ttft_mean = statistics.mean([r["time_to_first_token_s"] for r in batch_results])

        # -------------------------------------------------------------
        # 3. Correctness / Equivalence Test
        # -------------------------------------------------------------
        batched_texts = batch_results[0]["generated_texts"]
        print("\n--- 3. Batching Equivalence Verification ---")
        all_matched = True
        for i, (seq_t, batch_t) in enumerate(zip(seq_texts, batched_texts)):
            match = (seq_t == batch_t)
            all_matched = all_matched and match
            status = "MATCH" if match else "MISMATCH"
            print(f"Prompt {i+1} [{status}]:\n  Seq  : {seq_t}\n  Batch: {batch_t}\n")

        speedup = batch_throughput_mean / seq_throughput
        print("-" * 65)
        print(f"BATCHING SPEEDUP SUMMARY:")
        print(f"Sequential Throughput : {seq_throughput:.2f} tok/s")
        print(f"Batched Throughput    : {batch_throughput_mean:.2f} tok/s")
        print(f"Net System Speedup    : {speedup:.2f}x")
        print(f"Outputs Identical     : {all_matched}")
        print("-" * 65)

        return {
            "engine_name": f"Batched (B={batch_size})",
            "generated_text": batched_texts[0],
            "ttft_mean": batch_ttft_mean,
            "throughput_mean": batch_throughput_mean,
            "ram_mean": batch_ram_mean,
            "avg_inter_token_latencies": [
                statistics.mean([r["inter_token_latencies_s"][i] for r in batch_results])
                for i in range(len(batch_results[0]["inter_token_latencies_s"]))
            ]
        }

    def _aggregate_single(self, engine_name: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "engine_name": engine_name,
            "generated_text": results[0]["generated_text"],
            "prompt_length": results[0]["prompt_length"],
            "final_sequence_length": results[0]["final_sequence_length"],
            "ttft_mean": statistics.mean([r["time_to_first_token_s"] for r in results]),
            "throughput_mean": statistics.mean([r["tokens_per_sec"] for r in results]),
            "ram_mean": statistics.mean([r["peak_ram_mb"] for r in results]),
            "avg_inter_token_latencies": [
                statistics.mean([r["inter_token_latencies_s"][i] for r in results])
                for i in range(len(results[0]["inter_token_latencies_s"]))
            ],
        }

    def _print_summary(self, agg: Dict[str, Any]) -> None:
        print("\n" + "-" * 65)
        print(f"FINAL SUMMARY: {agg['engine_name']}")
        print("-" * 65)
        print(f"Generated Text:\n{agg['generated_text']}\n")
        print(f"TTFT (Mean):       {agg['ttft_mean']:.4f} s")
        print(f"Throughput (Mean): {agg['throughput_mean']:.2f} tokens/s")
        print(f"Peak RAM (Mean):   {agg['ram_mean']:.2f} MB\n")

    def plot_comparison(self, aggregated_results: List[Dict[str, Any]], output_path: str = "latency_comparison.png") -> None:
        plt.figure(figsize=(10, 6))
        for agg in aggregated_results:
            steps = list(range(1, self.max_new_tokens + 1))
            latencies = [agg["ttft_mean"]] + agg["avg_inter_token_latencies"]
            plt.plot(steps, latencies, marker="o", label=agg["engine_name"])

        plt.xlabel("Generation Step")
        plt.ylabel("Latency (seconds)")
        plt.title("Per-Token Latency Comparison Across Inference Techniques")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        print(f"Comparison plot saved to: {output_path}")


if __name__ == "__main__":
    MAX_NEW_TOKENS = 20
    
    TEST_PROMPTS = [
        "In a warm summer evening, the ancient library",
        "The future of artificial intelligence is",
        "Deep learning models require significant",
        "Python is a popular programming language because"
    ]

    runner = BenchmarkRunner(max_new_tokens=MAX_NEW_TOKENS, num_runs=5)

    # Single-prompt benchmark (using Prompt 1)
    baseline_engine = BaselineGenerator(model_name="gpt2")
    baseline_stats = runner.run_single_engine("Baseline (No Cache)", baseline_engine, TEST_PROMPTS[0])

    kv_engine = KVCacheGenerator(model_name="gpt2")
    kv_stats = runner.run_single_engine("KV-Cache", kv_engine, TEST_PROMPTS[0])

    # Batched benchmark (Processing all 4 prompts concurrently vs sequentially)
    batch_engine = BatchGenerator(model_name="gpt2")
    batch_stats = runner.run_sequential_vs_batched(kv_engine, batch_engine, TEST_PROMPTS)

    # Plot baseline vs KV-cache vs Batched per-step decode latency
    runner.plot_comparison([baseline_stats, kv_stats, batch_stats])