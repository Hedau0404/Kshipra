import time

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaselineGenerator:
    """
    Naive autoregressive generator for GPT-2 (124M) without KV-caching.

    At every generation step, the entire accumulated sequence is passed
    through the model again.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        """
        Loads model and tokenizer, moves model to CPU, sets model to
        evaluation mode, and disables gradient calculations.
        """

        # Load the tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 does not have a dedicated padding token.
        # We aren't doing batching/padding here, but setting this makes
        # the tokenizer configuration complete.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the causal language model.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Explicitly use CPU for this baseline benchmark.
        self.model = self.model.to("cpu")

        # Evaluation mode disables training-specific behavior such as
        # dropout.
        self.model.eval()

        # Store the device so generate() can use it consistently.
        self.device = torch.device("cpu")

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 20) -> dict:
        """
        Generates text autoregressively using greedy decoding.

        Args:
            prompt: Input text string.
            max_new_tokens: Exact number of tokens to generate.

        Returns:
            Dict containing:
                - generated_text
                - time_to_first_token_s
                - inter_token_latencies_s
                - tokens_per_sec
                - peak_ram_mb
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        # ---------------------------------------------------------
        # Step 1: Tokenize the prompt
        # ---------------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        # input_ids has shape:
        #
        # (batch_size, sequence_length)
        #
        # For our single-prompt benchmark:
        #
        # (1, T)
        input_ids = inputs["input_ids"].to(self.device)

        # ---------------------------------------------------------
        # Step 2: Set up memory monitoring
        # ---------------------------------------------------------

        process = psutil.Process()

        # RSS = Resident Set Size.
        #
        # This is the amount of physical memory currently occupied
        # by this process.
        peak_ram_bytes = process.memory_info().rss

        # ---------------------------------------------------------
        # Step 3: Storage for timing measurements
        # ---------------------------------------------------------

        time_to_first_token_s = None

        # This will contain:
        #
        # [latency_token_2, latency_token_3, ...]
        #
        inter_token_latencies_s = []

        # Used to calculate total generation time.
        generation_start = time.perf_counter()

        # ---------------------------------------------------------
        # Step 4: Autoregressive generation loop
        # ---------------------------------------------------------

        for step in range(max_new_tokens):

            # Start timing this forward pass.
            step_start = time.perf_counter()

            # -----------------------------------------------------
            # IMPORTANT:
            #
            # We intentionally pass the ENTIRE input_ids tensor.
            #
            # This is the baseline without KV caching.
            # -----------------------------------------------------

            outputs = self.model(input_ids)

            # logits shape:
            #
            # (1, sequence_length, vocab_size)
            logits = outputs.logits

            # We only care about the logits at the final position.
            #
            # Shape before argmax:
            #
            # (1, vocab_size)
            #
            # argmax gives the vocabulary ID with the highest score.
            #
            # keepdim=True gives:
            #
            # (1, 1)
            #
            # This is important because input_ids must remain 2D.
            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            # -----------------------------------------------------
            # Append the newly generated token.
            # -----------------------------------------------------

            #
            # Before:
            #
            # input_ids = (1, seq_len)
            #
            # next_token = (1, 1)
            #
            # After:
            #
            # input_ids = (1, seq_len + 1)
            #
            input_ids = torch.cat(
                [input_ids, next_token],
                dim=1
            )

            # -----------------------------------------------------
            # Measure this step's latency.
            # -----------------------------------------------------

            step_latency = time.perf_counter() - step_start

            if step == 0:
                # First generated token.
                time_to_first_token_s = step_latency
            else:
                # Tokens 2..M.
                inter_token_latencies_s.append(step_latency)

            # -----------------------------------------------------
            # Update peak RSS measurement.
            # -----------------------------------------------------

            current_ram_bytes = process.memory_info().rss

            peak_ram_bytes = max(
                peak_ram_bytes,
                current_ram_bytes
            )

        # ---------------------------------------------------------
        # Step 5: Total generation time
        # ---------------------------------------------------------

        total_generation_time_s = (
            time.perf_counter() - generation_start
        )

        # ---------------------------------------------------------
        # Step 6: Calculate throughput
        # ---------------------------------------------------------

        tokens_per_sec = (
            max_new_tokens / total_generation_time_s
        )

        # ---------------------------------------------------------
        # Step 7: Decode the final sequence
        # ---------------------------------------------------------

        generated_text = self.tokenizer.decode(
            input_ids[0],
            skip_special_tokens=True
        )

        # Convert bytes -> MB.
        peak_ram_mb = peak_ram_bytes / (1024 * 1024)

        # ---------------------------------------------------------
        # Step 8: Return benchmark metrics
        # ---------------------------------------------------------
        prompt_length = inputs["input_ids"].shape[1]
        final_sequence_length = input_ids.shape[1]

        return {
            "generated_text": generated_text,
            "time_to_first_token_s": time_to_first_token_s,
            "inter_token_latencies_s": inter_token_latencies_s,
            "tokens_per_sec": tokens_per_sec,
            "peak_ram_mb": peak_ram_mb,
            "prompt_length": prompt_length,
            "final_sequence_length": final_sequence_length,
        }
    
#if __name__ == "__main__":
#    print("Loading model (this may take a minute on the first run)...")
#    generator = BaselineGenerator(model_name="gpt2")
#
#    prompt = "In a warm summer evening, the ancient library"
#    print(f"Running baseline generation for: '{prompt}'\n")
#
#    metrics = generator.generate(prompt, max_new_tokens=20)

#    print("--- Benchmark Results ---")
#    print(f"Generated Text:\n{metrics['generated_text']}\n")
#    print(f"Time to First Token (TTFT): {metrics['time_to_first_token_s']:.4f} s")
#    print(
#        f"Avg Inter-Token Latency:    "
#        f"{sum(metrics['inter_token_latencies_s'])/len(metrics['inter_token_latencies_s']):.4f} s"
#    )
#   print(f"Throughput:                 {metrics['tokens_per_sec']:.2f} tokens/s")
#    print(f"Peak RAM:                   {metrics['peak_ram_mb']:.2f} MB")