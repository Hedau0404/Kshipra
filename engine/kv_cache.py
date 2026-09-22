import time

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class KVCacheGenerator:
    """
    Autoregressive generator using Key-Value (KV) caching.

    The prompt is processed once during the prefill phase.
    During decoding, only the newest token is passed to the model.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        """
        Loads model and tokenizer, moves model to CPU,
        and sets model to evaluation mode.
        """

        # Load tokenizer.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # GPT-2 does not have a dedicated padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load GPT-2.
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Run on CPU for our benchmark.
        self.model = self.model.to("cpu")

        # Inference mode.
        self.model.eval()

        # Remember the device.
        self.device = torch.device("cpu")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 20
    ) -> dict:
        """
        Generates text autoregressively using KV caching
        and greedy decoding.
        """

        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

        # ---------------------------------------------------------
        # 1. Tokenize the prompt
        # ---------------------------------------------------------

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        )

        # The complete prompt is used during prefill.
        #
        # Shape:
        # (1, T)
        input_ids = inputs["input_ids"].to(self.device)

        # Keep all token IDs so we can decode the final result.
        all_token_ids = input_ids[0].tolist()

        # ---------------------------------------------------------
        # 2. Set up memory monitoring
        # ---------------------------------------------------------

        process = psutil.Process()

        peak_ram_bytes = process.memory_info().rss

        # ---------------------------------------------------------
        # 3. Set up timing
        # ---------------------------------------------------------

        time_to_first_token_s = None

        inter_token_latencies_s = []

        generation_start = time.perf_counter()

        # ---------------------------------------------------------
        # 4. KV cache starts empty
        # ---------------------------------------------------------

        past_key_values = None

        # ---------------------------------------------------------
        # 5. Autoregressive generation loop
        # ---------------------------------------------------------

        for step in range(max_new_tokens):

            step_start = time.perf_counter()

            # -----------------------------------------------------
            # PREFILL
            #
            # First iteration:
            #
            # input_ids shape = (1, T)
            #
            # We process the entire prompt.
            # -----------------------------------------------------

            if past_key_values is None:

                outputs = self.model(
                    input_ids=input_ids,
                    use_cache=True
                )

            # -----------------------------------------------------
            # DECODE
            #
            # Later iterations:
            #
            # input_ids shape = (1, 1)
            #
            # Only the newest token is processed.
            #
            # The previous K/V information comes from the cache.
            # -----------------------------------------------------

            else:

                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            # -----------------------------------------------------
            # Get prediction scores
            # -----------------------------------------------------

            logits = outputs.logits

            # Save the updated K/V cache.
            past_key_values = outputs.past_key_values

            # -----------------------------------------------------
            # Get the prediction for the newest position
            # -----------------------------------------------------

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            # -----------------------------------------------------
            # Add token to our complete token list
            # -----------------------------------------------------

            all_token_ids.append(
                next_token.item()
            )

            # -----------------------------------------------------
            # The next iteration only needs this new token.
            #
            # Shape:
            # (1, 1)
            # -----------------------------------------------------

            input_ids = next_token

            # -----------------------------------------------------
            # Measure latency
            # -----------------------------------------------------

            step_latency = (
                time.perf_counter() - step_start
            )

            if step == 0:

                # Prefill + first token.
                time_to_first_token_s = step_latency

            else:

                # Decode token latency.
                inter_token_latencies_s.append(
                    step_latency
                )

            # -----------------------------------------------------
            # Measure memory
            # -----------------------------------------------------

            current_ram_bytes = process.memory_info().rss

            peak_ram_bytes = max(
                peak_ram_bytes,
                current_ram_bytes
            )

        # ---------------------------------------------------------
        # 6. Total generation time
        # ---------------------------------------------------------

        total_generation_time_s = (
            time.perf_counter() - generation_start
        )

        # ---------------------------------------------------------
        # 7. Throughput
        # ---------------------------------------------------------

        tokens_per_sec = (
            max_new_tokens /
            total_generation_time_s
        )

        # ---------------------------------------------------------
        # 8. Convert token IDs back to text
        # ---------------------------------------------------------

        generated_text = self.tokenizer.decode(
            all_token_ids,
            skip_special_tokens=True
        )

        # ---------------------------------------------------------
        # 9. Convert RAM bytes → MB
        # ---------------------------------------------------------

        peak_ram_mb = (
            peak_ram_bytes /
            (1024 * 1024)
        )

        # ---------------------------------------------------------
        # 10. Return benchmark results
        # ---------------------------------------------------------
        prompt_length = inputs["input_ids"].shape[1]
        final_sequence_length = len(all_token_ids)
        return {
            "generated_text": generated_text,
            "time_to_first_token_s": time_to_first_token_s,
            "inter_token_latencies_s": inter_token_latencies_s,
            "tokens_per_sec": tokens_per_sec,
            "peak_ram_mb": peak_ram_mb,
            "prompt_length": prompt_length,
            "final_sequence_length": final_sequence_length,
        }