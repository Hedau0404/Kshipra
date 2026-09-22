import time

import psutil
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class BatchGenerator:
    """
    Static batch autoregressive generator using KV caching.

    Processes multiple input prompts concurrently in a single
    forward pass.
    """

    def __init__(self, model_name: str = "gpt2") -> None:
        """
        Loads model and tokenizer.

        GPT-2 is configured for left padding because this is
        required for correct batched autoregressive generation.
        """

        # ---------------------------------------------------------
        # 1. Load tokenizer
        # ---------------------------------------------------------

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        # ---------------------------------------------------------
        # 2. Configure padding
        # ---------------------------------------------------------

        # Decoder-only models should use left padding during
        # batched generation.
        self.tokenizer.padding_side = "left"

        # GPT-2 does not have a dedicated padding token.
        # Use EOS as the padding token.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ---------------------------------------------------------
        # 3. Load model
        # ---------------------------------------------------------

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name
        )

        # Run benchmark on CPU.
        self.model = self.model.to("cpu")

        # Inference mode.
        self.model.eval()

        # Store device.
        self.device = torch.device("cpu")

    @torch.no_grad()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 20
    ) -> dict:
        """
        Generates text for multiple prompts concurrently
        using greedy decoding and KV caching.
        """

        # ---------------------------------------------------------
        # 1. Validate inputs
        # ---------------------------------------------------------

        if not prompts:
            raise ValueError(
                "prompts must contain at least one prompt"
            )

        if max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be >= 1"
            )

        # ---------------------------------------------------------
        # 2. Tokenize and LEFT-PAD the prompts
        # ---------------------------------------------------------

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True
        )

        # input_ids:
        # (B, T_max)
        # B     = number of prompts
        # T_max = length of longest prompt

        input_ids = inputs["input_ids"].to(
            self.device
        )

        # attention_mask:
        # (B, T_max)
        # 1 = real token, 0 = padding

        attention_mask = inputs["attention_mask"].to(
            self.device
        )

        batch_size = input_ids.shape[0]

        # ---------------------------------------------------------
        # 3. Store all original/generated token IDs
        # ---------------------------------------------------------

        # We keep the complete token sequences for final decoding.
        all_token_ids = [
            input_ids[i].tolist()
            for i in range(batch_size)
        ]

        # ---------------------------------------------------------
        # 4. Set up memory monitoring
        # ---------------------------------------------------------

        process = psutil.Process()

        peak_ram_bytes = process.memory_info().rss

        # ---------------------------------------------------------
        # 5. Set up timing
        # ---------------------------------------------------------

        time_to_first_token_s = None

        inter_token_latencies_s = []

        generation_start = time.perf_counter()

        # ---------------------------------------------------------
        # 6. KV cache starts empty
        # ---------------------------------------------------------

        past_key_values = None

        # ---------------------------------------------------------
        # 7. Generation loop
        # ---------------------------------------------------------

        for step in range(max_new_tokens):

            step_start = time.perf_counter()

            # -----------------------------------------------------
            # PREFILL
            #
            # First iteration:
            # Calculate explicit position_ids so that left-padding
            # does not offset positional embeddings. First real token
            # gets index 0.
            # -----------------------------------------------------

            if past_key_values is None:

                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 0)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=True
                )

            # -----------------------------------------------------
            # DECODE
            #
            # Later iterations:
            # Position ID for the single new token is equal to the
            # total count of valid tokens so far minus 1.
            # -----------------------------------------------------

            else:

                position_ids = (
                    attention_mask.long().sum(dim=-1, keepdim=True) - 1
                )

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            # -----------------------------------------------------
            # 8. Get logits
            # -----------------------------------------------------

            logits = outputs.logits

            # -----------------------------------------------------
            # 9. Save updated KV cache
            # -----------------------------------------------------

            past_key_values = outputs.past_key_values

            # -----------------------------------------------------
            # 10. Greedy decoding
            # -----------------------------------------------------

            next_token = torch.argmax(
                logits[:, -1, :],
                dim=-1,
                keepdim=True
            )

            # Shape: (B, 1)

            # -----------------------------------------------------
            # 11. Save generated tokens
            # -----------------------------------------------------

            for i in range(batch_size):

                all_token_ids[i].append(
                    next_token[i, 0].item()
                )

            # -----------------------------------------------------
            # 12. Next decode step only needs the new tokens
            # -----------------------------------------------------

            input_ids = next_token

            # -----------------------------------------------------
            # 13. Update attention mask
            # -----------------------------------------------------

            new_attention = torch.ones(
                (batch_size, 1),
                dtype=attention_mask.dtype,
                device=self.device
            )

            attention_mask = torch.cat(
                [attention_mask, new_attention],
                dim=1
            )

            # -----------------------------------------------------
            # 14. Measure latency
            # -----------------------------------------------------

            step_latency = (
                time.perf_counter() - step_start
            )

            if step == 0:

                # First token for the whole batch.
                time_to_first_token_s = step_latency

            else:

                # Tokens 2..N.
                inter_token_latencies_s.append(
                    step_latency
                )

            # -----------------------------------------------------
            # 15. Measure memory
            # -----------------------------------------------------

            current_ram_bytes = (
                process.memory_info().rss
            )

            peak_ram_bytes = max(
                peak_ram_bytes,
                current_ram_bytes
            )

        # ---------------------------------------------------------
        # 16. Total generation time
        # ---------------------------------------------------------

        total_generation_time_s = (
            time.perf_counter() - generation_start
        )

        # ---------------------------------------------------------
        # 17. Calculate system throughput
        # ---------------------------------------------------------

        total_generated_tokens = (
            batch_size * max_new_tokens
        )

        tokens_per_sec = (
            total_generated_tokens
            / total_generation_time_s
        )

        # ---------------------------------------------------------
        # 18. Decode each sequence separately
        # ---------------------------------------------------------

        generated_texts = []

        for token_ids in all_token_ids:

            text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True
            )

            generated_texts.append(text)

        # ---------------------------------------------------------
        # 19. Convert RAM bytes → MB
        # ---------------------------------------------------------

        peak_ram_mb = (
            peak_ram_bytes
            / (1024 * 1024)
        )

        # ---------------------------------------------------------
        # 20. Return metrics
        # ---------------------------------------------------------

        return {
            "generated_texts": generated_texts,
            "time_to_first_token_s": time_to_first_token_s,
            "inter_token_latencies_s": (
                inter_token_latencies_s
            ),
            "tokens_per_sec": tokens_per_sec,
            "peak_ram_mb": peak_ram_mb,
        }