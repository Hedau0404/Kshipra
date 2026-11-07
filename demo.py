"""Kshipra end-to-end inference demo.

Usage:
    python demo.py --model Qwen/Qwen2.5-0.5B-Instruct --prompt "Explain KV cache." --max-new-tokens 40
    python demo.py --random   # no download; tiny random Llama to prove wiring
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM

from engine.model import KshipraModel


def _run_random() -> None:
    """Prove wiring is correct on a tiny random Llama — no model download needed."""
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    hf = LlamaForCausalLM(cfg).eval()
    tok = None

    km = KshipraModel(hf, quantize=None)

    prompt = [1, 42, 100, 201, 7]
    print(f"Prompt tokens: {prompt}")
    t0 = time.perf_counter()
    kshipra_out = km.generate(prompt, max_new_tokens=10, block_size=16)
    t_k = time.perf_counter() - t0
    print(f"Kshipra tokens: {kshipra_out}  ({t_k*1000:.1f} ms)")

    # HF reference
    t0 = time.perf_counter()
    hf_ids = torch.tensor([prompt])
    hf_out_full = hf.generate(
        hf_ids, max_new_tokens=10, do_sample=False, num_beams=1,
        pad_token_id=0, eos_token_id=None,
    )
    t_h = time.perf_counter() - t0
    hf_tokens = hf_out_full[0, len(prompt):].tolist()
    print(f"HF tokens:     {hf_tokens}  ({t_h*1000:.1f} ms)")

    if kshipra_out == hf_tokens:
        print("✓ PASSED: Kshipra output is token-for-token identical to HF generate")
    else:
        print("✗ FAILED: token mismatch")
        print(f"  first divergence at index {next(i for i,(a,b) in enumerate(zip(kshipra_out, hf_tokens)) if a!=b)}")


def _run_real(model_id: str, prompt: str, max_new_tokens: int, quantize: str | None) -> None:
    print(f"Loading {model_id} (quantize={quantize})...")
    t0 = time.perf_counter()
    hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    tok = AutoTokenizer.from_pretrained(model_id)
    km = KshipraModel(hf, quantize=quantize)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")

    input_ids = tok(prompt, return_tensors="pt").input_ids[0].tolist()
    print(f"Prompt: {prompt!r}")
    print(f"Prompt token count: {len(input_ids)}")

    t0 = time.perf_counter()
    out_ids = km.generate(input_ids, max_new_tokens=max_new_tokens, eos_token_id=tok.eos_token_id)
    elapsed = time.perf_counter() - t0

    text = tok.decode(out_ids, skip_special_tokens=True)
    print(f"\nOutput:\n  {text}\n")
    print(f"Generated {len(out_ids)} tokens in {elapsed:.2f}s ({len(out_ids)/elapsed:.2f} tok/s)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--random", action="store_true", help="Tiny random Llama, no download")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--prompt", default="Explain KV cache in one sentence.")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--quantize", default=None, choices=[None, "int8", "int4"])
    args = p.parse_args()

    if args.random:
        _run_random()
    else:
        _run_real(args.model, args.prompt, args.max_new_tokens, args.quantize)