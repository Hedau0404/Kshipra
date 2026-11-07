"""Unit tests for engine.baseline (Milestone 1).

Covers:
* Chat-template native path and deterministic fallback.
* Greedy-decode equivalence against ``model.generate``.
* EOS-based early stopping.
* Context-overflow guards (prompt > max_position_embeddings, and decode
  step crossing the ceiling at 1024 tokens).
* KV cache sizing from the model config.
* Isolated peak-RSS bound under 4 GiB for end-to-end generation.

The tiny Llama config here is GQA (4 query heads, 2 KV heads) so the
``num_kv_heads`` path in :meth:`BaselineEngine.build_kv_cache` is exercised.
"""

from __future__ import annotations

import sys

import pytest
import torch

from benchmarks.run_benchmarks import isolated_run
from engine.baseline import (
    BaselineEngine,
    ContextOverflowError,
    GenerationConfig,
    apply_chat_template,
)


# ---------------------------------------------------------------------- #
# Test doubles
# ---------------------------------------------------------------------- #


def _make_tiny_llama(seed: int = 0, eos_token_id=None):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        pad_token_id=0,
        eos_token_id=eos_token_id,
        bos_token_id=1,
    )
    model = LlamaForCausalLM(cfg)
    model.eval()
    return model


class _Enc:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class _FakeTokenizer:
    """Minimal tokenizer sufficient for chat-template and ID plumbing.

    ``with_native_template=True`` simulates a tokenizer that ships a
    ``chat_template`` string. ``False`` exercises the fallback path in
    :func:`apply_chat_template`.
    """

    def __init__(self, vocab_size: int = 128, with_native_template: bool = False):
        self.vocab_size = vocab_size
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.chat_template = (
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n"
            "{% endfor %}{% if add_generation_prompt %}assistant:{% endif %}"
            if with_native_template
            else None
        )

    def __call__(self, text: str, return_tensors=None):
        ids = [max(2, ord(c) % self.vocab_size) for c in text][:256] or [2]
        if return_tensors == "pt":
            return _Enc(torch.tensor([ids], dtype=torch.long))
        return _Enc(ids)

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True
    ):
        assert not tokenize, "tests exercise the string path"
        out = []
        for m in messages:
            out.append(f"{m['role']}: {m['content']}\n")
        if add_generation_prompt:
            out.append("assistant:")
        return "".join(out)

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(chr(max(32, int(i))) for i in ids)


# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    model = _make_tiny_llama(seed=1234, eos_token_id=None)
    tok = _FakeTokenizer()
    return model, tok


@pytest.fixture(scope="module")
def engine(tiny_model_and_tokenizer):
    model, tok = tiny_model_and_tokenizer
    return BaselineEngine(
        model, tok, device=torch.device("cpu"), dtype=torch.float32
    )


# ---------------------------------------------------------------------- #
# Chat templating
# ---------------------------------------------------------------------- #


def test_chat_template_fallback():
    tok = _FakeTokenizer(with_native_template=False)
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_chat_template(tok, msgs, add_generation_prompt=True)
    assert "User: hi" in out
    assert out.rstrip().endswith("Assistant:")


def test_chat_template_native_path():
    tok = _FakeTokenizer(with_native_template=True)
    msgs = [{"role": "user", "content": "hi"}]
    out = apply_chat_template(tok, msgs, add_generation_prompt=True)
    assert "user: hi" in out
    assert "assistant:" in out


def test_chat_template_without_generation_prompt():
    tok = _FakeTokenizer(with_native_template=False)
    msgs = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
    ]
    out = apply_chat_template(tok, msgs, add_generation_prompt=False)
    assert "Assistant:" not in out


def test_chat_template_empty_messages_raises():
    tok = _FakeTokenizer()
    with pytest.raises(ValueError):
        apply_chat_template(tok, [])


# ---------------------------------------------------------------------- #
# Numerical equivalence
# ---------------------------------------------------------------------- #


def test_greedy_matches_hf_baseline(engine):
    """Bit-identical greedy decode vs. ``model.generate``."""
    model = engine.model
    ids = torch.tensor([[5, 6, 7, 8, 9]], dtype=torch.long)
    cfg = GenerationConfig(
        max_new_tokens=12, do_sample=False, eos_token_id=None
    )

    with torch.no_grad():
        ours = engine.generate_ids(ids, cfg)
        ref = model.generate(
            ids,
            max_new_tokens=12,
            do_sample=False,
            num_beams=1,
            eos_token_id=None,
            pad_token_id=0,
        )

    assert ours.shape == ref.shape, (
        f"shape mismatch: ours={tuple(ours.shape)} ref={tuple(ref.shape)}"
    )
    assert torch.equal(ours, ref), (
        f"greedy mismatch:\n ours={ours.tolist()}\n ref ={ref.tolist()}"
    )


def test_greedy_zero_new_tokens_returns_prompt(engine):
    ids = torch.tensor([[5, 6, 7]], dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=0, do_sample=False)
    out = engine.generate_ids(ids, cfg)
    assert torch.equal(out, ids)


def test_greedy_stops_at_eos():
    model = _make_tiny_llama(seed=42, eos_token_id=None)
    tok = _FakeTokenizer()
    eng = BaselineEngine(
        model, tok, device=torch.device("cpu"), dtype=torch.float32
    )

    ids = torch.tensor([[10, 20, 30]], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=ids, use_cache=False).logits[:, -1, :]
    first = int(logits.argmax(dim=-1).item())

    cfg = GenerationConfig(
        max_new_tokens=8, do_sample=False, eos_token_id=first
    )
    out = eng.generate_ids(ids, cfg)
    assert out.shape[1] == ids.shape[1] + 1, (
        f"expected exactly 1 new token, got {out.shape[1] - ids.shape[1]}"
    )


# ---------------------------------------------------------------------- #
# Context overflow
# ---------------------------------------------------------------------- #


def test_context_overflow_at_1024():
    """Prompt of exactly 1024 tokens; any new token overflows."""
    model = _make_tiny_llama(seed=7, eos_token_id=None)
    model.config.max_position_embeddings = 1024
    tok = _FakeTokenizer()
    eng = BaselineEngine(
        model, tok, device=torch.device("cpu"), dtype=torch.float32
    )

    ids = torch.randint(2, 100, (1, 1024), dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False, eos_token_id=None)
    with pytest.raises(ContextOverflowError):
        eng.generate_ids(ids, cfg)


def test_prompt_exceeding_max_position_embeddings_raises():
    model = _make_tiny_llama(seed=8, eos_token_id=None)
    model.config.max_position_embeddings = 512
    tok = _FakeTokenizer()
    eng = BaselineEngine(
        model, tok, device=torch.device("cpu"), dtype=torch.float32
    )

    ids = torch.randint(2, 100, (1, 513), dtype=torch.long)
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False)
    with pytest.raises(ContextOverflowError):
        eng.generate_ids(ids, cfg)


def test_generate_ids_rejects_wrong_shape(engine):
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False)
    bad = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    with pytest.raises(ValueError):
        engine.generate_ids(bad, cfg)


def test_generate_ids_rejects_non_long(engine):
    cfg = GenerationConfig(max_new_tokens=1, do_sample=False)
    bad = torch.tensor([[1, 2]], dtype=torch.float32)
    with pytest.raises(ValueError):
        engine.generate_ids(bad, cfg)


# ---------------------------------------------------------------------- #
# Cache sizing from model config
# ---------------------------------------------------------------------- #


def test_build_kv_cache_reflects_model_config(engine):
    cache = engine.build_kv_cache(max_context_len=128)
    assert cache.config.num_layers == 2
    assert cache.config.num_kv_heads == 2  # GQA, NOT num_attention_heads=4
    assert cache.config.head_dim == 64 // 4  # LlamaConfig-derived head_dim
    assert cache.capacity == 128
    assert cache.mem_bytes() == cache.config.total_bytes()


# ---------------------------------------------------------------------- #
# Chat plumbing
# ---------------------------------------------------------------------- #


def test_chat_returns_string(engine):
    cfg = GenerationConfig(max_new_tokens=4, do_sample=False, eos_token_id=None)
    out = engine.chat([{"role": "user", "content": "hello"}], cfg)
    assert isinstance(out, str)


# ---------------------------------------------------------------------- #
# Isolated memory bound — full generation path
# ---------------------------------------------------------------------- #


def _workload_generate_tiny() -> dict:
    model = _make_tiny_llama(seed=0, eos_token_id=None)
    tok = _FakeTokenizer()
    eng = BaselineEngine(
        model, tok, device=torch.device("cpu"), dtype=torch.float32
    )
    ids = torch.tensor([[5, 6, 7, 8]], dtype=torch.long)
    cfg = GenerationConfig(
        max_new_tokens=32, do_sample=False, eos_token_id=None
    )
    out = eng.generate_ids(ids, cfg)
    return {
        "shape": list(out.shape),
        "checksum": float(out.float().sum().item()),
    }


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_generation_under_4gib_peak_rss():
    res = isolated_run(_workload_generate_tiny, timeout_s=180.0)
    assert res.result["shape"][0] == 1
    assert res.result["shape"][1] == 4 + 32
    assert res.peak_bytes < 4 * 1024 ** 3, (
        f"peak RSS {res.peak_gib:.3f} GiB exceeded 4 GiB ceiling; "
        f"baseline {res.baseline_bytes / 1024**2:.1f} MiB, "
        f"delta {res.delta_mib:.1f} MiB"
    )