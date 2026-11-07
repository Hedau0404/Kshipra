"""Milestone 1 — Single-sequence baseline, chat templating, generation loop.

This module provides:

* :func:`apply_chat_template` — chat formatting with a deterministic fallback
  for tokenizers that do not ship a ``chat_template`` Jinja string.
* :class:`GenerationConfig` — the sampling / stopping surface for generation.
* :class:`BaselineEngine` — a thin wrapper around a HuggingFace causal LM
  that (a) runs the *reference* greedy loop we will equivalence-test against,
  and (b) hands out a :class:`~engine.kv_cache.KVCache` sized from the model
  config, so later milestones can swap HF's cache for ours without changing
  the calling convention.

Device policy
-------------
Every public entry point takes a ``device: torch.device``. Default is CPU.
No CUDA-only ops are used; ``model.to(device)`` is the only device transfer
and it is idempotent on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch

from engine.kv_cache import KVCache, KVConfig


class ContextOverflowError(RuntimeError):
    """Raised when generation would exceed the model's positional capacity."""


@dataclass(frozen=True)
class GenerationConfig:
    """Sampling and stopping surface for :meth:`BaselineEngine.generate_ids`.

    Defaults are *greedy decoding*. Sampling is intentionally delegated to
    HuggingFace's ``generate`` in Milestone 1 so that later milestones can
    equivalence-test against a stable reference; our own sampler lands in
    Milestone 6 alongside speculative decoding.

    Attributes
    ----------
    max_new_tokens:
        Hard cap on newly generated tokens. The engine never generates
        more than this even if EOS is not reached.
    do_sample:
        If ``False``, greedy decode. If ``True``, delegate to HF's
        ``generate`` with the parameters below.
    temperature, top_p, top_k:
        Sampling controls. Ignored when ``do_sample=False``.
    seed:
        Optional CPU generator seed applied before sampling. Greedy decode
        ignores this (it is deterministic).
    eos_token_id:
        If set, generation stops as soon as this id is produced. If
        ``None``, fall back to ``model.config.eos_token_id``; if that is
        also ``None``, generation never stops early and always produces
        exactly ``max_new_tokens`` new tokens.
    """

    max_new_tokens: int = 64
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    seed: Optional[int] = None
    eos_token_id: Optional[int] = None


ChatMessage = Dict[str, str]


def apply_chat_template(
    tokenizer: Any,
    messages: Sequence[ChatMessage],
    add_generation_prompt: bool = True,
) -> str:
    """Render ``messages`` to a prompt string using ``tokenizer``'s template.

    If ``tokenizer.chat_template`` is a non-empty string, delegate to
    ``tokenizer.apply_chat_template`` (the HuggingFace path). Otherwise use
    a deterministic fallback of the form::

        Role: content
        Role: content
        Assistant:

    The fallback is intentionally trivial so that unit tests can exercise
    the prompt-construction pipeline without needing a real tokenizer
    download.

    Parameters
    ----------
    tokenizer:
        Any object with an optional ``chat_template`` attribute and,
        preferably, an ``apply_chat_template`` method.
    messages:
        Sequence of ``{"role": ..., "content": ...}`` dicts.
    add_generation_prompt:
        Whether to append the assistant-turn opener.

    Returns
    -------
    str
        The rendered prompt. Never raises on well-formed input.
    """
    if not messages:
        raise ValueError("messages must be a non-empty sequence")
    template = getattr(tokenizer, "chat_template", None)
    if template:
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for m in messages:
        if "role" not in m or "content" not in m:
            raise ValueError(f"message missing role/content: {m!r}")
        parts.append(f"{m['role'].capitalize()}: {m['content']}")
    if add_generation_prompt:
        parts.append("Assistant:")
    return "\n".join(parts)


class BaselineEngine:
    """Single-sequence reference engine.

    Owns a HuggingFace causal LM and its tokenizer. Provides:

    * :meth:`generate_ids` — greedy or sampled generation with a running
      HF ``past_key_values`` cache.
    * :meth:`chat` — chat-templated, single-turn convenience entry.
    * :meth:`build_kv_cache` — construct a :class:`KVCache` sized from the
      model config, for later milestones that own the forward pass.

    Numerics
    --------
    The greedy loop in :meth:`_generate_greedy` is guaranteed bit-identical
    to ``model.generate(do_sample=False, num_beams=1)`` under the same
    weights, dtype, and device — this is the equivalence contract our test
    suite asserts.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.model = model.to(device=device, dtype=dtype).eval()
        self.tokenizer = tokenizer
        self._max_pos: Optional[int] = getattr(
            self.model.config, "max_position_embeddings", None
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: str,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float16,
        **kwargs: Any,
    ) -> "BaselineEngine":
        """Load model + tokenizer from the HF hub or a local path."""
        # Imported lazily so that tests which inject a pre-built model do
        # not pay the transformers import at module load.
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, **kwargs)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, **kwargs
        )
        return cls(model, tokenizer, device=device, dtype=dtype)

    # ------------------------------------------------------------------ #
    # Cache sizing
    # ------------------------------------------------------------------ #

    def build_kv_cache(self, max_context_len: int) -> KVCache:
        """Build a :class:`KVCache` whose shape is derived from the model.

        Uses ``num_key_value_heads`` when present (GQA) and falls back to
        ``num_attention_heads`` (MHA). ``head_dim`` is read directly when
        the config exposes it and derived from
        ``hidden_size // num_attention_heads`` otherwise.
        """
        cfg = self.model.config
        num_layers = int(cfg.num_hidden_layers)
        num_attention_heads = int(cfg.num_attention_heads)
        num_kv_heads = int(
            getattr(cfg, "num_key_value_heads", num_attention_heads)
        )
        head_dim = int(
            getattr(cfg, "head_dim", cfg.hidden_size // num_attention_heads)
        )
        return KVCache(
            KVConfig(
                num_layers=num_layers,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                max_context_len=max_context_len,
                dtype=self.dtype,
                device=self.device,
            )
        )

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def generate_ids(
        self, input_ids: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """Generate from a batch-of-1 prompt.

        Parameters
        ----------
        input_ids:
            Long tensor of shape ``[1, S]``. Batch is fixed at 1 in
            Milestone 1; continuous batching arrives in Milestone 5.
        config:
            See :class:`GenerationConfig`.

        Returns
        -------
        torch.Tensor
            Long tensor of shape ``[1, S + G]`` where ``G <=
            config.max_new_tokens`` is the number of tokens actually
            generated (shorter if EOS was hit).

        Raises
        ------
        ValueError
            If ``input_ids`` is not a ``[1, S]`` long tensor.
        ContextOverflowError
            If the prompt already exceeds ``max_position_embeddings``, or
            if any decode step would.
        """
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(
                f"BaselineEngine expects input_ids of shape [1, S], got "
                f"{tuple(input_ids.shape)}"
            )
        if input_ids.dtype != torch.long:
            raise ValueError(
                f"input_ids must be torch.long, got {input_ids.dtype}"
            )
        input_ids = input_ids.to(self.device)
        if self._max_pos is not None and input_ids.shape[1] > self._max_pos:
            raise ContextOverflowError(
                f"prompt length {input_ids.shape[1]} exceeds "
                f"max_position_embeddings={self._max_pos}"
            )
        if config.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be >= 0")

        if config.do_sample:
            return self._generate_sampled(input_ids, config)
        return self._generate_greedy(input_ids, config)

    def _generate_greedy(
        self, input_ids: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """Manual greedy loop, numerically equivalent to ``model.generate``.

        Runs an incremental forward: the first step consumes the full
        prompt and produces the HF KV cache; subsequent steps feed one
        token at a time and reuse the cache. This is the pattern we will
        replace with our own :class:`~engine.kv_cache.KVCache` in
        Milestones 4-5.
        """
        eos = config.eos_token_id
        if eos is None:
            eos = getattr(self.model.config, "eos_token_id", None)
        eos_set: set[int] = set()
        if isinstance(eos, (list, tuple)):
            eos_set = {int(x) for x in eos if x is not None}
        elif eos is not None:
            eos_set = {int(eos)}

        generated = input_ids
        cur = input_ids
        past = None
        for _ in range(config.max_new_tokens):
            if (
                self._max_pos is not None
                and generated.shape[1] + 1 > self._max_pos
            ):
                raise ContextOverflowError(
                    f"generation would produce token at position "
                    f"{generated.shape[1]} which exceeds "
                    f"max_position_embeddings={self._max_pos}"
                )
            out = self.model(
                input_ids=cur, past_key_values=past, use_cache=True
            )
            past = out.past_key_values
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_id], dim=-1)
            if int(next_id.item()) in eos_set:
                break
            cur = next_id
        return generated

    def _generate_sampled(
        self, input_ids: torch.Tensor, config: GenerationConfig
    ) -> torch.Tensor:
        """Delegate sampling to HF's ``generate`` for reference parity."""
        gen_kwargs: Dict[str, Any] = dict(
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            temperature=config.temperature,
            top_p=config.top_p,
            pad_token_id=getattr(self.tokenizer, "pad_token_id", 0),
        )
        if config.top_k > 0:
            gen_kwargs["top_k"] = config.top_k
        if config.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = config.eos_token_id
        if config.seed is not None:
            torch.manual_seed(config.seed)
        return self.model.generate(input_ids, **gen_kwargs)

    # ------------------------------------------------------------------ #
    # Chat convenience
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def chat(
        self, messages: Sequence[ChatMessage], config: GenerationConfig
    ) -> str:
        """Render ``messages``, generate, and decode only the new tokens.

        Tokenizer notes
        ---------------
        We tokenize the *string* prompt ourselves rather than relying on
        ``apply_chat_template(tokenize=True)`` so that the fallback path
        (which returns a string) and the HF path share one code path.
        """
        prompt = apply_chat_template(
            self.tokenizer, messages, add_generation_prompt=True
        )
        enc = self.tokenizer(prompt, return_tensors="pt")
        ids = enc.input_ids.to(self.device)
        out = self.generate_ids(ids, config)
        new_ids = out[0, ids.shape[1]:]
        return self.tokenizer.decode(
            new_ids.tolist(), skip_special_tokens=True
        )