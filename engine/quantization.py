"""Milestone 2 — INT8/INT4 dynamic weight quantization (weight-only).

Scope and terminology
---------------------
This module implements **weight-only** quantization:

* Weights are quantized offline (once, at load time), symmetric per-channel
  for INT8 and symmetric per-group for INT4.
* Activations remain FP16 at inference. The ``dynamic`` in the milestone
  name refers to the per-call **dequantization** of weights inside
  :meth:`QuantizedLinear.forward` — *not* to the RNN-era convention
  (``torch.quantization.quantize_dynamic``) in which **activation** scales
  are computed at runtime. That is a different technique.

Weight-only is the right choice for a decode-bound CPU engine because the
per-token activation vector is tiny — 4096 FP16 elements = 8 KiB — while
the weight matrix dominates the bandwidth budget — 4096 × 4096 FP16 = 32
MiB. Compressing the activation vector saves nothing; compressing the
weight matrix saves 75 %.

Symmetric vs. asymmetric
------------------------
We use **symmetric** quantization (no zero-point). Rationale:

* Post-training LLM weights are approximately zero-centered per row /
  per group because they are trained against LayerNorm'd activations.
* Asymmetric quantization requires a zero-point tensor the same shape as
  ``scales`` — doubling metadata. For INT4 with ``G=128``, that takes the
  effective bit-width from 4.125 to 4.25 bits per weight.
* The sign bit is worth more than the offset for these distributions.

Effective bit-widths (including scales, FP16 metadata, excluding padding):

    FP16                                : 16.0   bits/weight
    INT8 per-channel                    :  8.015 bits/weight
    INT4 per-group G=128                :  4.125 bits/weight

Layouts
-------
* INT8 payload: ``int8 [out_features, in_features]``.
* INT8 scales:  ``fp16 [out_features]``.
* INT4 payload: ``uint8 [out_features, in_features_padded // 2]`` — two
  nibbles per byte, low nibble = even index, high nibble = odd index,
  two's-complement signed values in ``[-8, 7]``.
* INT4 scales:  ``fp16 [out_features, num_groups]`` where
  ``num_groups = in_features_padded // group_size``.

Cache-line note: for ``G=128`` and FP16 scales, each group's payload is
``128 × 0.5 = 64 B`` — exactly one cache line. This is why 128 is the
canonical group size. Non-128 group sizes produce partial-cache-line reads
at group boundaries.

Numerical expectations on synthetic Gaussian weights (measured below in
tests): INT8 per-channel L2 relative output error ~1 %; INT4 per-group
G=128 L2 relative output error ~15 %. Real LLM weights have heavier tails
than Gaussian, so the INT4 figure should be treated as a lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantizationError(ValueError):
    """Raised on invalid weight, scale, or group-size inputs."""


# ---------------------------------------------------------------------- #
# Container
# ---------------------------------------------------------------------- #


@dataclass
class QuantizedWeight:
    """A quantized weight tensor plus the metadata needed to dequantize it.

    Attributes
    ----------
    payload:
        Storage. ``int8`` for INT8, ``uint8`` (packed nibbles) for INT4.
    scales:
        FP16 scales. Shape ``[out_features]`` for INT8, or
        ``[out_features, num_groups]`` for INT4.
    scheme:
        ``"int8_per_channel"`` or ``"int4_per_group"``.
    original_shape:
        ``(out_features, in_features)``. Used to slice off right-padding
        introduced by the INT4 group alignment.
    group_size:
        ``G`` for INT4, ``None`` for INT8.
    in_features_padded:
        ``ceil(in_features / G) * G`` for INT4, ``None`` for INT8. Even by
        construction because ``G`` is required to be even.
    """

    payload: torch.Tensor
    scales: torch.Tensor
    scheme: str
    original_shape: Tuple[int, int]
    group_size: Optional[int] = None
    in_features_padded: Optional[int] = None

    def __post_init__(self) -> None:
        if self.scheme not in ("int8_per_channel", "int4_per_group"):
            raise QuantizationError(f"unknown scheme {self.scheme!r}")
        out_features, _ = self.original_shape
        if self.scales.shape[0] != out_features:
            raise QuantizationError(
                f"scales.shape[0] = {self.scales.shape[0]} does not match "
                f"out_features = {out_features}"
            )
        if self.scheme == "int8_per_channel":
            if self.payload.dtype != torch.int8:
                raise QuantizationError(
                    f"INT8 payload must be int8, got {self.payload.dtype}"
                )
            if self.scales.dim() != 1:
                raise QuantizationError(
                    f"INT8 scales must be 1-D [out_features], got "
                    f"{tuple(self.scales.shape)}"
                )
            if self.payload.shape != self.original_shape:
                raise QuantizationError(
                    f"INT8 payload shape {tuple(self.payload.shape)} != "
                    f"original_shape {self.original_shape}"
                )
        else:  # int4_per_group
            if self.payload.dtype != torch.uint8:
                raise QuantizationError(
                    f"INT4 payload must be uint8, got {self.payload.dtype}"
                )
            if self.group_size is None or self.in_features_padded is None:
                raise QuantizationError(
                    "INT4 requires group_size and in_features_padded"
                )
            expected_groups = self.in_features_padded // self.group_size
            if self.scales.shape != (out_features, expected_groups):
                raise QuantizationError(
                    f"INT4 scales shape {tuple(self.scales.shape)} != "
                    f"({out_features}, {expected_groups})"
                )
            expected_payload_cols = self.in_features_padded // 2
            if self.payload.shape != (out_features, expected_payload_cols):
                raise QuantizationError(
                    f"INT4 payload shape {tuple(self.payload.shape)} != "
                    f"({out_features}, {expected_payload_cols})"
                )

    def storage_bytes(self) -> int:
        """Bytes occupied by payload + scales (excludes Python object overhead)."""
        return (
            self.payload.numel() * self.payload.element_size()
            + self.scales.numel() * self.scales.element_size()
        )

    def dequantize(self, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Materialize an FP16 (or ``dtype``) tensor of ``original_shape``.

        This is a real allocation: the result is ``out_features ×
        in_features × itemsize(dtype)`` bytes. In Milestone 2 it is called
        once per forward; Milestone 7 will fuse it into the matmul.
        """
        if self.scheme == "int8_per_channel":
            w = _dequantize_int8_per_channel(self)
        else:
            w = _dequantize_int4_per_group(self)
        return w.to(dtype)


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def _validate_weight(w: torch.Tensor) -> None:
    if w.dim() != 2:
        raise QuantizationError(f"weight must be 2-D, got {w.dim()}-D")
    if w.shape[0] == 0 or w.shape[1] == 0:
        raise QuantizationError(
            f"weight must have non-zero dims, got {tuple(w.shape)}"
        )
    if not torch.isfinite(w).all():
        raise QuantizationError("weight contains non-finite values")


# ---------------------------------------------------------------------- #
# INT8 per-channel
# ---------------------------------------------------------------------- #


def quantize_int8_per_channel(w: torch.Tensor) -> QuantizedWeight:
    """Symmetric per-output-channel INT8 quantization.

    Scale per output row: ``s[r] = max(|W[r, :]|) / 127``, with a floor to
    avoid division by zero on all-zero rows. Quantized values live in
    ``[-127, 127]`` as ``int8`` (note: ``-128`` is unused so that the range
    is symmetric).
    """
    _validate_weight(w)
    w_fp32 = w.detach().to(torch.float32)
    max_abs = w_fp32.abs().amax(dim=1)  # [out_features]
    scales_fp32 = (max_abs / 127.0).clamp_min(1e-8)
    scales_fp16 = scales_fp32.to(torch.float16)
    # Quantize using the FP16-rounded scale, NOT scales_fp32. The error
    # bound |w - w_hat| <= 0.5 * s holds if and only if the quantization
    # scale equals the dequantization scale. Using scales_fp32 here while
    # storing scales_fp16 introduces an extra |q| * |s_fp32 - s_fp16|
    # term, up to ~127 * 2**-11 * s = 0.062 * s, which pushes the error
    # past the half-scale bound on some rows.
    scales_for_quant = scales_fp16.to(torch.float32)
    q = torch.round(w_fp32 / scales_for_quant.unsqueeze(1)).clamp(-127, 127)
    q_int8 = q.to(torch.int8)
    return QuantizedWeight(
        payload=q_int8,
        scales=scales_fp16,
        scheme="int8_per_channel",
        original_shape=(w.shape[0], w.shape[1]),
    )


def _dequantize_int8_per_channel(qw: QuantizedWeight) -> torch.Tensor:
    return (
        qw.payload.to(torch.float32)
        * qw.scales.to(torch.float32).unsqueeze(1)
    )


# ---------------------------------------------------------------------- #
# INT4 per-group
# ---------------------------------------------------------------------- #


def quantize_int4_per_group(
    w: torch.Tensor, group_size: int = 128
) -> QuantizedWeight:
    """Symmetric per-group INT4 quantization with nibble-packed payload.

    Parameters
    ----------
    w:
        Weight ``[out_features, in_features]``.
    group_size:
        ``G``. Must be positive and even (byte alignment of nibbles).
        The canonical choice is 128: 64 bytes of packed payload == one
        cache line.

    Padding
    -------
    If ``in_features`` is not a multiple of ``G``, the last group is
    right-padded with zeros. Padded weights quantize to zero and contribute
    nothing to the dequantized matmul; the padded columns are sliced off
    before returning from :meth:`QuantizedWeight.dequantize`.

    Packing convention
    ------------------
    For a row ``q[0..d-1]`` of signed values in ``[-8, 7]``::

        u  = q & 0x0F              # two's-complement to unsigned nibble
        b[i] = u[2i] | (u[2i+1] << 4)

    Decode::

        low  = b & 0x0F
        high = (b >> 4) & 0x0F
        q    = (u ^ 8) - 8         # two's-complement sign restore
    """
    _validate_weight(w)
    if group_size <= 0 or group_size % 2 != 0:
        raise QuantizationError(
            f"group_size must be a positive even integer, got {group_size}"
        )

    out_features, in_features = w.shape
    num_groups = (in_features + group_size - 1) // group_size
    in_features_padded = num_groups * group_size

    w_fp32 = w.detach().to(torch.float32)
    if in_features_padded > in_features:
        pad_cols = in_features_padded - in_features
        padding = torch.zeros(
            out_features, pad_cols, dtype=torch.float32, device=w.device
        )
        w_padded = torch.cat([w_fp32, padding], dim=1)
    else:
        w_padded = w_fp32

    w_grouped = w_padded.view(out_features, num_groups, group_size)
    max_abs = w_grouped.abs().amax(dim=-1)          # [out, num_groups]
    scales_fp32 = (max_abs / 7.0).clamp_min(1e-8)   # symmetric INT4: [-8, 7]
    scales_fp16 = scales_fp32.to(torch.float16)
    # Same reasoning as quantize_int8_per_channel: quantize with the
    # FP16-rounded scale so the half-scale error bound holds exactly.
    scales_for_quant = scales_fp16.to(torch.float32)
    q = torch.round(
        w_grouped / scales_for_quant.unsqueeze(-1)
    ).clamp(-8, 7).to(torch.int8)
    q = q.view(out_features, in_features_padded)

    # Two's-complement int8 -> unsigned nibble: mask with 0x0F. This works
    # because the int8 bit pattern of -k in [-8, -1] already ends in the
    # correct two's-complement nibble (e.g., -2 -> 0b11111110 -> & 0x0F = 0xE).
    q_uint = q.view(torch.uint8) & 0x0F
    q_pairs = q_uint.view(out_features, in_features_padded // 2, 2)
    packed = (q_pairs[..., 0] | (q_pairs[..., 1] << 4)).to(torch.uint8)

    return QuantizedWeight(
        payload=packed,
        scales=scales_fp32.to(torch.float16),
        scheme="int4_per_group",
        original_shape=(out_features, in_features),
        group_size=group_size,
        in_features_padded=in_features_padded,
    )


def _dequantize_int4_per_group(qw: QuantizedWeight) -> torch.Tensor:
    assert qw.group_size is not None and qw.in_features_padded is not None
    out_features, in_features = qw.original_shape
    in_padded = qw.in_features_padded
    G = qw.group_size

    packed = qw.payload  # uint8 [out, in_padded // 2]
    low = (packed & 0x0F).to(torch.int16)
    high = ((packed >> 4) & 0x0F).to(torch.int16)
    # Interleave: [low_0, high_0, low_1, high_1, ...]
    q_uint = torch.stack([low, high], dim=-1).reshape(
        out_features, in_padded
    )
    # Two's-complement sign restore: u >= 8 -> u - 16
    q = torch.where(q_uint >= 8, q_uint - 16, q_uint).to(torch.int8)

    scales_expanded = (
        qw.scales.to(torch.float32)
        .unsqueeze(-1)
        .expand(-1, -1, G)
        .reshape(out_features, in_padded)
    )
    w = q.to(torch.float32) * scales_expanded
    return w[:, :in_features]


# ---------------------------------------------------------------------- #
# Module wrapper
# ---------------------------------------------------------------------- #


class QuantizedLinear(nn.Module):
    """A drop-in replacement for ``nn.Linear`` with quantized weights.

    The forward pass:

    1. Materializes an FP16 weight via :meth:`QuantizedWeight.dequantize`.
    2. Runs an ordinary ``F.linear`` against the FP16 activation and bias.

    No kernel fusion happens here on purpose — we want Milestone 2's
    memory and latency numbers to reflect *dequantization cost paid in
    full*, so that Milestone 7's fused kernel can be measured as a strict
    improvement over this baseline.
    """

    def __init__(
        self,
        linear: nn.Linear,
        scheme: str = "int8",
        group_size: int = 128,
    ) -> None:
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise QuantizationError(
                f"expected nn.Linear, got {type(linear).__name__}"
            )
        if scheme not in ("int8", "int4"):
            raise QuantizationError(f"scheme must be 'int8' or 'int4', got {scheme!r}")

        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.scheme = scheme
        w = linear.weight.detach()
        if scheme == "int8":
            self.weight = quantize_int8_per_channel(w)
        else:
            self.weight = quantize_int4_per_group(w, group_size)

        if linear.bias is not None:
            self.register_buffer("bias", linear.bias.detach().clone())
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight.dequantize(dtype=x.dtype)
        return F.linear(x, w, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"scheme={self.scheme}, "
            f"storage_bytes={self.weight.storage_bytes()}"
        )