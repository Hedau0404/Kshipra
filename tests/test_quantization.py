"""Unit tests for engine.quantization (Milestone 2).

Covers:
* Storage accounting (metadata fractions, effective bit-width).
* INT8 / INT4 round-trip error bounds on synthetic weights.
* Numerical equivalence of QuantizedLinear against unquantized nn.Linear.
* Edge cases: non-multiple-of-G in_features, G > in_features, d=1, o=1,
  odd group_size, non-finite weights.
* Isolated subprocess RSS: INT4 payload loads with materially lower peak
  than FP16, with both paths measured on freshly-spawned children.
"""

from __future__ import annotations

import gc
import sys

import pytest
import torch
import torch.nn as nn

from benchmarks.run_benchmarks import isolated_run
from engine.quantization import (
    QuantizedLinear,
    QuantizedWeight,
    QuantizationError,
    quantize_int4_per_group,
    quantize_int8_per_channel,
)


# ---------------------------------------------------------------------- #
# Storage accounting
# ---------------------------------------------------------------------- #


def test_int8_metadata_fraction():
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.float32)
    qw = quantize_int8_per_channel(w)
    fp16_payload = 4096 * 4096 * 2
    # payload + 4096 scales * 2 bytes
    assert qw.storage_bytes() == 4096 * 4096 * 1 + 4096 * 2
    metadata_ratio = (4096 * 2) / fp16_payload
    assert metadata_ratio == pytest.approx(8192 / 33554432, rel=1e-12)


def test_int4_metadata_fraction():
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.float32)
    qw = quantize_int4_per_group(w, group_size=128)
    out_features, in_features = w.shape
    num_groups = in_features // 128
    expected = out_features * (in_features // 2) + out_features * num_groups * 2
    assert qw.storage_bytes() == expected
    # 256 KiB metadata over 8 MiB payload == 3.125%
    metadata = out_features * num_groups * 2
    payload = out_features * (in_features // 2)
    assert metadata / payload == pytest.approx(0.03125, rel=1e-12)


def test_int4_total_is_under_one_third_of_fp16():
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.float32)
    fp16_bytes = w.numel() * 2
    qw = quantize_int4_per_group(w, group_size=128)
    assert qw.storage_bytes() < fp16_bytes / 3


def test_effective_bit_width_is_documented():
    """4.125 bits/weight for INT4 G=128, matching the module docstring."""
    torch.manual_seed(0)
    w = torch.randn(4096, 4096, dtype=torch.float32)
    qw = quantize_int4_per_group(w, group_size=128)
    bits_per_weight = qw.storage_bytes() * 8 / w.numel()
    assert bits_per_weight == pytest.approx(4.125, abs=1e-3)


# ---------------------------------------------------------------------- #
# Round-trip error bounds
# ---------------------------------------------------------------------- #


def test_int8_roundtrip_max_error_bounded_by_half_scale():
    torch.manual_seed(1)
    w = torch.randn(64, 512, dtype=torch.float32)
    qw = quantize_int8_per_channel(w)
    w_hat = qw.dequantize(dtype=torch.float32)
    # Per-row bound: |w - w_hat| <= scale / 2
    scales = qw.scales.to(torch.float32)
    max_err_per_row = (w - w_hat).abs().amax(dim=1)
    assert torch.all(max_err_per_row <= scales / 2 + 1e-6)


def test_int4_roundtrip_max_error_bounded_by_half_group_scale():
    torch.manual_seed(2)
    w = torch.randn(32, 512, dtype=torch.float32)
    qw = quantize_int4_per_group(w, group_size=128)
    w_hat = qw.dequantize(dtype=torch.float32)
    # Per-group bound: |w - w_hat| <= s_g / 2 within each group
    scales = qw.scales.to(torch.float32)  # [32, 4]
    err = (w - w_hat).abs().view(32, 4, 128)
    max_err = err.amax(dim=-1)  # [32, 4]
    assert torch.all(max_err <= scales / 2 + 1e-6)


def test_int4_padding_slice_is_exact():
    """When in_features is not a multiple of G, padded columns must be sliced."""
    torch.manual_seed(3)
    w = torch.randn(16, 500, dtype=torch.float32)  # 500 = 3*128 + 116
    qw = quantize_int4_per_group(w, group_size=128)
    assert qw.in_features_padded == 512
    w_hat = qw.dequantize(dtype=torch.float32)
    assert w_hat.shape == (16, 500)
    # Relative Frobenius error is bounded; loose bound because 500 is small.
    rel = (w_hat - w).norm() / w.norm()
    assert rel < 0.25


# ---------------------------------------------------------------------- #
# Linear equivalence
# ---------------------------------------------------------------------- #


def _relative_l2(y_ref: torch.Tensor, y: torch.Tensor) -> float:
    return float((y - y_ref).norm() / y_ref.norm().clamp_min(1e-12))


def test_int8_linear_matches_fp16_reference():
    torch.manual_seed(4)
    lin = nn.Linear(512, 512, bias=True)
    x = torch.randn(16, 512)
    with torch.no_grad():
        y_ref = lin(x)
    qlin = QuantizedLinear(lin, scheme="int8")
    with torch.no_grad():
        y = qlin(x)
    rel = _relative_l2(y_ref, y)
    assert rel < 2e-2, f"INT8 relative L2 error {rel:.4f} exceeds 2%"


def test_int4_linear_matches_fp16_reference():
    torch.manual_seed(5)
    lin = nn.Linear(512, 512, bias=True)
    x = torch.randn(16, 512)
    with torch.no_grad():
        y_ref = lin(x)
    qlin = QuantizedLinear(lin, scheme="int4", group_size=128)
    with torch.no_grad():
        y = qlin(x)
    rel = _relative_l2(y_ref, y)
    # 15-20% is expected for random Gaussian weights on INT4 with G=128.
    assert rel < 2.5e-1, f"INT4 relative L2 error {rel:.4f} exceeds 25%"


def test_int4_tighter_group_beats_looser_group():
    """Smaller G should strictly reduce L2 error."""
    torch.manual_seed(6)
    lin = nn.Linear(512, 512, bias=False)
    x = torch.randn(16, 512)
    with torch.no_grad():
        y_ref = lin(x)
    qlin_32 = QuantizedLinear(lin, scheme="int4", group_size=32)
    qlin_256 = QuantizedLinear(lin, scheme="int4", group_size=256)
    with torch.no_grad():
        rel_32 = _relative_l2(y_ref, qlin_32(x))
        rel_256 = _relative_l2(y_ref, qlin_256(x))
    assert rel_32 < rel_256, (
        f"G=32 should beat G=256, got {rel_32:.4f} vs {rel_256:.4f}"
    )


def test_linear_without_bias():
    torch.manual_seed(7)
    lin = nn.Linear(128, 256, bias=False)
    x = torch.randn(4, 128)
    qlin = QuantizedLinear(lin, scheme="int8")
    with torch.no_grad():
        y_ref = lin(x)
        y = qlin(x)
    assert _relative_l2(y_ref, y) < 2e-2


# ---------------------------------------------------------------------- #
# Edge cases
# ---------------------------------------------------------------------- #


def test_group_size_larger_than_in_features():
    torch.manual_seed(8)
    w = torch.randn(16, 100, dtype=torch.float32)
    qw = quantize_int4_per_group(w, group_size=128)
    assert qw.in_features_padded == 128
    w_hat = qw.dequantize(dtype=torch.float32)
    assert w_hat.shape == w.shape
    rel = (w_hat - w).norm() / w.norm()
    assert rel < 0.25


def test_odd_group_size_rejected():
    w = torch.randn(4, 128, dtype=torch.float32)
    with pytest.raises(QuantizationError):
        quantize_int4_per_group(w, group_size=127)


def test_zero_group_size_rejected():
    w = torch.randn(4, 128, dtype=torch.float32)
    with pytest.raises(QuantizationError):
        quantize_int4_per_group(w, group_size=0)


def test_one_by_one_weight():
    w = torch.tensor([[1.5]], dtype=torch.float32)
    qw = quantize_int8_per_channel(w)
    w_hat = qw.dequantize(dtype=torch.float32)
    assert w_hat.shape == (1, 1)
    rel = float((w_hat - w).abs().item() / w.abs().item())
    assert rel < 1e-3  # single weight maps exactly to 127 * s / s = 1.0 * w


def test_in_features_one_int4():
    w = torch.tensor([[0.5], [-0.3]], dtype=torch.float32)
    qw = quantize_int4_per_group(w, group_size=128)
    w_hat = qw.dequantize(dtype=torch.float32)
    assert w_hat.shape == (2, 1)


def test_non_finite_weight_rejected():
    w = torch.tensor([[1.0, float("nan")]], dtype=torch.float32)
    with pytest.raises(QuantizationError):
        quantize_int8_per_channel(w)


def test_zero_weight_row():
    """All-zero row must not cause division by zero (scale floored)."""
    w = torch.zeros(4, 128, dtype=torch.float32)
    qw = quantize_int8_per_channel(w)
    w_hat = qw.dequantize(dtype=torch.float32)
    assert torch.all(w_hat == 0)


def test_bad_scheme_rejected():
    lin = nn.Linear(8, 8)
    with pytest.raises(QuantizationError):
        QuantizedLinear(lin, scheme="int3")


# ---------------------------------------------------------------------- #
# Isolated memory measurement
# ---------------------------------------------------------------------- #


def _touch_pages(t: torch.Tensor) -> None:
    """Force every 4 KiB page of ``t`` resident with minimal transient allocation.

    ``t.view(torch.uint8).view(-1)[::4096]`` selects exactly one byte per
    page, which is sufficient to fault the whole tensor into RSS. We then
    reduce that tiny view, so the intermediate allocation is
    ``O(numel / 4096)`` bytes instead of ``O(numel)``.

    Critically, we do **not** call ``t.sum()``: on integer dtypes, PyTorch
    upcasts to ``int64`` for accumulation, which would allocate an
    intermediate ``8 * numel``-byte tensor and swamp the measurement.
    """
    _ = t.view(torch.uint8).view(-1)[::4096].sum().item()


def _child_load_fp16(path: str) -> int:
    w = torch.load(path, map_location="cpu", weights_only=False)
    _touch_pages(w)
    return 0


def _child_load_int4(path: str) -> int:
    d = torch.load(path, map_location="cpu", weights_only=False)
    assert isinstance(d, dict) and "payload" in d
    _touch_pages(d["payload"])
    _touch_pages(d["scales"])
    return 0


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_int4_payload_loads_with_lower_peak_than_fp16(tmp_path):
    """INT4 payload on disk must occupy at least 50 MiB less peak RSS than
    the FP16 tensor it was derived from.

    Two distinct assertions:

    1. **File size.** The INT4 file on disk is smaller than the FP16 file.
       This catches a class of bugs (e.g., accidentally saving the original
       FP16 tensor alongside the packed payload) before we pay the cost of
       two subprocess spawns.
    2. **Delta RSS.** The child that loads the INT4 file grows by less RSS
       than the child that loads the FP16 file. We compare *deltas*
       (peak − baseline), not absolute peaks, because the two children
       share a post-import baseline in principle but small transient
       differences from torch.load internals can otherwise flake the test.
    """
    import os

    torch.manual_seed(0)
    w = torch.randn(8192, 8192, dtype=torch.float16)
    fp16_path = tmp_path / "w_fp16.pt"
    int4_path = tmp_path / "w_int4.pt"
    torch.save(w, str(fp16_path))
    qw = quantize_int4_per_group(w, group_size=128)
    # Save a *dict of raw tensors*, not the dataclass. See the docstring
    # of _child_load_int4 for the reason.
    torch.save({"payload": qw.payload, "scales": qw.scales}, str(int4_path))

    fp16_size = os.path.getsize(str(fp16_path))
    int4_size = os.path.getsize(str(int4_path))
    print(
        f"\n  fp16 file: {fp16_size / 1024**2:.1f} MiB\n"
        f"  int4 file: {int4_size / 1024**2:.1f} MiB"
    )
    # Assertion 1: on-disk payload must actually be smaller.
    assert fp16_size > int4_size + 50 * 1024 * 1024, (
        f"INT4 file should be at least 50 MiB smaller on disk: "
        f"fp16={fp16_size / 1024**2:.1f} MiB, "
        f"int4={int4_size / 1024**2:.1f} MiB"
    )

    del w, qw
    gc.collect()

    fp16_res = isolated_run(
        _child_load_fp16, {"path": str(fp16_path)}, timeout_s=180.0
    )
    int4_res = isolated_run(
        _child_load_int4, {"path": str(int4_path)}, timeout_s=180.0
    )

    fp16_growth = fp16_res.delta_bytes
    int4_growth = int4_res.delta_bytes
    gap_bytes = fp16_growth - int4_growth
    assert gap_bytes > 50 * 1024 * 1024, (
        f"INT4 delta did not beat FP16 delta by 50 MiB:\n"
        f"  fp16: peak={fp16_res.peak_gib:.3f} GiB, "
        f"baseline={fp16_res.baseline_bytes / 1024**2:.1f} MiB, "
        f"delta={fp16_growth / 1024**2:.1f} MiB\n"
        f"  int4: peak={int4_res.peak_gib:.3f} GiB, "
        f"baseline={int4_res.baseline_bytes / 1024**2:.1f} MiB, "
        f"delta={int4_growth / 1024**2:.1f} MiB\n"
        f"  gap: {gap_bytes / 1024**2:.1f} MiB"
    )


@pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="isolated_run requires Linux /proc/self/status",
)
def test_quantized_linear_forward_under_4gib_peak_rss():
    res = isolated_run(
        _workload_quantized_linear_forward, timeout_s=180.0
    )
    assert res.result["int8_rel"] < 2e-2
    assert res.result["int4_rel"] < 2.5e-1
    assert res.peak_bytes < 4 * 1024 ** 3


def _workload_quantized_linear_forward() -> dict:
    torch.manual_seed(0)
    lin = nn.Linear(2048, 2048, bias=True)
    x = torch.randn(8, 2048)
    with torch.no_grad():
        y_ref = lin(x)
    q8 = QuantizedLinear(lin, scheme="int8")
    q4 = QuantizedLinear(lin, scheme="int4", group_size=128)
    with torch.no_grad():
        y8 = q8(x)
        y4 = q4(x)
    rel8 = float((y8 - y_ref).norm() / y_ref.norm().clamp_min(1e-12))
    rel4 = float((y4 - y_ref).norm() / y_ref.norm().clamp_min(1e-12))
    return {"int8_rel": rel8, "int4_rel": rel4}