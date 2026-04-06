import pytest
import torch
import torch.nn.functional as F

from flashinfer import (
    autotune,
    dual_weight_mm,
    dual_weight_mm_e5m2,
    dual_weight_mm_e5m2_trunc,
    dual_weight_mm_sm90,
    dual_weight_mm_sm90_e5m2,
    dual_weight_mm_sm90_e5m2_trunc,
    prepare_dual_weight_mm_weights,
    prepare_dual_weight_mm_weights_e5m2,
)
from flashinfer.utils import get_compute_capability


# --- helpers ---


def _make_column_major(x: torch.Tensor) -> torch.Tensor:
    rows, cols = x.shape
    y = torch.empty_strided(
        (cols, rows),
        (1, cols),
        device=x.device,
        dtype=x.dtype,
    )
    y_row_major = torch.as_strided(y, size=(rows, cols), stride=(y.stride(1), 1))
    y_row_major.copy_(x)
    return y


def _make_finite_bf16_cast_fp16_weights(numel: int, device: str = "cuda") -> torch.Tensor:
    sign = torch.randint(0, 2, (numel,), dtype=torch.int32, device=device)
    # BF16 exponent 112-126 maps to FP16 exponent range ~1e-5 to ~1.0,
    # keeping magnitudes small enough to avoid FP16 overflow after GEMM.
    exp = torch.randint(112, 127, (numel,), dtype=torch.int32, device=device)
    mant = torch.randint(0, 128, (numel,), dtype=torch.int32, device=device)
    raw = ((sign << 15) | (exp << 7) | mant).to(torch.uint16)
    return raw.view(dtype=torch.bfloat16).to(torch.float16)


def pack_fp16_to_dual_fp8(
    fp16_tensor: torch.Tensor,
    *,
    column_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = fp16_tensor.view(torch.int16)
    lower = raw & 0x00FF
    upper = (raw >> 8) & 0x00FF

    sub = (lower >> 7) & 0x1
    encoded_upper = ((upper & 0x80) | (((upper & 0x3F) << 1) + sub)) & 0xFF

    upper_fp8 = encoded_upper.to(torch.uint8).view(torch.float8_e4m3fn)
    lower_fp8 = lower.to(torch.uint8).view(torch.float8_e4m3fn)

    if column_major:
        return _make_column_major(upper_fp8), _make_column_major(lower_fp8)
    return upper_fp8, lower_fp8


def reconstruct_fp16_from_dual_fp8(
    upper: torch.Tensor, lower: torch.Tensor
) -> torch.Tensor:
    upper_u8 = upper.view(torch.uint8)
    lower_u8 = lower.view(torch.uint8)

    sign = upper_u8 & 0x80
    sub = (lower_u8 & 0x80) >> 7
    packed_upper = ((upper_u8 - sub) >> 1) & 0x3F
    packed_upper |= sign

    raw_i32 = (packed_upper.to(torch.int32) << 8) | lower_u8.to(torch.int32)
    return raw_i32.to(torch.uint16).view(torch.float16)


def pack_fp16_to_dual_fp8_e5m2(
    fp16_tensor: torch.Tensor,
    *,
    column_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = fp16_tensor.view(torch.int16).to(torch.int32)
    upper = (raw >> 8) & 0xFF
    lower = raw & 0xFF
    exp = (raw >> 10) & 0x1F
    finite_normal = ((exp != 0) & (exp != 31)).to(torch.int32)
    inc = (
        ((lower > 0x80) | ((lower == 0x80) & ((upper & 1) == 1))) & finite_normal.bool()
    ).to(torch.int32)
    stored_upper = ((upper + inc) & 0xFF).to(torch.uint8).view(torch.float8_e5m2)
    stored_lower = torch.where(
        finite_normal.bool(), (lower | inc).to(torch.int32), lower
    ).to(torch.uint8).view(torch.float8_e5m2)

    if column_major:
        return _make_column_major(stored_upper), _make_column_major(stored_lower)
    return stored_upper, stored_lower


def reconstruct_fp16_from_dual_fp8_e5m2(
    upper: torch.Tensor, lower: torch.Tensor
) -> torch.Tensor:
    upper_u8 = upper.view(torch.uint8).to(torch.int32)
    lower_u8 = lower.view(torch.uint8).to(torch.int32)
    exp_bits = upper_u8 & 0x7C
    normal = (exp_bits != 0).to(torch.int32)
    inc = lower_u8 & normal
    upper_orig = (upper_u8 - inc) & 0xFF
    lower_orig = lower_u8 & ~normal
    raw = (upper_orig << 8) | lower_orig
    return raw.to(torch.int16).view(torch.float16)


def pack_fp16_to_dual_fp8_e5m2_trunc(
    fp16_tensor: torch.Tensor,
    *,
    column_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = fp16_tensor.view(torch.int16)
    upper = ((raw >> 8) & 0x00FF).to(torch.uint8).view(torch.float8_e5m2)
    lower = (raw & 0x00FF).to(torch.uint8).view(torch.float8_e5m2)

    if column_major:
        return _make_column_major(upper), _make_column_major(lower)
    return upper, lower


def reconstruct_fp16_from_dual_fp8_e5m2_trunc(
    upper: torch.Tensor, lower: torch.Tensor
) -> torch.Tensor:
    upper_u8 = upper.view(torch.uint8).to(torch.int32)
    lower_u8 = lower.view(torch.uint8).to(torch.int32)
    raw = (upper_u8 << 8) | lower_u8
    return raw.to(torch.int16).view(torch.float16)


# --- tests ---


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
def test_dual_weight_mm(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("dual_weight_mm requires SM80 (A100).")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16)

    w_upper, w_lower = pack_fp16_to_dual_fp8(weight_fp16, column_major=True)
    prepared_upper, prepared_lower = prepare_dual_weight_mm_weights(w_upper, w_lower)

    reconstructed_weight = reconstruct_fp16_from_dual_fp8(*pack_fp16_to_dual_fp8(weight_fp16))
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm(a, prepared_upper, prepared_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
def test_dual_weight_mm_e5m2(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("dual_weight_mm_e5m2 requires SM80 (A100).")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weight_fp16 = _make_finite_bf16_cast_fp16_weights(n * k, device="cuda").reshape(n, k)

    w_upper, w_lower = pack_fp16_to_dual_fp8_e5m2(weight_fp16, column_major=True)
    prepared_upper, prepared_lower = prepare_dual_weight_mm_weights_e5m2(w_upper, w_lower)

    w_upper_row, w_lower_row = pack_fp16_to_dual_fp8_e5m2(weight_fp16)
    reconstructed_weight = reconstruct_fp16_from_dual_fp8_e5m2(w_upper_row, w_lower_row)
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm_e5m2(a, prepared_upper, prepared_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
def test_dual_weight_mm_e5m2_trunc(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("dual_weight_mm_e5m2_trunc requires SM80 (A100).")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.1

    w_upper, w_lower = pack_fp16_to_dual_fp8_e5m2_trunc(weight_fp16, column_major=True)
    prepared_upper, prepared_lower = prepare_dual_weight_mm_weights_e5m2(
        w_upper, w_lower
    )

    w_upper_row, w_lower_row = pack_fp16_to_dual_fp8_e5m2_trunc(weight_fp16)
    reconstructed_weight = reconstruct_fp16_from_dual_fp8_e5m2_trunc(
        w_upper_row, w_lower_row
    )
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm_e5m2_trunc(a, prepared_upper, prepared_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


# --- SM90 tests ---


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
def test_dual_weight_mm_sm90(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("dual_weight_mm_sm90 requires SM90+.")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weight_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16)

    w_upper, w_lower = pack_fp16_to_dual_fp8(weight_fp16, column_major=True)

    reconstructed_weight = reconstruct_fp16_from_dual_fp8(*pack_fp16_to_dual_fp8(weight_fp16))
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm_sm90(a, w_upper, w_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.6f}"


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
def test_dual_weight_mm_sm90_e5m2(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("dual_weight_mm_sm90_e5m2 requires SM90+.")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    weight_fp16 = _make_finite_bf16_cast_fp16_weights(n * k, device="cuda").reshape(n, k)

    w_upper, w_lower = pack_fp16_to_dual_fp8_e5m2(weight_fp16, column_major=True)

    w_upper_row, w_lower_row = pack_fp16_to_dual_fp8_e5m2(weight_fp16)
    reconstructed_weight = reconstruct_fp16_from_dual_fp8_e5m2(w_upper_row, w_lower_row)
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm_sm90_e5m2(a, w_upper, w_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.6f}"


@pytest.mark.parametrize("m, n, k", [(32, 128, 128), (63, 256, 256)])
@pytest.mark.skip(reason="SM90 standalone E5M2 trunc kernel not yet implemented — uses E4M3 reconstruction")
def test_dual_weight_mm_sm90_e5m2_trunc(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability[0] < 9:
        pytest.skip("dual_weight_mm_sm90_e5m2_trunc requires SM90+.")

    torch.manual_seed(42)
    a = torch.randn((m, k), device="cuda", dtype=torch.float16) * 0.1
    weight_fp16 = torch.randn((n, k), device="cuda", dtype=torch.float16) * 0.1

    w_upper, w_lower = pack_fp16_to_dual_fp8_e5m2_trunc(weight_fp16, column_major=True)

    w_upper_row, w_lower_row = pack_fp16_to_dual_fp8_e5m2_trunc(weight_fp16)
    reconstructed_weight = reconstruct_fp16_from_dual_fp8_e5m2_trunc(w_upper_row, w_lower_row)
    reference = F.linear(a.float(), reconstructed_weight.float()).to(torch.float16)

    with autotune():
        out = dual_weight_mm_sm90_e5m2_trunc(a, w_upper, w_lower)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99, f"cos_sim={cos_sim:.6f}"


if __name__ == "__main__":
    pytest.main([__file__])
