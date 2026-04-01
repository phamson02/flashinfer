import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mixed_mm_e4m3, mixed_mm_e5m2, prepare_mixed_mm_weights
from flashinfer.utils import get_compute_capability


def pack_fp16_to_nested_fp_e4m3(fp16: torch.Tensor) -> torch.Tensor:
    raw = fp16.view(torch.int16).to(torch.int32)
    sign = (raw >> 8) & 0x80
    em = (raw >> 7) & 0x7F
    return ((sign | em) & 0xFF).to(torch.uint8).view(torch.float8_e4m3fn)


def reconstruct_fp16_from_nested_e4m3(e4m3: torch.Tensor) -> torch.Tensor:
    b = e4m3.view(torch.uint8).to(torch.int32)
    return (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(torch.int16).view(torch.float16)


def pack_fp16_to_e5m2_trunc(fp16: torch.Tensor) -> torch.Tensor:
    raw = fp16.view(torch.int16)
    return ((raw >> 8) & 0x00FF).to(torch.uint8).view(torch.float8_e5m2)


def reconstruct_fp16_from_e5m2_trunc(e5m2: torch.Tensor) -> torch.Tensor:
    b = e5m2.view(torch.uint8).to(torch.int32)
    return (b << 8).to(torch.int16).view(torch.float16)


@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n,k", [(4096, 4096), (2048, 1024)])
def test_mixed_mm_e4m3(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("mixed_mm_e4m3 requires SM80 (A100).")

    torch.manual_seed(0)
    weight_fp16 = (torch.randn(n, k, dtype=torch.float16) * 0.1).cuda()
    a = torch.randn(m, k, dtype=torch.float16).cuda() * 0.5

    b_e4m3 = pack_fp16_to_nested_fp_e4m3(weight_fp16.cpu()).cuda()
    weight_approx = reconstruct_fp16_from_nested_e4m3(b_e4m3.cpu()).cuda()
    reference = torch.mm(a, weight_approx.T)

    b_prepared = prepare_mixed_mm_weights(b_e4m3)
    with autotune():
        out = mixed_mm_e4m3(a, b_prepared)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("m", [1, 16, 64, 128])
@pytest.mark.parametrize("n,k", [(4096, 4096), (2048, 1024)])
def test_mixed_mm_e5m2(m: int, n: int, k: int) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("mixed_mm_e5m2 requires SM80 (A100).")

    torch.manual_seed(1)
    weight_fp16 = (torch.randn(n, k, dtype=torch.float16) * 0.1).cuda()
    a = torch.randn(m, k, dtype=torch.float16).cuda() * 0.5

    b_e5m2 = pack_fp16_to_e5m2_trunc(weight_fp16.cpu()).cuda()
    weight_approx = reconstruct_fp16_from_e5m2_trunc(b_e5m2.cpu()).cuda()
    reference = torch.mm(a, weight_approx.T)

    b_prepared = prepare_mixed_mm_weights(b_e5m2)
    with autotune():
        out = mixed_mm_e5m2(a, b_prepared)

    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


@pytest.mark.parametrize("dtype", ["e4m3", "e5m2"])
def test_mixed_mm_with_scale(dtype: str) -> None:
    compute_capability = get_compute_capability(torch.device("cuda"))
    if compute_capability != (8, 0):
        pytest.skip("mixed_mm requires SM80 (A100).")

    torch.manual_seed(2)
    m, n, k = 32, 256, 256
    a = torch.randn(m, k, dtype=torch.float16).cuda() * 0.1

    if dtype == "e4m3":
        weight_fp16 = (torch.randn(n, k, dtype=torch.float16) * 0.1).cuda()
        b = pack_fp16_to_nested_fp_e4m3(weight_fp16.cpu()).cuda()
        weight_approx = reconstruct_fp16_from_nested_e4m3(b.cpu()).cuda()
        mm_fn = mixed_mm_e4m3
    else:
        weight_fp16 = (torch.randn(n, k, dtype=torch.float16) * 0.1).cuda()
        b = pack_fp16_to_e5m2_trunc(weight_fp16.cpu()).cuda()
        weight_approx = reconstruct_fp16_from_e5m2_trunc(b.cpu()).cuda()
        mm_fn = mixed_mm_e5m2

    b_prepared = prepare_mixed_mm_weights(b)
    scale = torch.rand(n, dtype=torch.float16).cuda() * 2.0

    with autotune():
        out = mm_fn(a, b_prepared, scale=scale)

    reference = torch.mm(a, weight_approx.T) * scale.unsqueeze(0)
    cos_sim = F.cosine_similarity(reference.reshape(-1), out.reshape(-1), dim=0)
    assert cos_sim > 0.99


if __name__ == "__main__":
    pytest.main([__file__])
