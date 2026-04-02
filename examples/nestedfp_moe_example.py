"""
Example: NestedFP with FlashInfer Fused MOE APIs

Demonstrates two NestedFP integration paths:
  1. FP8 Scale Mode  — FP8 weights + fixed 1/256 dequant scale via cutlass_fused_moe
  2. Dual Weight Mode — lossless FP16-in-FP8 encoding via cutlass_dual_weight_fused_moe
"""

import torch
from torch.nn import functional as F

import flashinfer.fused_moe as fused_moe
from flashinfer.fused_moe.core import ActivationType

# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def compute_routing(router_logits: torch.Tensor, top_k: int):
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    return routing_weights.float(), selected_experts


def pack_fp16_to_dual_fp8(fp16_tensor: torch.Tensor):
    """Pack FP16 weights into NestedFP dual FP8 encoding (upper + lower bytes)."""
    raw = fp16_tensor.view(torch.int16)
    lo = raw & 0x00FF
    hi = (raw >> 8) & 0x00FF
    sub = (lo >> 7) & 0x1
    upper = ((hi & 0x80) | (((hi & 0x3F) << 1) + sub)) & 0xFF
    return (
        upper.to(torch.uint8).view(torch.float8_e4m3fn),
        lo.to(torch.uint8).view(torch.float8_e4m3fn),
    )


def quantize_fp16_to_nestedfp_fp8(x: torch.Tensor):
    """Quantize FP16 → FP8 using the NestedFP convention (clamp to FP8 range, cast)."""
    FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
    return x.clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)


# ─────────────────────────────────────────────────────────────────────
# Model config
# ─────────────────────────────────────────────────────────────────────

num_tokens = 64
hidden_size = 256
intermediate_size = 512
num_experts = 4
top_k = 2

torch.manual_seed(42)
device = "cuda"

# Shared inputs
x_fp16 = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device=device) / 5
router_logits = torch.randn(num_tokens, num_experts, dtype=torch.float32, device=device)
routing_weights, selected_experts = compute_routing(router_logits, top_k)

# FP16 reference weights (gated SwiGLU → fc1 is [E, 2*N, K])
fc1_fp16 = torch.randn(num_experts, 2 * intermediate_size, hidden_size,
                        dtype=torch.float16, device=device) / 5
fc2_fp16 = torch.randn(num_experts, hidden_size, intermediate_size,
                        dtype=torch.float16, device=device) / 5

print("=" * 70)
print("NestedFP with FlashInfer Fused MOE — Example")
print("=" * 70)
print(f"  tokens={num_tokens}  hidden={hidden_size}  inter={intermediate_size}")
print(f"  experts={num_experts}  top_k={top_k}")
print()


# ─────────────────────────────────────────────────────────────────────
# 1. FP16 Baseline — cutlass_fused_moe with FP16 weights
# ─────────────────────────────────────────────────────────────────────

print("1) FP16 Baseline")
out_fp16 = torch.empty(num_tokens, hidden_size, dtype=torch.float16, device=device)
fused_moe.cutlass_fused_moe(
    x_fp16,
    selected_experts.to(torch.int),
    routing_weights,
    fc1_fp16,
    fc2_fp16,
    torch.float16,
    quant_scales=None,
    output=out_fp16,
)
print(f"   output shape: {out_fp16.shape}  dtype: {out_fp16.dtype}")
print()


# ─────────────────────────────────────────────────────────────────────
# 2. NestedFP FP8 Scale Mode — FP8 weights with fixed 1/256 dequant
#
#    This uses the standard cutlass_fused_moe with FP8 weights.
#    NestedFP applies a fixed alpha = 1/256 in the epilogue.
#    We achieve this via per-expert quant_scales filled with 1/256.
#
#    The per-expert alpha is a single scalar multiply per output tile
#    in the epilogue — effectively zero overhead vs hardcoded alpha.
# ─────────────────────────────────────────────────────────────────────

print("2) NestedFP FP8 Scale Mode (cutlass_fused_moe + fixed 1/256 scale)")

# Quantize input and weights to FP8
x_fp8 = quantize_fp16_to_nestedfp_fp8(x_fp16)
fc1_fp8 = quantize_fp16_to_nestedfp_fp8(fc1_fp16)
fc2_fp8 = quantize_fp16_to_nestedfp_fp8(fc2_fp16)

# NestedFP fixed scale: alpha = 1/256
NESTEDFP_ALPHA = 1.0 / 256.0

# FP8 quant_scales format: [fc1_dequant, fc2_quant, fc2_dequant, fc1_input_dequant]
#   fc1_dequant:       per-expert (num_experts,) — applied as alpha to GEMM1 output
#   fc2_quant:         scalar or per-expert      — requantization scale for GEMM2 input
#   fc2_dequant:       per-expert (num_experts,) — applied as alpha to GEMM2 output
#   fc1_input_dequant: scalar                    — input activation dequant
quant_scales_fp8 = [
    torch.full((num_experts,), NESTEDFP_ALPHA, dtype=torch.float32, device=device),  # fc1_dequant
    torch.tensor(1.0, dtype=torch.float32, device=device),                           # fc2_quant
    torch.full((num_experts,), NESTEDFP_ALPHA, dtype=torch.float32, device=device),  # fc2_dequant
    torch.tensor(1.0, dtype=torch.float32, device=device),                           # fc1_input_dequant
]

out_fp8_scale = torch.empty(num_tokens, hidden_size, dtype=torch.float16, device=device)
fused_moe.cutlass_fused_moe(
    x_fp8,
    selected_experts.to(torch.int),
    routing_weights,
    fc1_fp8,
    fc2_fp8,
    torch.float16,
    quant_scales=quant_scales_fp8,
    output=out_fp8_scale,
)
print(f"   output shape: {out_fp8_scale.shape}  dtype: {out_fp8_scale.dtype}")
print(f"   quant_scales: [fc1_dequant=1/256, fc2_quant=1.0, fc2_dequant=1/256, input_dequant=1.0]")
print()


# ─────────────────────────────────────────────────────────────────────
# 3. NestedFP Dual Weight Mode — lossless FP16 via upper/lower FP8
#
#    Packs FP16 weights into two FP8 tensors (upper + lower byte).
#    The kernel reconstructs exact FP16 in-flight via transform2.
#    No precision loss, no scaling needed.
# ─────────────────────────────────────────────────────────────────────

print("3) NestedFP Dual Weight Mode (cutlass_dual_weight_fused_moe)")

# Pack FP16 weights → dual FP8
fc1_upper, fc1_lower = pack_fp16_to_dual_fp8(fc1_fp16)
fc2_upper, fc2_lower = pack_fp16_to_dual_fp8(fc2_fp16)

print(f"   fc1_fp16 shape: {fc1_fp16.shape}  →  upper: {fc1_upper.shape} + lower: {fc1_lower.shape}")
print(f"   Weight memory: {fc1_fp16.numel() * 2 / 1024:.1f} KB (FP16)"
      f"  →  {(fc1_upper.numel() + fc1_lower.numel()) / 1024:.1f} KB (dual FP8, same size, lossless)")

out_dual = torch.empty(num_tokens, hidden_size, dtype=torch.float16, device=device)
fused_moe.cutlass_dual_weight_fused_moe(
    x_fp16,
    selected_experts.to(torch.int32),
    routing_weights,
    fc1_upper,
    fc1_lower,
    fc2_upper,
    fc2_lower,
    output=out_dual,
)
print(f"   output shape: {out_dual.shape}  dtype: {out_dual.dtype}")

# Verify dual weight matches FP16 baseline (lossless reconstruction)
max_diff = (out_fp16 - out_dual).abs().max().item()
print(f"   max |FP16 - DualWeight| = {max_diff:.6f}  (should be small — lossless weights)")
print()


# ─────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Summary")
print("=" * 70)
print("""
  Method              | Weight dtype  | Weight memory | Precision | Scales needed
  --------------------|---------------|---------------|-----------|---------------
  FP16 Baseline       | float16       | 2 bytes/param | Full      | None
  FP8 Scale (1/256)   | float8_e4m3   | 1 byte/param  | Lossy     | Fixed 1/256
  Dual Weight         | 2×float8_e4m3 | 2 bytes/param | Lossless  | None

  FP8 Scale mode gives 2× memory savings with lossy quantization.
  Dual Weight mode preserves exact FP16 precision (same memory as FP16).
  Both use SM90 TMA + GMMA for near-FP16 performance.
""")
