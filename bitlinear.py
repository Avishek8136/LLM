"""
BitLinear: BitNet b1.58 ternary quantization layer.
Weights quantized to {-1, 0, +1}. Activations to int8. STE for backprop.

Reference: Ma et al. (2024) — BitNet b1.58: 1-bit LLMs
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Straight-Through Estimator for rounding ──────────────────────────────
class _STEQuantizeFn(torch.autograd.Function):
    """STE: forward applies round+clamp. backward passes gradient through as identity."""

    @staticmethod
    def forward(ctx, x, scale, clamp_min, clamp_max):
        ctx.save_for_backward(x, scale)
        ctx.clamp_min = clamp_min
        ctx.clamp_max = clamp_max
        x_q = (x / scale).round()
        x_q = x_q.clamp(clamp_min, clamp_max)
        return x_q

    @staticmethod
    def backward(ctx, grad_output):
        # STE: gradient flows through quantization unchanged (scaled by 1/scale)
        _, scale = ctx.saved_tensors
        grad_input = grad_output / scale
        return grad_input, None, None, None


def ste_quantize(x, scale, clamp_min, clamp_max):
    """Quantize with straight-through estimator for gradients.
    
    Forward: x_q = clamp(round(x / scale), min, max)
    Backward: grad flows through as grad / scale
    """
    return _STEQuantizeFn.apply(x, scale, clamp_min, clamp_max)


class BitLinear(nn.Module):
    """
    BitNet b1.58 linear layer.
    
    Forward pass:
      1. Weights quantized to ternary {-1, 0, +1} with scale alpha = mean(|W|)
      2. Activations quantized to int8 per-token with scale beta = max(|X|)/127
      3. Output = alpha * beta ⊙ (X_q @ W_q^T)
      4. Gradients flow through via STE
      
    No bias — disabled per BitNet spec.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Small-weight initialization per BitNet spec: N(0, 0.006)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.weight, mean=0.0, std=0.006)
    
    def _quantize_weight(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize weights to ternary {-1, 0, +1}.
        
        Returns:
            w_q: ternary weights, shape (out, in)
            alpha: scalar scale factor
        """
        alpha = w.abs().mean()
        alpha = alpha.clamp(min=1e-8)
        w_q = ste_quantize(w, alpha, -1.0, 1.0)
        return w_q, alpha
    
    def _quantize_activation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize activations to int8 per-token.
        
        Args:
            x: (batch, seq, d_in)
            
        Returns:
            x_q: int8 activations clamped to [-127, 127]
            beta: per-token scale, shape (batch, seq, 1)
        """
        # Per-token absmax
        x_absmax = x.abs().max(dim=-1, keepdim=True)[0]  # (batch, seq, 1)
        beta = x_absmax / 127.0
        beta = beta.clamp(min=1e-8)
        x_q = ste_quantize(x, beta, -127.0, 127.0)
        return x_q, beta
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with full BitNet quantization.
        
        Args:
            x: (batch, seq, in_features)
            
        Returns:
            y: (batch, seq, out_features)
        """
        w_q, alpha = self._quantize_weight(self.weight)
        x_q, beta = self._quantize_activation(x)
        
        # Quantized matmul can overflow FP16 (1024 * 127 = 130048 > 65504).
        # Force FP32 for this operation regardless of surrounding autocast.
        with torch.autocast(device_type="cuda", enabled=False):
            y = F.linear(x_q.float(), w_q.float())
        y = y * alpha * beta    # broadcast: (batch,seq,out) * scalar * (batch,seq,1)
        return y
    
    @torch.no_grad()
    def quantization_error(self) -> float:
        """Return ||W - W_q|| / ||W|| for monitoring stability."""
        w = self.weight
        alpha = w.abs().mean().clamp(min=1e-8)
        w_q = (w / alpha).round().clamp(-1.0, 1.0) * alpha
        error = (w - w_q).norm() / w.norm().clamp(min=1e-8)
        return error.item()
    
    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False (BitNet)"


class BitLinearInferenceOnly(BitLinear):
    """
    Inference-mode BitLinear that caches the ternary weight matrix.
    For deployment where weights are frozen.
    """
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self._cached_w_q: torch.Tensor | None = None
        self._cached_alpha: torch.Tensor | None = None
    
    def cache_weights(self):
        """Pre-compute ternary weights for fast inference."""
        self._cached_w_q, self._cached_alpha = self._quantize_weight(self.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._cached_w_q is None:
            self.cache_weights()
        w_q = self._cached_w_q
        alpha = self._cached_alpha
        x_q, beta = self._quantize_activation(x)
        with torch.autocast(device_type="cuda", enabled=False):
            y = F.linear(x_q.float(), w_q.float())
        y = y * alpha * beta
        return y
