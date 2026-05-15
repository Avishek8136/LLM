"""
Transformer building blocks: RMSNorm, RoPE, Grouped Query Attention, SwiGLU FFN.
All linear layers use BitLinear for native BitNet b1.58 training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from bitlinear import BitLinear


# ── RMS Normalization ───────────────────────────────────────────────────
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization — no bias, no mean subtraction."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        x = (x / rms) * self.weight.float()
        return x.to(dtype)


# ── Rotary Position Embedding (RoPE) ────────────────────────────────────
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding with theta=10000 — standard LLaMA-style."""
    
    def __init__(self, dim: int, theta: float = 10000.0, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        self._cache: dict[str, torch.Tensor] = {}
    
    @torch.no_grad()
    def _compute_freqs(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute cos/sin tables. Cached per-device."""
        key = str(device)
        if key in self._cache:
            return self._cache[key]
        
        freqs = 1.0 / (
            self.theta ** (torch.arange(0, self.dim, 2, device=device).float() / self.dim)
        )
        t = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(t, freqs)  # (max_seq_len, dim/2)
        
        cos = angles.cos()  # (max_seq_len, dim/2)
        sin = angles.sin()
        self._cache[key] = (cos, sin)
        return cos, sin
    
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Apply rotary embedding to x.
        
        Args:
            x: (batch, n_heads, seq, d_head)
            position_ids: (batch, seq) or None for causal positions
            
        Returns:
            rotated tensor of same shape
        """
        batch, n_heads, seq_len, d_head = x.shape
        cos, sin = self._compute_freqs(x.device)
        
        if position_ids is not None:
            cos = cos[position_ids]  # (batch, seq, dim/2)
            sin = sin[position_ids]
        else:
            cos = cos[:seq_len].unsqueeze(0)  # (1, seq, dim/2)
            sin = sin[:seq_len].unsqueeze(0)
        
        cos = cos.unsqueeze(1)  # (batch, 1, seq, dim/2)
        sin = sin.unsqueeze(1)
        
        # Rotate half the dimensions
        x_rot = x.float()
        x1 = x_rot[..., : d_head // 2]
        x2 = x_rot[..., d_head // 2:]
        
        # Complement: combine first and second halves with cos/sin
        rotated1 = x1 * cos - x2 * sin
        rotated2 = x2 * cos + x1 * sin
        
        return torch.cat([rotated1, rotated2], dim=-1).to(x.dtype)
    
    def _clear_cache(self):
        self._cache.clear()


# ── Grouped Query Attention ─────────────────────────────────────────────
class GroupedQueryAttention(nn.Module):
    """
    Multi-head attention with Grouped Query Attention (GQA).
    Query heads > KV heads. KV heads shared across query groups.
    d_head is computed as d_model // n_heads.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads
        
        # Q projection: all query heads
        self.wq = BitLinear(d_model, n_heads * self.d_head)
        
        # K, V projections: only KV heads (shared within groups)
        self.wk = BitLinear(d_model, n_kv_heads * self.d_head)
        self.wv = BitLinear(d_model, n_kv_heads * self.d_head)
        
        # Output projection
        self.wo = BitLinear(n_heads * self.d_head, d_model)
        
        # RoPE applied to query and key
        self.rotary = RotaryEmbedding(self.d_head, theta=rope_theta, max_seq_len=max_seq_len)
        
        self.max_seq_len = max_seq_len
    
    def _reshape_for_attention(self, x: torch.Tensor, n_heads: int) -> torch.Tensor:
        """Reshape from (batch, seq, n_heads*d_head) to (batch, n_heads, seq, d_head)."""
        batch, seq, _ = x.shape
        return x.view(batch, seq, n_heads, self.d_head).transpose(1, 2)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq, d_model)
            position_ids: (batch, seq)
            causal_mask: (seq, seq) or None (causal attention by default)
        """
        batch, seq, _ = x.shape
        
        # Project
        q = self.wq(x)  # (batch, seq, n_heads * d_head)
        k = self.wk(x)  # (batch, seq, n_kv_heads * d_head)
        v = self.wv(x)  # (batch, seq, n_kv_heads * d_head)
        
        # Reshape: (batch, n_heads, seq, d_head)
        q = self._reshape_for_attention(q, self.n_heads)
        k = self._reshape_for_attention(k, self.n_kv_heads)
        v = self._reshape_for_attention(v, self.n_kv_heads)
        
        # Apply RoPE
        q = self.rotary(q, position_ids)
        k = self.rotary(k, position_ids)
        
        # Expand KV heads to match Q heads for GQA
        # (batch, n_kv, seq, d_head) -> (batch, n_groups, n_kv, seq, d_head) -> (batch, n_heads, seq, d_head)
        k = k.unsqueeze(1).expand(batch, self.n_groups, self.n_kv_heads, seq, self.d_head)
        k = k.reshape(batch, self.n_heads, seq, self.d_head)
        v = v.unsqueeze(1).expand(batch, self.n_groups, self.n_kv_heads, seq, self.d_head)
        v = v.reshape(batch, self.n_heads, seq, self.d_head)
        
        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_head)
        
        if causal_mask is None:
            # Causal mask for autoregressive
            causal_mask = torch.triu(
                torch.ones(seq, seq, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=causal_mask,
            dropout_p=0.0,
            is_causal=(causal_mask is None),
            scale=scale,
        )
        
        # Merge heads: (batch, n_heads, seq, d_head) -> (batch, seq, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq, -1)
        
        # Output projection
        return self.wo(attn_output)


# ── FFN with Squared ReLU activation ────────────────────────────────────
class ReLUFFN(nn.Module):
    """
    Feed-forward network with ReLU-gated activation (GLU variant).
    Architecture: gate + up → ReLU(gate) * up → down
    Uses standard ReLU (not squared) to avoid activation amplification.
    """
    
    def __init__(self, d_model: int, ffn_dim: int):
        super().__init__()
        self.gate_proj = BitLinear(d_model, ffn_dim)
        self.up_proj = BitLinear(d_model, ffn_dim)
        self.down_proj = BitLinear(ffn_dim, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


# ── Transformer Block (BitNet-style with SubLN) ─────────────────────────
class TransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm (SubLN per BitNet spec).
    Uses:
      - RMSNorm (no bias)
      - GQA attention with RoPE
      - ReLU-gated FFN
      - All linear layers are BitLinear
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.attention_norm = RMSNorm(d_model)
        self.attention = GroupedQueryAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            max_seq_len=max_seq_len,
            rope_theta=rope_theta,
        )
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = ReLUFFN(d_model=d_model, ffn_dim=ffn_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        causal_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SubLN: pre-norm for attention
        x = x + self.attention(self.attention_norm(x), position_ids, causal_mask)
        # SubLN: pre-norm for FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x


# ── Monitoring helpers ──────────────────────────────────────────────────
def get_quantization_errors(blocks: nn.ModuleList) -> list[float]:
    """Collect per-layer quantization errors from all BitLinear layers in blocks."""
    errors = []
    for block in blocks:
        for module in block.modules():
            if isinstance(module, BitLinear):
                errors.append(module.quantization_error())
    return errors


def average_quantization_error(blocks: nn.ModuleList) -> float:
    errors = get_quantization_errors(blocks)
    if not errors:
        return 0.0
    return sum(errors) / len(errors)
