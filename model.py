"""
Full BitNet b1.58 Transformer Language Model.
Assembles embedding, transformer blocks, and LM head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_components import TransformerBlock, RMSNorm


class BitNetTransformer(nn.Module):
    """
    BitNet b1.58 autoregressive language model.
    
    Architecture:
      - Token embedding (weight-tied with LM head for efficiency)
      - N transformer blocks (BitLinear throughout)
      - Final RMSNorm
      - LM head (shares weights with embedding)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        n_kv_heads: int,
        ffn_dim: int,
        max_seq_len: int = 2048,
        rope_theta: float = 10000.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                ffn_dim=ffn_dim,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
            )
            for _ in range(n_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        # Weight tying: LM head shares embedding weights
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        
        self.max_seq_len = max_seq_len
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize non-BitLinear params. BitLinear layers self-initialize."""
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
    
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch, seq) token indices
            position_ids: (batch, seq) or None
            labels: (batch, seq) for loss computation (dataset provides shifted labels)
            
        Returns:
            dict with 'logits', 'loss' (if labels provided), 'ppl'
        """
        batch, seq = input_ids.shape
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)
        
        x = self.embedding(input_ids)  # (batch, seq, d_model)
        
        causal_mask = self._create_causal_mask(seq, device)
        
        for layer in self.layers:
            if hasattr(self, '_gradient_checkpointing') and self._gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, position_ids, causal_mask, use_reentrant=False
                )
            else:
                x = layer(x, position_ids, causal_mask)
        
        x = self.norm(x)
        logits = self.lm_head(x)  # (batch, seq, vocab_size)
        
        result = {"logits": logits}
        
        if labels is not None:
            # Dataset already shifts labels by 1 — use them directly
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss
            result["ppl"] = torch.exp(loss)
        
        return result
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing on all transformer blocks."""
        self._gradient_checkpointing = True
    
    def disable_gradient_checkpointing(self):
        self._gradient_checkpointing = False
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p and top-k sampling."""
        self.eval()
        batch, seq = input_ids.shape
        device = input_ids.device
        
        generated = input_ids.clone()
        
        for _ in range(max_new_tokens):
            # Truncate to max_seq_len if needed
            if generated.shape[1] > self.max_seq_len:
                generated = generated[:, -self.max_seq_len:]
            
            curr_seq = generated.shape[1]
            position_ids = torch.arange(curr_seq, device=device).unsqueeze(0).expand(batch, -1)
            
            x = self.embedding(generated)
            causal_mask = self._create_causal_mask(curr_seq, device)
            
            for layer in self.layers:
                x = layer(x, position_ids, causal_mask)
            
            x = self.norm(x)
            logits = self.lm_head(x[:, -1:, :])  # only last position
            logits = logits[:, -1, :] / temperature  # (batch, vocab)
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for b in range(batch):
                    unique_tokens = set(generated[b].tolist())
                    for token_id in unique_tokens:
                        if logits[b, token_id] < 0:
                            logits[b, token_id] *= repetition_penalty
                        else:
                            logits[b, token_id] /= repetition_penalty
            
            # Top-k filtering
            if top_k > 0:
                top_k_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_k_vals[:, -1:]] = float("-inf")
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs > top_p
                sorted_mask[:, 1:] = sorted_mask[:, :-1].clone()
                sorted_mask[:, 0] = False
                indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
                logits[indices_to_remove] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
            
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}
    
    def parameter_count_str(self) -> str:
        counts = self.count_parameters()
        total_m = counts["total"] / 1e6
        return f"{total_m:.1f}M params ({counts['trainable']/1e6:.1f}M trainable)"


def create_model(config) -> BitNetTransformer:
    """Factory: build BitNetTransformer from a config object."""
    return BitNetTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )
