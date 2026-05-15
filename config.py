"""
Configuration for MoE-BitNet English LLM training.
All hyperparameters are adjustable here. Compatible with P100 (16GB), T4x2 (32GB), RTX (24GB+).
"""

import os
import multiprocessing

class Config:
    # ── Paths ──────────────────────────────────────────────────────────
    checkpoint_dir: str = os.environ.get(
        "CHECKPOINT_DIR",
        "/kaggle/working/checkpoints" if os.path.exists("/kaggle/working") else "./checkpoints",
    )
    tokenizer_path: str = os.path.join(checkpoint_dir, "tokenizer.json")
    log_dir: str = os.path.join(checkpoint_dir, "logs")
    resume_from: str | None = None  # path to checkpoint dir to resume; auto-discovers latest

    # ── Model Architecture ─────────────────────────────────────────────
    # Defaults sized for P100 16GB ~255M params. Scale up for bigger GPUs.
    d_model: int = 1024
    n_layers: int = 16
    n_heads: int = 16
    n_kv_heads: int = 4  # GQA ratio: n_heads // n_kv_heads = 4
    d_head: int = d_model // n_heads  # 64
    ffn_dim: int = 2816  # ~2.75x d_model
    vocab_size: int = 32000
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    use_recommended_taste: bool = True  # BitNet b1.58 + BitLinear layers

    # ── BitNet Quantization ────────────────────────────────────────────
    weight_init_std: float = 0.006  # N(0, 0.006) — smaller than standard for BitNet
    activation_bits: int = 8
    clamp_activation: int = 127  # 2^(bits-1) - 1
    monitor_quant_error: bool = True
    quant_error_threshold: float = 0.15

    # ── Training ───────────────────────────────────────────────────────
    # Batch sizing — adjust based on VRAM
    micro_batch_size: int = 2  # per GPU
    gradient_accumulation_steps: int = 8
    seq_len: int = 2048
    effective_batch_size: int = micro_batch_size * gradient_accumulation_steps  # 16

    # Optimizer
    optimizer: str = "adamw"  # adamw, came, lion
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 500

    # Schedule
    lr_schedule: str = "cosine"  # cosine, linear
    total_steps: int = 30000  # ~12 hours on P100 at ~1.5s/step with eff_batch=16

    # Stochastic Weight Averaging
    use_swa: bool = True
    swa_start_pct: float = 0.80  # start SWA at 80% of training
    swa_lr: float = 1e-4
    swa_annealing_steps: int = 1000

    # Gradient checkpointing (trades compute for memory)
    use_gradient_checkpointing: bool = True

    # Mixed precision
    mixed_precision: str = "bf16"  # bf16, fp16, no

    # ── Dataset ────────────────────────────────────────────────────────
    datasets: list[str] = [
        "yahma/alpaca-cleaned",
        "OpenAssistant/oasst2",
    ]
    dataset_split: str = "train"
    val_split_pct: float = 0.02

    # ── Checkpointing / Resume ─────────────────────────────────────────
    save_every_steps: int = 2000
    eval_every_steps: int = 500
    keep_last_n_checkpoints: int = 3
    save_optimizer: bool = True  # full resume support

    # ── Logging ────────────────────────────────────────────────────────
    use_wandb: bool = False
    wandb_project: str = "bitnet-moe-llm"
    wandb_entity: str | None = None
    log_every_steps: int = 10

    # ── Inference ──────────────────────────────────────────────────────
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_new_tokens: int = 512
    repetition_penalty: float = 1.1

    # ── Hardware / Distributed ─────────────────────────────────────────
    num_workers: int = min(4, multiprocessing.cpu_count() // 2)
    pin_memory: bool = True
    ddp_backend: str = "nccl"  # nccl for GPU, gloo for CPU fallback

    @property
    def device(self) -> str:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    @property
    def swa_start_step(self) -> int:
        return int(self.total_steps * self.swa_start_pct)


# Pre-built model sizes for different GPU budgets
class SmallConfig(Config):
    """P100 16GB — ~150M params"""
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    n_kv_heads: int = 4
    d_head: int = d_model // n_heads
    ffn_dim: int = 2048
    micro_batch_size: int = 2
    gradient_accumulation_steps: int = 8


class MediumConfig(Config):
    """T4x2 or single RTX 24GB — ~255M params"""
    pass  # defaults


class LargeConfig(Config):
    """RTX with 48GB+ VRAM — ~500M params"""
    d_model: int = 1536
    n_layers: int = 20
    n_heads: int = 24
    n_kv_heads: int = 6
    d_head: int = d_model // n_heads
    ffn_dim: int = 4096
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 4096
    seq_len: int = 4096
