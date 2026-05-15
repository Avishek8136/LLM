"""
Training pipeline for BitNet b1.58 LLM.
Supports: resume from checkpoint, gradient checkpointing, SWA, DDP multi-GPU,
cosine LR schedule, mixed precision, and periodic evaluation.
"""

import os
import sys
import json
import time
import math
import random
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

from config import Config
from config import check_gpu_compatibility
from model import BitNetTransformer, create_model
from model_components import average_quantization_error


# ── Seed everything for reproducibility ─────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ── Optimizer factory ───────────────────────────────────────────────────
def create_optimizer(model: nn.Module, config: Config) -> torch.optim.Optimizer:
    """Create optimizer with decay/no-decay param groups (BitNet spec)."""
    # Separate params that should get weight decay
    decay_params = []
    nodecay_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # No weight decay for norms, biases, embeddings
        if any(x in name for x in ["norm", "bias", "embedding"]):
            nodecay_params.append(param)
        else:
            decay_params.append(param)
    
    param_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    
    if config.optimizer == "adamw":
        return torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
    elif config.optimizer == "lion":
        try:
            from lion_pytorch import Lion
            return Lion(param_groups, lr=config.learning_rate, betas=(config.beta1, config.beta2))
        except ImportError:
            print("Warning: lion_pytorch not installed, falling back to AdamW")
            return torch.optim.AdamW(
                param_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2),
                eps=config.eps,
            )
    else:
        return torch.optim.AdamW(
            param_groups,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )


# ── LR Scheduler ────────────────────────────────────────────────────────
def create_scheduler(optimizer, config: Config):
    """Create cosine LR scheduler with linear warmup."""
    
    def lr_lambda(current_step: int) -> float:
        if current_step < config.warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, config.warmup_steps))
        elif current_step > config.total_steps:
            return config.min_lr / config.learning_rate
        else:
            # Cosine decay
            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.total_steps - config.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            decayed = (1.0 - config.min_lr / config.learning_rate) * cosine_decay
            decayed = decayed + config.min_lr / config.learning_rate
            return max(decayed, config.min_lr / config.learning_rate)
    
    return LambdaLR(optimizer, lr_lambda)


# ── DDP setup / teardown ────────────────────────────────────────────────
def setup_ddp():
    """Initialize distributed training if launched with torchrun."""
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        return True, local_rank
    return False, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


# ── Checkpoint save/load ────────────────────────────────────────────────
def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Config,
    step: int,
    epoch: int,
    metrics: dict,
    is_best: bool = False,
):
    """Save training checkpoint for pause/resume."""
    if not is_main_process():
        return
    
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # Unwrap DDP model if needed
    model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    
    checkpoint = {
        "model_state_dict": model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "step": step,
        "epoch": epoch,
        "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")},
        "metrics": metrics,
        "timestamp": datetime.now().isoformat(),
        "rng_states": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    
    base_name = f"checkpoint_step{step:06d}"
    path = os.path.join(config.checkpoint_dir, f"{base_name}.pt")
    torch.save(checkpoint, path)
    
    # Save latest symlink
    latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path) or os.path.islink(latest_path):
        os.remove(latest_path)
    os.symlink(f"{base_name}.pt", latest_path)
    
    # Cleanup old checkpoints
    _cleanup_old_checkpoints(config)
    
    print(f"  Checkpoint saved: {base_name}.pt (step {step})")
    
    if is_best:
        best_path = os.path.join(config.checkpoint_dir, "best.pt")
        shutil.copy(path, best_path)


def _cleanup_old_checkpoints(config: Config):
    """Keep only last N checkpoints to save disk space."""
    ckpt_files = sorted(
        [f for f in os.listdir(config.checkpoint_dir) if f.startswith("checkpoint_step")],
        key=lambda x: os.path.getmtime(os.path.join(config.checkpoint_dir, x)),
    )
    while len(ckpt_files) > config.keep_last_n_checkpoints:
        old = ckpt_files.pop(0)
        os.remove(os.path.join(config.checkpoint_dir, old))


def find_latest_checkpoint(config: Config) -> str | None:
    """Find the most recent checkpoint to resume from."""
    if config.resume_from and os.path.exists(config.resume_from):
        return config.resume_from
    
    latest_path = os.path.join(config.checkpoint_dir, "latest.pt")
    if os.path.exists(latest_path):
        return latest_path
    
    # Search for any checkpoint
    ckpt_files = sorted(
        [f for f in os.listdir(config.checkpoint_dir) if f.startswith("checkpoint_step") and f.endswith(".pt")],
        reverse=True,
    )
    if ckpt_files:
        return os.path.join(config.checkpoint_dir, ckpt_files[0])
    
    return None


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
) -> dict:
    """Load checkpoint and restore training state.
    
    Returns:
        dict with 'step', 'epoch', 'metrics'
    """
    print(f"Loading checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    if scheduler and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    
    # Restore RNG states
    if "rng_states" in checkpoint:
        rs = checkpoint["rng_states"]
        random.setstate(rs["python"])
        np.random.set_state(rs["numpy"])
        torch.set_rng_state(rs["torch"])
        if torch.cuda.is_available() and rs["torch_cuda"]:
            torch.cuda.set_rng_state_all(rs["torch_cuda"])
    
    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


# ── Training loop ───────────────────────────────────────────────────────
def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    config: Config,
    epoch: int,
    global_step: int,
    scaler,
    log_file: str | None = None,
) -> int:
    """Train one epoch. Returns updated global_step."""
    model.train()
    total_loss = 0.0
    start_time = time.time()
    optimizer.zero_grad()
    
    is_ddp = hasattr(model, "module")
    device = next(model.parameters()).device
    
    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward with mixed precision
        if scaler:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs["loss"]
                # Scale loss for gradient accumulation
                loss = loss / config.gradient_accumulation_steps
        else:
            outputs = model(input_ids, labels=labels)
            loss = outputs["loss"] / config.gradient_accumulation_steps
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        total_loss += loss.item() * config.gradient_accumulation_steps
        
        # Gradient accumulation step
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters() if not is_ddp else model.module.parameters(),
                config.grad_clip,
            )
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            
            # Logging
            if is_main_process() and global_step % config.log_every_steps == 0:
                elapsed = time.time() - start_time
                steps_done = global_step - (epoch * len(train_loader) // config.gradient_accumulation_steps)
                steps_done = max(1, steps_done)
                avg_loss = total_loss / steps_done
                lr = scheduler.get_last_lr()[0]
                
                ppl = math.exp(min(avg_loss, 20))
                
                log_msg = (
                    f"Epoch {epoch} | Step {global_step}/{config.total_steps} | "
                    f"Loss: {avg_loss:.4f} | PPL: {ppl:.1f} | LR: {lr:.2e} | "
                    f"Time: {elapsed:.0f}s"
                )
                
                if config.monitor_quant_error and hasattr(model.module if is_ddp else model, "layers"):
                    actual_model = model.module if is_ddp else model
                    q_err = average_quantization_error(actual_model.layers)
                    log_msg += f" | QErr: {q_err:.4f}"
                
                print(log_msg)
                
                if log_file:
                    with open(log_file, "a") as f:
                        f.write(log_msg + "\n")
            
            # Evaluation
            if is_main_process() and config.eval_every_steps > 0 and global_step % config.eval_every_steps == 0:
                pass  # eval called externally
            
            # Checkpoint
            if is_main_process() and global_step % config.save_every_steps == 0:
                save_checkpoint(
                    model, optimizer, scheduler, config,
                    step=global_step, epoch=epoch, metrics={"loss": total_loss / max(1, batch_idx + 1)},
                )
            
            # SWA check
            if config.use_swa and global_step >= config.swa_start_step and not _swa_started.get("done", False):
                if is_main_process():
                    print(f"\n  Starting SWA at step {global_step}")
                _swa_started["done"] = True
                _swa_started["model_state"] = {
                    k: v.clone() for k, v in
                    (model.module.state_dict() if is_ddp else model.state_dict()).items()
                }
                _swa_started["n_averaged"] = 0
        
        if global_step >= config.total_steps:
            break
    
    return global_step, total_loss / max(1, len(train_loader))


_swa_started: dict = {"done": False, "model_state": None, "n_averaged": 0}


def update_swa(model: nn.Module, config: Config):
    """Update stochastic weight average."""
    if not _swa_started["done"]:
        return
    
    is_ddp = hasattr(model, "module")
    current_state = model.module.state_dict() if is_ddp else model.state_dict()
    n = _swa_started["n_averaged"]
    n += 1
    _swa_started["n_averaged"] = n
    
    for key in _swa_started["model_state"]:
        _swa_started["model_state"][key] = (
            _swa_started["model_state"][key] * (n - 1) / n + current_state[key] / n
        )


def finalize_swa(model: nn.Module):
    """Apply SWA weights to model."""
    if not _swa_started["done"] or _swa_started["model_state"] is None:
        return
    
    is_ddp = hasattr(model, "module")
    if is_ddp:
        model.module.load_state_dict(_swa_started["model_state"])
    else:
        model.load_state_dict(_swa_started["model_state"])
    print("SWA weights applied to model.")


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        outputs = model(input_ids, labels=labels)
        loss = outputs["loss"]
        
        batch_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * batch_tokens
        total_tokens += batch_tokens
    
    model.train()
    
    avg_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(min(avg_loss, 20))
    
    return {"val_loss": avg_loss, "val_ppl": ppl}


# ── Main entry point ────────────────────────────────────────────────────
def main(config: Config | None = None):
    if config is None:
        config = Config()
    
    set_seed(42)
    
    # DDP
    use_ddp, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    if is_main_process():
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        print(f"Device: {device}")
        print(f"Checkpoint dir: {config.checkpoint_dir}")
        
        gpu_status = check_gpu_compatibility()
        if gpu_status == "error":
            print("GPU incompatible with installed PyTorch. See message above for fix.")
            if not use_ddp:
                sys.exit(1)
        elif gpu_status == "no_cuda":
            print("Warning: No CUDA GPU detected. Running on CPU — training will be very slow.")
        elif gpu_status == "warn":
            print("Continuing with warnings (low VRAM or older GPU).")
    
    # ── Data pipeline ───────────────────────────────────────────────
    if is_main_process():
        print("\n=== Preparing Data ===")
    
    from data import load_or_train_tokenizer, load_and_format_datasets, TextDataset
    
    # Load/format texts
    texts = load_and_format_datasets(config)
    
    # Tokenizer
    tokenizer = load_or_train_tokenizer(config, texts)
    config.vocab_size = tokenizer.vocab_size  # sync with actual tokenizer vocab
    
    if config.vocab_size != len(tokenizer):
        print(f"Warning: config.vocab_size ({config.vocab_size}) != tokenizer vocab ({len(tokenizer)}). Using tokenizer vocab.")
        config.vocab_size = len(tokenizer)
    
    # Create datasets
    val_size = max(1, int(len(texts) * config.val_split_pct))
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    
    if is_main_process():
        print(f"Train examples: {len(train_texts)}, Val examples: {len(val_texts)}")
    
    train_dataset = TextDataset(train_texts, tokenizer, seq_len=config.seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, seq_len=config.seq_len) if val_texts else None
    
    # DataLoaders with optional DDP sampler
    train_sampler = None
    val_sampler = None
    if use_ddp:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        if val_dataset:
            val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=True,
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.micro_batch_size,
            shuffle=False,
            sampler=val_sampler,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )
    
    # ── Model ───────────────────────────────────────────────────────
    if is_main_process():
        print("\n=== Building Model ===")
    
    model = create_model(config)
    
    if is_main_process():
        print(f"Model: {model.parameter_count_str()}")
    
    model = model.to(device)
    
    if config.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
        if is_main_process():
            print("Gradient checkpointing: enabled")
    
    # ── Optimizer & Scheduler ───────────────────────────────────────
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config)
    
    # Mixed precision
    scaler = None
    if config.mixed_precision == "fp16":
        scaler = torch.amp.GradScaler("cuda")
        if is_main_process():
            print("Mixed precision: FP16")
    else:
        if is_main_process():
            print("Mixed precision: BF16 (autocast)")
    
    # ── DDP wrapping ────────────────────────────────────────────────
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
        if is_main_process():
            print(f"DDP: enabled on {dist.get_world_size()} GPUs")
    
    # ── Resume from checkpoint ──────────────────────────────────────
    start_epoch = 0
    global_step = 0
    best_val_ppl = float("inf")
    
    ckpt_path = find_latest_checkpoint(config)
    if ckpt_path:
        state = load_checkpoint(ckpt_path, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        global_step = state["step"]
        best_val_ppl = state.get("metrics", {}).get("best_val_ppl", float("inf"))
        if is_main_process():
            print(f"\nResumed from: {ckpt_path}")
            print(f"  Epoch: {start_epoch}, Step: {global_step}, Best PPL: {best_val_ppl:.2f}")
    else:
        if is_main_process():
            print("\nStarting fresh training.")
    
    # ── Training ────────────────────────────────────────────────────
    if is_main_process():
        print(f"\n=== Training ({config.total_steps} steps) ===")
        print(f"  Effective batch size: {config.effective_batch_size}")
        print(f"  Sequence length: {config.seq_len}")
        print(f"  Steps per epoch: {len(train_loader) // config.gradient_accumulation_steps}")
    
    log_file = os.path.join(config.log_dir, "training.log")
    
    total_epochs = math.ceil(
        (config.total_steps - global_step) / (len(train_loader) // config.gradient_accumulation_steps)
    ) + start_epoch
    
    for epoch in range(start_epoch, total_epochs):
        if use_ddp and train_sampler:
            train_sampler.set_epoch(epoch)
        
        if is_main_process():
            print(f"\n--- Epoch {epoch + 1}/{total_epochs} ---")
        
        global_step, epoch_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            config, epoch, global_step, scaler, log_file,
        )
        
        # SWA update (main process only — DDP workers share weights via sync)
        if is_main_process() and config.use_swa and global_step >= config.swa_start_step:
            update_swa(model, config)
        
        # Evaluation
        if is_main_process() and val_loader:
            val_metrics = evaluate(model, val_loader, device)
            print(f"  Val Loss: {val_metrics['val_loss']:.4f} | Val PPL: {val_metrics['val_ppl']:.1f}")
            
            is_best = val_metrics["val_ppl"] < best_val_ppl
            if is_best:
                best_val_ppl = val_metrics["val_ppl"]
            
            # Save checkpoint at epoch end
            save_checkpoint(
                model, optimizer, scheduler, config,
                step=global_step, epoch=epoch,
                metrics={
                    "epoch_loss": epoch_loss,
                    "val_loss": val_metrics["val_loss"],
                    "val_ppl": val_metrics["val_ppl"],
                    "best_val_ppl": best_val_ppl,
                },
                is_best=is_best,
            )
        elif is_main_process():
            save_checkpoint(
                model, optimizer, scheduler, config,
                step=global_step, epoch=epoch,
                metrics={"epoch_loss": epoch_loss},
            )
        
        if global_step >= config.total_steps:
            break
    
    # ── Finalize ────────────────────────────────────────────────────
    if config.use_swa and _swa_started["done"]:
        finalize_swa(model)
        
        if is_main_process():
            final_path = os.path.join(config.checkpoint_dir, "final_swa.pt")
            model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
            torch.save({"model_state_dict": model_state, "config": {k: v for k, v in config.__dict__.items() if not k.startswith("_")}}, final_path)
            print(f"\nFinal SWA model saved: {final_path}")
    
    if is_main_process():
        print(f"\n=== Training Complete ===")
        print(f"  Total steps: {global_step}")
        print(f"  Best val PPL: {best_val_ppl:.2f}")
        print(f"  Checkpoints: {config.checkpoint_dir}")
    
    cleanup_ddp()


if __name__ == "__main__":
    # Allow config override from command line: python train.py --d_model 1536 --n_layers 20
    config = Config()
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith("--"):
            key = sys.argv[i][2:]
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--"):
                val = sys.argv[i + 1]
                i += 1
            else:
                val = "true"  # flag
            # Try to parse as number, then bool, then string
            try:
                if "." in val:
                    val = float(val)
                else:
                    val = int(val)
            except ValueError:
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
            setattr(config, key, val)
        i += 1
    
    main(config)
