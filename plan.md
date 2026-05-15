# MoE-BitNet LLM — Implementation Plan

> Based on SRS v1.0 (May 2026). English & Nepali Expert System with Modular Checkpoint Architecture.

## Phase 0: Environment Setup ✅
- [x] PyTorch environment with CUDA
- [x] HuggingFace libraries (datasets, tokenizers, accelerate, transformers)
- [x] GPU compatibility detection (P100 sm_60 → PyTorch 2.0.1+cu118, RTX → 2.2+)
- [x] Configurable checkpoint directory for Kaggle persistent storage

## Phase 1: Tokenizer & Data Pipeline ✅
- [x] BPE tokenizer with 32K vocab (shared EN+NE, configurable to 65K)
- [x] Dataset loading: yahma/alpaca-cleaned, OpenAssistant/oasst2
- [x] TextDataset with sequence chunking (2048 tokens)
- [x] Train/val split (98/2)
- [x] HuggingFace PreTrainedTokenizerFast wrapper for compatibility

## Phase 2: Architecture Implementation ✅
### Core Layers
- [x] BitLinear — BitNet b1.58 ternary weight quantization {-1, 0, +1}
- [x] STE (Straight-Through Estimator) for gradient flow through quantization
- [x] Per-token int8 activation quantization
- [x] Weight initialization: N(0, 0.006) per BitNet spec

### Transformer Components
- [x] RMSNorm — no bias, no mean subtraction
- [x] RoPE — Rotary Position Embedding with theta=10000
- [x] Grouped Query Attention (GQA) — 4:1 KV head ratio
- [x] SquaredReLU FFN — ReLU² activation for BitNet stability
- [x] SubLN pre-normalization (norm before attention + norm before FFN)
- [x] TransformerBlock assembly

### Full Model
- [x] BitNetTransformer — embedding + N blocks + final norm + LM head
- [x] Weight tying (LM head shares embedding weights)
- [x] Autoregressive generation (top-p, top-k, temperature, repetition penalty)
- [x] Gradient checkpointing support
- [x] Parameter counting utilities

## Phase 3: English Expert Training 🔄
- [x] Training loop with AdamW optimizer
- [x] Cosine LR schedule with linear warmup
- [x] Gradient clipping at norm 1.0
- [x] Mixed precision: BF16 autocast
- [x] Gradient accumulation
- [x] Stochastic Weight Averaging (SWA) from 80% of training
- [x] Full save/resume via checkpoints (optimizer, scheduler, RNG states)
- [x] DDP multi-GPU support (T4×2)
- [x] Per-layer quantization error monitoring
- [x] Periodic evaluation (perplexity on held-out set)
- [ ] **Train english_expert.pt to target PPL < 10**
- [ ] Wandb integration for experiment tracking

## Phase 4: Nepali Expert Training ⬜
- [ ] Dataset integration: Nepali Wikipedia, CC-100 Nepali, OSCAR Nepali, FLORES-200
- [ ] Curriculum learning (Wikipedia → CC-100)
- [ ] Data augmentation via back-translation
- [ ] Reduced seq_len=2048, up to 3 epochs
- [ ] **Train nepali_expert.pt to target PPL < 15**
- [ ] Monitor quantization error per layer (Nepali script sensitivity)

## Phase 5: Translation Module ⬜
- [ ] Encoder-decoder transformer (12+12 layers, 1024 dim, ~350M params)
- [ ] Freeze both expert checkpoints during training
- [ ] Latent-space bridging: source encoder output → translation decoder → target decoder input
- [ ] Teacher forcing with target expert embeddings
- [ ] Parallel corpus: FLORES-200, CCAligned, OPUS EN-NE
- [ ] **Train translation_model.pt to target BLEU > 25**
- [ ] Evaluation on FLORES-200 dev set

## Phase 6: TRM Integration ⬜
- [ ] Implement Tiny Recursive Model refinement head (~256M params)
- [ ] K=3 iterative refinement loop on latent representation
- [ ] RL-style reward signal (ROUGE/BLEU for refinement quality)
- [ ] TRM stored as separate expert_trm.pt — never modifies base weights
- [ ] **Train TRM heads (20k steps)**

## Phase 7: Evaluation & Optimization ⬜
- [ ] Perplexity benchmarks on held-out sets
- [ ] BLEU scores on FLORES-200 EN-NE
- [ ] Grammar accuracy on validation sets
- [ ] Inference latency measurement (< 200ms/token target)
- [ ] VRAM profiling (target < 10GB per expert at inference)
- [ ] PyTorch Profiler + Nsight profiling
- [ ] KV-cache optimization
- [ ] Conditional checkpoint loading logic (idle timeout, LRU eviction)

## Phase 8: Deployment ⬜
- [ ] Inference manager with conditional loading protocol
- [ ] Language detection router
- [ ] Checkpoint file packaging (english_expert.pt, nepali_expert.pt, translation_model.pt)
- [ ] Shared tokenizer packaging (tokenizer.json)
- [ ] Router network packaging (router.pt)
- [ ] Load testing and stress testing

---

## GPU Configurations
| GPU | VRAM | Recommended Config | Expected Params |
|-----|------|-------------------|----------------|
| P100 | 16 GB | SmallConfig | ~100M |
| T4 | 16 GB | SmallConfig | ~100M |
| T4×2 | 32 GB | MediumConfig | ~213M |
| RTX 3090 | 24 GB | MediumConfig | ~213M |
| RTX 4090 | 24 GB | MediumConfig | ~213M |
| A100 | 40 GB | LargeConfig | ~545M |
| RTX (SRS target) | 96 GB | Custom (SRS spec) | 4B |

## Key Files
```
UN/
├── config.py              Configuration + GPU detection
├── bitlinear.py           BitNet b1.58 quantization layer
├── model_components.py    RMSNorm, RoPE, GQA, FFN, TransformerBlock
├── model.py               Full BitNetTransformer + generation
├── data.py                Dataset loading, tokenizer, DataLoaders
├── train.py               Training loop + checkpointing + DDP
├── inference.py           Interactive/batch inference
├── requirements.txt       Dependencies
├── plan.md                This file
└── progress.md            Progress tracker
```
