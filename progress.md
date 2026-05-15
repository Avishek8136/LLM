# MoE-BitNet LLM — Progress Tracker

> Last updated: 2026-05-15 | Current Phase: Phase 3 (English Expert Training)

## Overall Status

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 0: Environment Setup | ✅ Complete | 100% |
| Phase 1: Tokenizer & Data Pipeline | ✅ Complete | 100% |
| Phase 2: Architecture Implementation | ✅ Complete | 100% |
| Phase 3: English Expert Training | 🔄 In Progress | 10% (code ready, training starting) |
| Phase 4: Nepali Expert Training | ⬜ Not Started | 0% |
| Phase 5: Translation Module | ⬜ Not Started | 0% |
| Phase 6: TRM Integration | ⬜ Not Started | 0% |
| Phase 7: Evaluation & Optimization | ⬜ Not Started | 0% |
| Phase 8: Deployment | ⬜ Not Started | 0% |

## Current Sprint: Phase 3 — English Expert Training

### Active Dataset
- **yahma/alpaca-cleaned**: 51,760 examples ✅ Loaded
- **OpenAssistant/oasst2**: 124,782 valid examples ✅ Loaded
- **Total**: 176,542 formatted examples, 27.6M tokens, 13,469 sequences of 2048

### Training Configuration
- **Model**: MediumConfig (~213M params) or SmallConfig (~100M for P100)
- **Optimizer**: AdamW (lr=3e-4, cosine to 3e-5)
- **Steps**: 30,000 total
- **Effective batch**: 16 (2 micro × 8 accumulation)
- **Target PPL**: < 10 on validation

### Known Issues
| # | Issue | Status | Resolution |
|---|-------|--------|------------|
| 1 | P100 CUDA capability mismatch (sm_60 vs sm_70+) | 🔧 Fixed | Added GPU compatibility check + PyTorch 2.0.1+cu118 instructions |
| 2 | 426M params on P100 16GB may OOM | ⚠️ Monitor | Use SmallConfig (100M) for P100 |
| 3 | Tokenizer vocab vs model vocab sync | 🔧 Fixed | Auto-sync in train.py main() |

### Next Steps
1. Install PyTorch 2.0.1+cu118 for P100 compatibility
2. Run training with SmallConfig on P100
3. Monitor loss, PPL, and quantization error
4. Evaluate first checkpoint at step 2000
5. Resume training if interrupted (auto-discovers latest.pt)

---

## Checkpoint Registry

| Checkpoint | Step | Epoch | Val PPL | Notes |
|-----------|------|-------|---------|-------|
| None yet | - | - | - | Training not started |

---

## Metrics Log

### Training Metrics (target: loss < 2.3 for PPL < 10)
| Date | Step | Epoch | Train Loss | Train PPL | Val Loss | Val PPL | Q Error | LR |
|------|------|-------|-----------|-----------|---------|---------|---------|-----|
| - | - | - | - | - | - | - | - | - |

### Hardware Metrics
| GPU | Model Size | Batch Size | VRAM Used | Steps/sec | Tokens/sec |
|-----|-----------|------------|-----------|-----------|------------|
| - | - | - | - | - | - |

---

## File Manifest

```
UN/
├── config.py              ✅ v1.0 — 3 presets + GPU detection
├── bitlinear.py           ✅ v1.0 — STE + ternary + int8
├── model_components.py    ✅ v1.0 — RMSNorm, RoPE, GQA, FFN
├── model.py               ✅ v1.0 — Full transformer + generation
├── data.py                ✅ v1.0 — Dataset loading + BPE tokenizer
├── train.py               ✅ v1.0 — Full loop + DDP + resume
├── inference.py           ✅ v1.0 — Interactive/batch inference
├── requirements.txt       ✅ v1.0 — PyTorch 2.0.1+ compatible
├── plan.md                ✅ v1.0 — This implementation plan
└── progress.md            ✅ v1.0 — This progress tracker
```
