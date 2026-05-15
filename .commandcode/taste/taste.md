# Architecture
- Use BitNet b1.58 ternary quantization with BitLinear layers for all linear transforms. Confidence: 0.75

# Training
- Support training pause/resume via checkpointing so training restarts from the last saved epoch. Confidence: 0.70
- Ensure training code is compatible with T4x2 and P100 GPU configurations (modest VRAM, multi-GPU via DDP). Confidence: 0.70
