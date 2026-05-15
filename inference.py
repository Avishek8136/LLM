"""
Inference script for BitNet b1.58 LLM.
Loads a trained checkpoint and runs interactive or batch generation.
Usage: python inference.py --checkpoint checkpoints/latest.pt --prompt "Hello"
       python inference.py --checkpoint checkpoints/latest.pt --interactive
"""

import sys
import os
import torch

from config import Config
from model import BitNetTransformer, create_model


def load_model_for_inference(checkpoint_path: str, device: str = "cuda"):
    """Load model from checkpoint for inference."""
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Reconstruct config
    if "config" in ckpt:
        cfg_dict = ckpt["config"]
        # Build a minimal config
        class InferConfig:
            pass
        config = InferConfig()
        for k, v in cfg_dict.items():
            setattr(config, k, v)
    else:
        config = Config()
    
    model = BitNetTransformer(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_layers=config.n_layers,
        n_heads=config.n_heads,
        n_kv_heads=config.n_kv_heads,
        ffn_dim=config.ffn_dim,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
    )
    
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded: {model.parameter_count_str()}")
    return model, config


def load_tokenizer(checkpoint_dir: str):
    """Load tokenizer from checkpoint directory."""
    from transformers import PreTrainedTokenizerFast
    
    hf_path = os.path.join(checkpoint_dir, "tokenizer_hf")
    json_path = os.path.join(checkpoint_dir, "tokenizer.json")
    
    # Try HuggingFace path first
    if os.path.exists(hf_path):
        return PreTrainedTokenizerFast.from_pretrained(hf_path)
    
    # Try raw tokenizer file
    if os.path.exists(json_path):
        from tokenizers import Tokenizer
        raw_tok = Tokenizer.from_file(json_path)
        return PreTrainedTokenizerFast(
            tokenizer_object=raw_tok,
            bos_token="<bos>",
            eos_token="<eos>",
            pad_token="<pad>",
            unk_token="<unk>",
        )
    
    raise FileNotFoundError(f"No tokenizer found in {checkpoint_dir}")


def generate_response(
    model: BitNetTransformer,
    tokenizer,
    prompt: str,
    config,
    device: str = "cuda",
) -> str:
    """Generate a response for a given prompt."""
    # Format prompt for instruction model
    formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{prompt}\n\n### Response:\n"
    
    input_ids = tokenizer.encode(formatted_prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            input_tensor,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=config.repetition_penalty,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    generated_ids = generated_ids[0].tolist()
    # Decode only the new tokens
    new_tokens = generated_ids[len(input_ids):]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return response.strip()


def interactive_mode(model, tokenizer, config, device: str):
    """Run interactive chat loop."""
    print("\n=== BitNet LLM Interactive Mode ===")
    print("Type 'quit' or 'exit' to stop.\n")
    
    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if not prompt:
            continue
        
        print("Assistant: ", end="", flush=True)
        response = generate_response(model, tokenizer, prompt, config, device)
        print(response)
        print()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="BitNet LLM Inference")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt for generation")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    model, config = load_model_for_inference(args.checkpoint, args.device)
    
    # Override config with CLI args
    config.temperature = args.temperature
    config.top_p = args.top_p
    config.top_k = args.top_k
    config.max_new_tokens = args.max_tokens
    
    # Load tokenizer from same directory as checkpoint
    checkpoint_dir = os.path.dirname(args.checkpoint)
    tokenizer = load_tokenizer(checkpoint_dir)
    
    if args.interactive:
        interactive_mode(model, tokenizer, config, args.device)
    elif args.prompt:
        response = generate_response(model, tokenizer, args.prompt, config, args.device)
        print(response)
    else:
        # Default: generate a test response
        test_prompt = "Explain what BitNet b1.58 quantization is in simple terms."
        print(f"Prompt: {test_prompt}\n")
        response = generate_response(model, tokenizer, test_prompt, config, args.device)
        print(f"Response: {response}")


if __name__ == "__main__":
    main()
