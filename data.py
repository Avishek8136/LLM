"""
Data pipeline: dataset loading, preprocessing, BPE tokenizer training, DataLoader.
Handles Alpaca, OpenAssistant, and other instruction/conversation datasets.
"""

import os
import json
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from datasets import load_dataset, concatenate_datasets, Dataset as HFDataset


# ── Dataset formatting ──────────────────────────────────────────────────
ALPACA_PROMPT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

ALPACA_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_alpaca(example: dict) -> str:
    """Format a single Alpaca example into a text string."""
    instruction = example.get("instruction", "").strip()
    inp = example.get("input", "").strip()
    output = example.get("output", "").strip()
    
    if inp:
        return ALPACA_PROMPT.format(instruction=instruction, input=inp, output=output)
    else:
        return ALPACA_PROMPT_NO_INPUT.format(instruction=instruction, output=output)


def format_oasst2(example: dict) -> str:
    """Format an OpenAssistant message tree into conversation text.
    
    OASST2 has a tree structure. We extract linear conversation threads.
    """
    messages = []
    if "messages" in example:
        # Handle nested structure
        for msg in example["messages"]:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = "user"
                content = str(msg)
            messages.append(f"{role.title()}: {content}")
    elif "text" in example:
        return example["text"]
    else:
        # Try common fields
        parts = []
        for key in ["instruction", "input", "output", "text", "prompt", "response"]:
            if key in example and example[key]:
                if key == "instruction":
                    parts.append(f"Instruction: {example[key]}")
                elif key == "input":
                    parts.append(f"Input: {example[key]}")
                elif key in ("output", "response"):
                    parts.append(f"Response: {example[key]}")
                else:
                    parts.append(str(example[key]))
        if parts:
            return "\n\n".join(parts)
        return json.dumps(example)
    
    return "\n".join(messages)


def load_and_format_datasets(config) -> list[str]:
    """Load all configured datasets and return list of formatted text strings."""
    all_texts = []
    
    for ds_name in config.datasets:
        print(f"  Loading dataset: {ds_name}")
        try:
            ds = load_dataset(ds_name, split=config.dataset_split)
        except Exception as e:
            print(f"  Warning: Could not load {ds_name} ({e}), trying 'train' split...")
            try:
                ds = load_dataset(ds_name, split="train")
            except Exception as e2:
                print(f"  Skipping {ds_name}: {e2}")
                continue
        
        if ds is None:
            continue
        
        print(f"  Loaded {len(ds)} examples from {ds_name}")
        
        is_alpaca = "alpaca" in ds_name.lower()
        
        count = 0
        for example in ds:
            try:
                if is_alpaca:
                    text = format_alpaca(example)
                elif "oasst" in ds_name.lower():
                    text = format_oasst2(example)
                else:
                    text = format_oasst2(example)
                
                if text and len(text.strip()) > 10:
                    all_texts.append(text)
                    count += 1
            except Exception:
                continue
        
        print(f"  Formatted {count} valid examples")
    
    print(f"Total formatted examples: {len(all_texts)}")
    return all_texts


# ── Tokenizer training ──────────────────────────────────────────────────
def train_tokenizer(
    texts: list[str],
    vocab_size: int = 32000,
    save_path: str | None = None,
) -> "Tokenizer":
    """Train a BPE tokenizer on the provided texts using HuggingFace Tokenizers."""
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, trainers
    
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[
            "<pad>", "<unk>", "<bos>", "<eos>",
            "<|user|>", "<|assistant|>", "<|system|>",
            "<|instruction|>", "<|response|>",
        ],
        min_frequency=2,
        show_progress=True,
    )
    
    # Train from iterator
    def text_iterator():
        for text in texts:
            yield text
    
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    
    # Set post-processing
    tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
    tokenizer.enable_truncation(max_length=4096)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
        print(f"Tokenizer saved to {save_path}")
    
    return tokenizer


def load_or_train_tokenizer(config, texts: list[str]):
    """Load tokenizer from disk or train a new one."""
    from transformers import PreTrainedTokenizerFast
    
    tokenizer_save_path = config.tokenizer_path
    hf_path = tokenizer_save_path.replace(".json", "_hf")
    
    # Try loading HuggingFace-compatible tokenizer first
    try:
        if os.path.exists(os.path.join(hf_path, "tokenizer.json")) or os.path.exists(hf_path):
            print(f"Loading existing tokenizer from {hf_path}")
            tokenizer = PreTrainedTokenizerFast.from_pretrained(hf_path)
            return tokenizer
    except Exception:
        pass
    
    # Try raw tokenizer
    if os.path.exists(tokenizer_save_path):
        print(f"Loading existing tokenizer from {tokenizer_save_path}")
        from tokenizers import Tokenizer
        raw_tok = Tokenizer.from_file(tokenizer_save_path)
    else:
        print(f"Training new BPE tokenizer (vocab_size={config.vocab_size})...")
        raw_tok = train_tokenizer(texts, vocab_size=config.vocab_size, save_path=tokenizer_save_path)
    
    # Wrap in HuggingFace PreTrainedTokenizerFast for compatibility
    os.makedirs(hf_path, exist_ok=True)
    
    # Save as HF-compatible
    tokenizer_dict = json.loads(raw_tok.to_str())
    # Save tokenizer.json in HF format
    with open(os.path.join(hf_path, "tokenizer.json"), "w") as f:
        json.dump(tokenizer_dict, f, ensure_ascii=False)
    
    # Create minimal config
    hf_config = {
        "model_type": "bitnet_llm",
        "vocab_size": raw_tok.get_vocab_size(),
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "unk_token": "<unk>",
    }
    with open(os.path.join(hf_path, "tokenizer_config.json"), "w") as f:
        json.dump(hf_config, f)
    
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tok,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(hf_path)
    
    print(f"Tokenizer ready: vocab_size={tokenizer.vocab_size}, path={hf_path}")
    return tokenizer


# ── PyTorch Dataset ─────────────────────────────────────────────────────
class TextDataset(Dataset):
    """Tokenized text dataset for language model training.
    
    Chunks all texts into fixed-length sequences for efficient training.
    """
    
    def __init__(
        self,
        texts: list[str],
        tokenizer,
        seq_len: int = 2048,
    ):
        self.seq_len = seq_len
        
        print("Tokenizing dataset...")
        all_token_ids = []
        chunk_size = 10000
        for i in range(0, len(texts), chunk_size):
            batch = texts[i : i + chunk_size]
            encoded = tokenizer(
                batch,
                add_special_tokens=True,
                truncation=False,
                padding=False,
                return_attention_mask=False,
            )
            for ids in encoded["input_ids"]:
                ids.append(tokenizer.eos_token_id)
                all_token_ids.extend(ids)
            if i % (chunk_size * 10) == 0 and i > 0:
                print(f"  Tokenized {i}/{len(texts)} texts...")
        
        print(f"Total tokens: {len(all_token_ids):,}")
        
        # Chunk into sequences of seq_len
        self.sequences = []
        for i in range(0, len(all_token_ids) - seq_len, seq_len):
            chunk = all_token_ids[i : i + seq_len + 1]  # +1 for target
            if len(chunk) == seq_len + 1:
                self.sequences.append(chunk)
        
        # Handle remaining tokens
        if len(all_token_ids) % seq_len > 1:
            remaining = all_token_ids[-(seq_len + 1):]
            if len(remaining) < seq_len + 1:
                remaining = [tokenizer.pad_token_id] * (seq_len + 1 - len(remaining)) + remaining
            self.sequences.append(remaining)
        
        print(f"Created {len(self.sequences)} sequences of length {seq_len}")
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = self.sequences[idx]
        input_ids = torch.tensor(seq[:-1], dtype=torch.long)
        labels = torch.tensor(seq[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def create_dataloaders(config, tokenizer) -> tuple[DataLoader, DataLoader | None]:
    """Create training and validation DataLoaders."""
    print("Preparing training data...")
    texts = load_and_format_datasets(config)
    
    # Split into train/val
    val_size = max(1, int(len(texts) * config.val_split_pct))
    train_texts = texts[val_size:]
    val_texts = texts[:val_size]
    
    print(f"Train texts: {len(train_texts)}, Val texts: {len(val_texts)}")
    
    train_dataset = TextDataset(train_texts, tokenizer, seq_len=config.seq_len)
    val_dataset = TextDataset(val_texts, tokenizer, seq_len=config.seq_len) if val_texts else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.micro_batch_size,
        shuffle=True,
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
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=False,
        )
    
    return train_loader, val_loader
