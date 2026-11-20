"""
LLM Fine-Tuning (Causal Language Model)
=========================================
Fine-tuning a pretrained Large Language Model (LLM) on a downstream task
using the Hugging Face `transformers` ecosystem. Covers full fine-tuning,
instruction fine-tuning, and LoRA (Low-Rank Adaptation) — the most
parameter-efficient approach for consumer hardware.

Architecture options covered:
    GPT-2        — small, fully open, good for learning the pipeline
    LLaMA 2/3    — state-of-the-art open-source LLM (requires HF access)
    Mistral      — strong performance at 7B parameters
    Phi-2/3      — Microsoft's small but capable models

Training paradigms:
    Full fine-tuning:     update ALL model parameters (expensive, highest quality)
    LoRA / QLoRA:         inject trainable low-rank matrices into attention layers
                          (train <1% of parameters, 90%+ memory reduction)
    Instruction tuning:   format data as instruction-response pairs for chat/assistants
    RLHF / DPO:           align with human preferences (beyond scope here)

Use Cases:
    - Domain-specific text generation (medical, legal, code)
    - Instruction following / chatbot
    - Text classification via generation ("Is this positive or negative? Answer:")
    - Summarization, translation, question answering
    - Code generation and completion

Fine-Tuning Recommendations:
    - Start with LoRA: reduces GPU memory by 10-50×; close to full fine-tuning quality.
    - LoRA hyperparameters:
        rank (r):    4–64. Higher rank = more capacity, more memory. Start at 8–16.
        alpha:       usually set to 2×r (scaling factor for LoRA updates).
        target_modules: 'q_proj', 'v_proj' are most impactful;
                        add 'k_proj', 'o_proj' for more capacity.
        dropout:     0.05–0.1 for LoRA layers.
    - QLoRA: 4-bit quantization + LoRA. Enables fine-tuning 7B models on 24GB GPU.
    - Learning rate: 1e-4 to 1e-5. LLMs are sensitive — use warmup.
    - Batch size: effective batch size = per_device_batch * grad_accumulation_steps.
      Target 64–128 for instruction tuning.
    - Max sequence length: 512–2048. Longer contexts = more memory. Truncate as needed.
    - Epochs: 1–3 for instruction tuning; 3–5 for domain adaptation.
      More epochs → overfitting on small datasets.
    - Data format: standardize instruction-input-output format (Alpaca, ChatML, etc.)
    - Flash Attention 2: speeds up training by 2-4× with identical outputs.
    - Gradient checkpointing: trades compute for memory (enables larger batches).
"""

import os
import json
from dataclasses import dataclass, field
from typing import Optional

import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)

# ─────────────────────────────────────────────
# OPTIONAL: PEFT (for LoRA / QLoRA)
# ─────────────────────────────────────────────
# Install: pip install peft bitsandbytes
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        prepare_model_for_kbit_training,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("PEFT not installed. LoRA training disabled. Run: pip install peft bitsandbytes")

import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# MODEL CONFIGURATION
# ─────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Central configuration for model selection and loading.

    Placeholder: swap model_name to change the base model.
    GPT-2 is used as default (no auth required, fast to download).
    For production-quality models, use LLaMA-2/3 or Mistral.
    """

    # Placeholder: change this to use a different model
    model_name: str = "gpt2"
    # Other options (require HuggingFace access/token):
    #   "meta-llama/Llama-2-7b-hf"           — LLaMA 2 7B (requires HF auth)
    #   "meta-llama/Meta-Llama-3-8B"          — LLaMA 3 8B
    #   "mistralai/Mistral-7B-v0.1"           — Mistral 7B
    #   "microsoft/phi-2"                     — Phi-2 (2.7B)
    #   "microsoft/Phi-3-mini-4k-instruct"    — Phi-3 Mini

    # Placeholder: set to True to use 4-bit quantization (requires bitsandbytes)
    # QLoRA: enables fine-tuning 7B models on consumer GPUs (24GB VRAM)
    use_4bit_quantization: bool = False

    # Compute dtype for quantized models
    bnb_4bit_compute_dtype: str = "float16"   # 'float16' or 'bfloat16'
    bnb_4bit_quant_type: str = "nf4"          # 'nf4' (recommended) or 'fp4'

    # HuggingFace token for gated models (set via env var HF_TOKEN)
    hf_token: Optional[str] = field(
        default_factory=lambda: os.environ.get("HF_TOKEN")
    )


@dataclass
class LoRAConfig:
    """
    LoRA (Low-Rank Adaptation) hyperparameters.

    LoRA injects trainable rank decomposition matrices A and B into the
    attention weight matrices: W' = W + αAB/r
    Only A and B are trained; the original W is frozen.

    This reduces trainable parameters from ~7B → ~4M for a 7B model.

    Fine-tuning guidance:
        r=8, alpha=16:   conservative (less overfitting, less capacity)
        r=16, alpha=32:  balanced (most common starting point)
        r=64, alpha=128: aggressive (more capacity, more parameters)
    """

    enabled: bool = True          # Placeholder: set to False for full fine-tuning
    r: int = 8                    # Placeholder: LoRA rank; try 8, 16, 32, 64
    alpha: int = 16               # Placeholder: scaling = alpha/r; typically alpha = 2×r
    dropout: float = 0.05         # Placeholder: try 0.05–0.1

    # Which weight matrices to apply LoRA to.
    # 'q_proj', 'v_proj' — minimal (standard BERT/LLaMA attention)
    # Adding 'k_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj' increases capacity
    # Placeholder: adjust for your model's architecture
    target_modules: list[str] = field(
        default_factory=lambda: ["c_attn"]  # GPT-2 uses 'c_attn' (combined QKV)
        # For LLaMA: ["q_proj", "v_proj", "k_proj", "o_proj"]
        # For Mistral: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"]
    )
    bias: str = "none"            # 'none', 'all', 'lora_only'


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Placeholder: set to your dataset name or local path
    dataset_name: str = "wikitext"      # HuggingFace dataset hub name
    dataset_config: str = "wikitext-2-raw-v1"  # dataset configuration
    text_column: str = "text"           # column containing text data
    max_length: int = 256               # Placeholder: 512–2048 for production
    train_split: str = "train"
    val_split: str = "validation"


@dataclass
class TrainConfig:
    """Training hyperparameters."""

    output_dir: str = "./finetuned_llm"
    num_train_epochs: int = 3             # Placeholder: 1–5 for instruction tuning
    per_device_train_batch_size: int = 4  # Placeholder: increase if VRAM allows
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8  # effective batch = 4 × 8 = 32
    learning_rate: float = 2e-4           # Placeholder: 1e-5 to 3e-4
    lr_scheduler_type: str = "cosine"     # 'cosine', 'linear', 'constant'
    warmup_ratio: float = 0.05            # 5% of steps for warmup
    weight_decay: float = 0.01            # L2 regularization
    max_grad_norm: float = 1.0            # gradient clipping
    fp16: bool = not torch.backends.mps.is_available()  # AMP: faster training on GPU
    bf16: bool = False                    # use bfloat16 instead of float16 (better for LLaMA)
    evaluation_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    logging_steps: int = 50
    save_total_limit: int = 2             # keep only best 2 checkpoints
    gradient_checkpointing: bool = True   # trades compute for memory; recommended for large models
    report_to: str = "none"              # Placeholder: "wandb" or "tensorboard" for experiment tracking


# ─────────────────────────────────────────────
# DATA PROCESSING
# ─────────────────────────────────────────────

def load_and_prepare_dataset(
    data_cfg: DataConfig,
    tokenizer,
) -> tuple[Dataset, Dataset]:
    """
    Load a text dataset and tokenize it for causal language modeling.

    For causal LM: input = all tokens, labels = input shifted by 1
    (the model predicts the next token at each position).

    This function handles the 'text' → 'input_ids' conversion.

    Replace with your own data loading:
        texts = ["instruction: ...\nresponse: ...", ...]
        dataset = Dataset.from_dict({"text": texts})

    For instruction tuning, use the Alpaca format:
        def format_alpaca(example):
            if example["input"]:
                return {"text": f"### Instruction:\\n{example['instruction']}\\n"
                                 f"### Input:\\n{example['input']}\\n"
                                 f"### Response:\\n{example['output']}"}
            else:
                return {"text": f"### Instruction:\\n{example['instruction']}\\n"
                                 f"### Response:\\n{example['output']}"}

    Args:
        data_cfg:  DataConfig instance
        tokenizer: HuggingFace tokenizer

    Returns:
        tokenized train_dataset, eval_dataset
    """
    print(f"Loading dataset: {data_cfg.dataset_name}")
    raw = load_dataset(data_cfg.dataset_name, data_cfg.dataset_config)

    def tokenize(examples):
        # Tokenize with padding and truncation
        output = tokenizer(
            examples[data_cfg.text_column],
            truncation=True,
            max_length=data_cfg.max_length,
            padding="max_length",
            return_tensors=None,
        )
        # For causal LM: labels = input_ids (shifted internally by the model)
        output["labels"] = output["input_ids"].copy()
        return output

    # Filter empty texts before tokenizing
    train_raw = raw[data_cfg.train_split].filter(
        lambda x: x[data_cfg.text_column] is not None and len(x[data_cfg.text_column].strip()) > 10
    )
    val_raw   = raw[data_cfg.val_split].filter(
        lambda x: x[data_cfg.text_column] is not None and len(x[data_cfg.text_column].strip()) > 10
    )

    train_ds = train_raw.map(tokenize, batched=True, remove_columns=train_raw.column_names)
    val_ds   = val_raw.map(tokenize,   batched=True, remove_columns=val_raw.column_names)

    train_ds.set_format("torch")
    val_ds.set_format("torch")

    print(f"  Train samples: {len(train_ds):,}")
    print(f"  Val samples:   {len(val_ds):,}")
    return train_ds, val_ds


def format_instruction_sample(
    instruction: str,
    response: str,
    input_text: str = "",
    system_prompt: str = "",
) -> str:
    """
    Format a single sample for instruction fine-tuning (Alpaca/LLaMA format).

    Placeholder: change the template to match your target model's chat format.
    Common formats:
        Alpaca:  "### Instruction:\\n...\\n### Response:\\n..."
        ChatML:  "<|im_start|>system\\n...\\n<|im_end|>\\n..."
        LLaMA3:  "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\\n..."

    Args:
        instruction: the user instruction/question
        response:    the desired model response
        input_text:  optional additional context
        system_prompt: optional system prompt for chat models

    Returns:
        formatted string ready for tokenization
    """
    if system_prompt:
        prompt = f"<|system|>\n{system_prompt}\n"
    else:
        prompt = ""

    if input_text:
        prompt += f"<|user|>\n{instruction}\n\nContext: {input_text}\n<|assistant|>\n{response}"
    else:
        prompt += f"<|user|>\n{instruction}\n<|assistant|>\n{response}"

    return prompt


# ─────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────

def load_model_and_tokenizer(
    model_cfg: ModelConfig,
    lora_cfg: LoRAConfig,
):
    """
    Load model and tokenizer with optional quantization and LoRA.

    Loading flow:
        1. Load tokenizer (handles text → tokens)
        2. Load model (optionally quantized with BitsAndBytes)
        3. Apply LoRA: freeze base model, inject trainable adapters
        4. Print parameter counts

    Quantization (BitsAndBytes):
        4-bit NF4 quantization dramatically reduces memory:
            7B model: ~14GB (float16) → ~4GB (4-bit)
        Requires bitsandbytes: pip install bitsandbytes

    Args:
        model_cfg: ModelConfig instance
        lora_cfg:  LoRAConfig instance

    Returns:
        model, tokenizer
    """
    print(f"Loading model: {model_cfg.model_name}")

    # ── Tokenizer ────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        model_cfg.model_name,
        token=model_cfg.hf_token,
        trust_remote_code=True,
    )

    # Most LLMs don't have a pad token by default — set it for training
    if tokenizer.pad_token is None:
        tokenizer.pad_token     = tokenizer.eos_token
        tokenizer.pad_token_id  = tokenizer.eos_token_id

    # Placeholder: set padding side for decoder-only models
    tokenizer.padding_side = "right"  # 'right' for training; 'left' for generation

    # ── Quantization config (4-bit QLoRA) ───────────────────────
    bnb_config = None
    if model_cfg.use_4bit_quantization:
        if not PEFT_AVAILABLE:
            raise ImportError("bitsandbytes required for 4-bit quantization. "
                              "Run: pip install bitsandbytes")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=getattr(torch, model_cfg.bnb_4bit_compute_dtype),
            bnb_4bit_use_double_quant=True,  # nested quantization saves extra memory
        )

    # ── Load model ───────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg.model_name,
        quantization_config=bnb_config,
        device_map="auto" if bnb_config else None,
        token=model_cfg.hf_token,
        trust_remote_code=True,
        torch_dtype=torch.float16 if not bnb_config else None,
    )

    if not bnb_config:
        model = model.to(
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )

    # ── LoRA ────────────────────────────────────────────────────
    if lora_cfg.enabled and PEFT_AVAILABLE:
        if model_cfg.use_4bit_quantization:
            # Prepare 4-bit model for LoRA training
            model = prepare_model_for_kbit_training(model)

        lora_config = LoraConfig(
            r=lora_cfg.r,
            lora_alpha=lora_cfg.alpha,
            target_modules=lora_cfg.target_modules,
            lora_dropout=lora_cfg.dropout,
            bias=lora_cfg.bias,
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    else:
        # Full fine-tuning: all parameters are trainable
        total = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters (full fine-tune): {total:,}")

    return model, tokenizer


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def fine_tune(
    model,
    tokenizer,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    train_cfg: TrainConfig,
) -> Trainer:
    """
    Run the fine-tuning loop using HuggingFace Trainer.

    The Trainer handles: batching, gradient accumulation, mixed precision,
    checkpointing, logging, evaluation, and LR scheduling.

    Key arguments to tune:
        - learning_rate: most impactful; start at 2e-4 for LoRA, 2e-5 for full
        - num_train_epochs: 1–3 for instruction, 3–5 for domain adaptation
        - gradient_accumulation_steps: multiply with batch_size for effective batch
        - warmup_ratio: ramp LR from 0 to peak over this fraction of steps

    For experiment tracking, set report_to="wandb" and run: wandb login

    Args:
        model:          fine-tuned or LoRA model
        tokenizer:      corresponding tokenizer
        train_dataset:  tokenized training data
        eval_dataset:   tokenized validation data
        train_cfg:      TrainConfig instance

    Returns:
        trained Trainer object
    """
    training_args = TrainingArguments(
        output_dir=train_cfg.output_dir,
        num_train_epochs=train_cfg.num_train_epochs,
        per_device_train_batch_size=train_cfg.per_device_train_batch_size,
        per_device_eval_batch_size=train_cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        learning_rate=train_cfg.learning_rate,
        lr_scheduler_type=train_cfg.lr_scheduler_type,
        warmup_ratio=train_cfg.warmup_ratio,
        weight_decay=train_cfg.weight_decay,
        max_grad_norm=train_cfg.max_grad_norm,
        fp16=train_cfg.fp16,
        bf16=train_cfg.bf16,
        evaluation_strategy=train_cfg.evaluation_strategy,
        save_strategy=train_cfg.save_strategy,
        load_best_model_at_end=train_cfg.load_best_model_at_end,
        metric_for_best_model=train_cfg.metric_for_best_model,
        greater_is_better=train_cfg.greater_is_better,
        logging_steps=train_cfg.logging_steps,
        save_total_limit=train_cfg.save_total_limit,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        report_to=train_cfg.report_to,
        dataloader_pin_memory=False,
    )

    # DataCollatorForLanguageModeling handles the label shifting for causal LM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,   # False = causal LM (GPT-style); True = masked LM (BERT-style)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    print("\nStarting fine-tuning...")
    trainer.train()
    return trainer


# ─────────────────────────────────────────────
# GENERATION / INFERENCE
# ─────────────────────────────────────────────

def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
    do_sample: bool = True,
) -> str:
    """
    Generate text from the fine-tuned model.

    Generation parameters:
        temperature:  controls randomness. 0 = deterministic (greedy).
                      0.7–1.0 = balanced creativity and coherence.
        top_p:        nucleus sampling — only sample from tokens with cumulative
                      probability > top_p. 0.9 is standard.
        top_k:        limit sampling to the top-k most likely tokens.
        repetition_penalty: > 1.0 penalizes repeated tokens. 1.1–1.3 reduces loops.
        do_sample:    True = sampling; False = greedy/beam search.

    Fine-tuning tip: if generation is repetitive or degenerate:
        - Increase temperature slightly (0.8–1.0)
        - Increase repetition_penalty (1.1–1.3)
        - Check that the model was not over-trained (too many epochs)

    Args:
        model:              fine-tuned model
        tokenizer:          corresponding tokenizer
        prompt:             input text to continue
        max_new_tokens:     maximum tokens to generate
        temperature:        sampling temperature
        top_p:              nucleus sampling threshold
        top_k:              top-k sampling
        repetition_penalty: penalty for repeated tokens
        do_sample:          whether to sample (True) or use greedy decoding (False)

    Returns:
        generated text string (excluding the input prompt)
    """
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    # Decode only the new tokens (exclude the input prompt tokens)
    new_tokens = output_ids[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ─────────────────────────────────────────────
# SAVING AND LOADING
# ─────────────────────────────────────────────

def save_model(model, tokenizer, output_dir: str) -> None:
    """
    Save the fine-tuned model and tokenizer.

    For LoRA models: saves only the adapter weights (very small, ~few MB).
    For full fine-tuning: saves all model weights (large, ~7GB+ for 7B).

    To merge LoRA weights back into the base model for deployment:
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to: {output_dir}")


def load_finetuned_model(
    base_model_name: str,
    adapter_dir: str,
    use_lora: bool = True,
):
    """
    Load a fine-tuned model from disk.

    For LoRA: load base model + adapter weights.
    For full fine-tuning: load directly from output_dir.

    Args:
        base_model_name: original model name (for LoRA, needed to load base)
        adapter_dir:     directory where fine-tuned model was saved
        use_lora:        whether the saved model uses LoRA adapters

    Returns:
        model, tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    if use_lora and PEFT_AVAILABLE:
        from peft import PeftModel
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, adapter_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(adapter_dir)

    return model, tokenizer


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("=== LLM Fine-Tuning Pipeline ===")

    # ── Configuration ──────────────────────────────────────────────
    model_cfg = ModelConfig(
        model_name="gpt2",            # Placeholder: swap to LLaMA/Mistral for production
        use_4bit_quantization=False,  # Placeholder: True for QLoRA with large models
    )
    lora_cfg = LoRAConfig(
        enabled=PEFT_AVAILABLE,       # auto-detect PEFT availability
        r=8,
        alpha=16,
        target_modules=["c_attn"],    # GPT-2 specific; change for other models
    )
    data_cfg = DataConfig(
        dataset_name="wikitext",
        dataset_config="wikitext-2-raw-v1",
        max_length=256,               # Placeholder: increase for longer contexts
    )
    train_cfg = TrainConfig(
        num_train_epochs=1,           # Placeholder: 3 for real training
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        output_dir="./gpt2_finetuned",
    )

    # ── Load model ─────────────────────────────────────────────────
    model, tokenizer = load_model_and_tokenizer(model_cfg, lora_cfg)

    # ── Load and tokenize data ──────────────────────────────────────
    train_ds, val_ds = load_and_prepare_dataset(data_cfg, tokenizer)

    # ── Fine-tune ──────────────────────────────────────────────────
    trainer = fine_tune(model, tokenizer, train_ds, val_ds, train_cfg)

    # ── Save ───────────────────────────────────────────────────────
    save_model(model, tokenizer, train_cfg.output_dir)

    # ── Generate samples ───────────────────────────────────────────
    print("\n=== Sample Generation ===")
    test_prompts = [
        "The history of artificial intelligence began",
        "In machine learning, the most important concept is",
        "Deep neural networks work by",
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        generated = generate_text(
            model, tokenizer, prompt,
            max_new_tokens=100,
            temperature=0.8,
        )
        print(f"Generated: {generated}")

    # ── Instruction tuning example ─────────────────────────────────
    print("\n=== Instruction Format Example ===")
    example_instruction = format_instruction_sample(
        instruction="Explain gradient descent in simple terms.",
        response="Gradient descent is an optimization algorithm that...",
        system_prompt="You are a helpful AI assistant that explains ML concepts clearly.",
    )
    print("Formatted instruction sample:")
    print(example_instruction[:500])


if __name__ == "__main__":
    main()
