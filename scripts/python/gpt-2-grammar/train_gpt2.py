"""
Train GPT-2 on BEA-2019 training set for grammatical error correction.
Usage: python train_gpt2.py --model_size gpt2 --output_dir ./models/gpt2_bea
"""

import argparse
import os
import json
import time
import pandas as pd
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


def load_bea_data(train_path):
    """Load BEA training data and format for GPT-2."""
    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Format as "input: [input_text] output: [target_text]"
    # This helps GPT-2 learn the task structure
    df['text'] = df.apply(
        lambda x: f"Correct this text: {x['input_text']}\nCorrected: {x['target_text']}<|endoftext|>",
        axis=1
    )
    
    print(f"Loaded {len(df)} training examples")
    return df


def prepare_dataset(df, tokenizer, max_length=512):
    """Tokenize the dataset."""
    print("Tokenizing dataset...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding='max_length'
        )
    
    dataset = Dataset.from_pandas(df[['text']])
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )
    
    return tokenized_dataset


def main():
    parser = argparse.ArgumentParser(description='Train GPT-2/GPT-Neo on BEA dataset')
    parser.add_argument('--train_data', type=str, default='hpc_datasets/bea_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--model_name', type=str, default='gpt2',
                        help='Model name or path (e.g., gpt2, gpt2-medium, gpt2-large, gpt2-xl, EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B)')
    parser.add_argument('--output_dir', type=str, default='./models/gpt2_bea',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Training batch size per device (increased for better GPU usage)')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                        help='Gradient accumulation steps (effective batch = batch_size * gradient_accumulation_steps)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--dataloader_num_workers', type=int, default=4,
                        help='Number of data loading workers (increased for faster data loading)')
    parser.add_argument('--use_gradient_checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory (useful for larger models)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model (Auto classes work with GPT-2, GPT-Neo, etc.)
    print(f"Loading {args.model_name} model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    # Enable gradient checkpointing if requested (saves memory for larger models)
    if args.use_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (saves memory)")
    
    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load and prepare data
    df = load_bea_data(args.train_data)
    train_dataset = prepare_dataset(df, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is causal LM, not masked LM
    )
    
    # Determine precision (bf16 is better than fp16 if available)
    use_bf16 = False
    use_fp16 = False
    if torch.cuda.is_available():
        # Check if bf16 is supported (A100, H100, etc.)
        if torch.cuda.is_bf16_supported():
            use_bf16 = True
            print("Using bfloat16 precision (faster and more stable)")
        else:
            use_fp16 = True
            print("Using float16 precision")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=True,  # Faster data transfer to GPU
        report_to='tensorboard',
        load_best_model_at_end=False,  # We save the final model manually
        optim='adamw_torch',  # Use PyTorch's AdamW (faster)
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    start_time = time.time()
    train_result = trainer.train()
    end_time = time.time()
    training_duration = end_time - start_time
    
    # Save final model
    print(f"Saving final model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    # Save training metadata (time, cost info)
    training_metadata = {
        "training_duration_seconds": training_duration,
        "training_duration_hours": training_duration / 3600,
        "training_duration_formatted": f"{int(training_duration // 3600)}h {int((training_duration % 3600) // 60)}m {int(training_duration % 60)}s",
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_examples": len(train_dataset),
        "total_steps": train_result.global_step,
        "final_loss": train_result.training_loss if hasattr(train_result, 'training_loss') else None,
        "gpu_used": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "gpu_memory_allocated_gb": torch.cuda.max_memory_allocated(0) / 1e9 if torch.cuda.is_available() else 0,
    }
    
    metadata_path = os.path.join(args.output_dir, "training_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(training_metadata, f, indent=2)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETED")
    print("="*70)
    print(f"Duration: {training_metadata['training_duration_formatted']}")
    print(f"Total steps: {training_metadata['total_steps']}")
    print(f"GPU: {training_metadata['gpu_used']}")
    if training_metadata['gpu_memory_allocated_gb'] > 0:
        print(f"Max GPU memory: {training_metadata['gpu_memory_allocated_gb']:.2f} GB")
    print(f"Metadata saved to: {metadata_path}")
    print("="*70)


if __name__ == "__main__":
    main()
