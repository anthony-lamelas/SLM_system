"""
Train GPT-2 on WikiAuto dataset for text simplification/readability.
Usage: python train_gpt2_readability.py --model_size gpt2 --output_dir ./models/gpt2_readability
"""

import argparse
import os
import json
import time
import pandas as pd
import torch
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import Dataset


def load_wikiauto_data(train_path):
    """Load WikiAuto training data and format for GPT-2."""
    print(f"Loading training data from {train_path}...")
    df = pd.read_csv(train_path)
    
    # Format as "Simplify: [input_text] Simplified: [target_text]"
    # This helps GPT-2 learn the simplification task
    df['text'] = df.apply(
        lambda x: f"Simplify this text: {x['input_text']}\nSimplified: {x['target_text']}<|endoftext|>",
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
    parser = argparse.ArgumentParser(description='Train GPT-2 on WikiAuto for text simplification')
    parser.add_argument('--train_data', type=str, default='hpc_datasets/wikiauto_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--model_size', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size')
    parser.add_argument('--output_dir', type=str, default='./models/gpt2_readability',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=1000,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    parser.add_argument('--max_train_samples', type=int, default=None,
                        help='Limit training samples (useful for testing)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    print(f"Loading {args.model_size} model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_size)
    model = GPT2LMHeadModel.from_pretrained(args.model_size)
    
    # Set pad token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id
    
    # Load and prepare data
    df = load_wikiauto_data(args.train_data)
    
    # Optionally limit training samples
    if args.max_train_samples:
        print(f"Limiting to {args.max_train_samples} training samples")
        df = df.head(args.max_train_samples)
    
    train_dataset = prepare_dataset(df, tokenizer, args.max_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # GPT-2 is causal LM, not masked LM
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,  # Effective batch size = 4 * 4 = 16
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=500,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,  # Keep only 3 best checkpoints
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to='tensorboard',
        load_best_model_at_end=False,  # We save the final model manually
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
    print(f"Training on {len(train_dataset)} examples for {args.epochs} epochs")
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
        "model_size": args.model_size,
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
