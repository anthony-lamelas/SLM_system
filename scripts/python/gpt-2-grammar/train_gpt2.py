"""
Train GPT-2 on BEA-2019 training set for grammatical error correction.
Usage: python train_gpt2.py --model_size gpt2 --output_dir ./models/gpt2_bea
"""

import argparse
import os
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
    parser = argparse.ArgumentParser(description='Train GPT-2 on BEA dataset')
    parser.add_argument('--train_data', type=str, default='hpc_datasets/bea_train.csv',
                        help='Path to training CSV file')
    parser.add_argument('--model_size', type=str, default='gpt2',
                        choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'],
                        help='GPT-2 model size')
    parser.add_argument('--output_dir', type=str, default='./models/gpt2_bea',
                        help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--save_steps', type=int, default=500,
                        help='Save checkpoint every N steps')
    parser.add_argument('--logging_steps', type=int, default=100,
                        help='Log every N steps')
    
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
    df = load_bea_data(args.train_data)
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
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        greater_is_better=False,
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
    trainer.train()
    
    # Save final model
    print(f"Saving final model to {args.output_dir}/final_model")
    trainer.save_model(f"{args.output_dir}/final_model")
    tokenizer.save_pretrained(f"{args.output_dir}/final_model")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
