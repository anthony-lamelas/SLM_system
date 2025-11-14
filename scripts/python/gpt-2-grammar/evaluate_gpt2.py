"""
Evaluate fine-tuned GPT-2 on BEA validation or test set.
Usage: python evaluate_gpt2.py --model_path ./models/gpt2_bea/final_model --test_data hpc_datasets/bea_dev.csv
"""

import argparse
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import json


def generate_correction(model, tokenizer, input_text, max_length=512, device='cuda'):
    """Generate correction for a single input text."""
    # Format prompt
    prompt = f"Correct this text: {input_text}\nCorrected:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the corrected part (after "Corrected:")
    if "Corrected:" in generated_text:
        correction = generated_text.split("Corrected:")[-1].strip()
    else:
        correction = generated_text.strip()
    
    return correction


def evaluate_dataset(model, tokenizer, test_data_path, output_path, device='cuda'):
    """Evaluate model on entire dataset."""
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    
    print(f"Evaluating on {len(df)} examples...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Generate prediction
        prediction = generate_correction(model, tokenizer, input_text, device=device)
        
        predictions.append({
            'input_text': input_text,
            'target_text': target_text,
            'prediction': prediction
        })
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return results_df


def calculate_metrics(results_df):
    """Calculate basic evaluation metrics."""
    # Exact match accuracy
    exact_matches = sum(results_df['prediction'] == results_df['target_text'])
    exact_match_acc = exact_matches / len(results_df) * 100
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Total examples: {len(results_df)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_match_acc:.2f}%")
    print("="*50)
    
    # Save metrics
    metrics = {
        'total_examples': len(results_df),
        'exact_matches': int(exact_matches),
        'exact_match_accuracy': float(exact_match_acc)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 on test data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions CSV (default: auto-generated)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        test_name = args.test_data.split('/')[-1].replace('.csv', '')
        args.output_path = f"results_{test_name}_predictions.csv"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {args.model_path}...")
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    
    # Set pad token
    tokenizer.pad_token = tokenizer.eos_token
    
    # Evaluate
    results_df = evaluate_dataset(
        model, 
        tokenizer, 
        args.test_data, 
        args.output_path,
        device=device
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results_df)
    
    # Save metrics
    metrics_path = args.output_path.replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
    print("\nEvaluation completed!")

if __name__ == "__main__":
    main()
