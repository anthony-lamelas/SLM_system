"""
Evaluate fine-tuned GPT-2 on ASSET validation or test set for text simplification.
Usage: python evaluate_gpt2_readability.py --model_path ./models/gpt2_readability/final_model --test_data hpc_datasets/asset_validation.csv
"""

import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import textstat
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def generate_simplification(model, tokenizer, input_text, max_length=512, device='cuda'):
    """Generate simplified text for a single input."""
    # Format prompt - MUST match training prompt exactly
    prompt = f"Simplify this text to make it shorter and easier to read: {input_text}\nSimplified:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    input_length = inputs['input_ids'].shape[1]  # Get actual input length
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Estimate target length (simplified text should be shorter)
    input_word_count = len(input_text.split())
    # Set max_new_tokens based on input length, but cap at reasonable limit
    # For simplification, we want roughly 60-80% of input length
    estimated_target_tokens = min(int(input_word_count * 0.8 * 1.3), 60)  # 1.3 tokens per word, cap at 60
    
    # Generate with proper stopping and repetition control
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=estimated_target_tokens,  # Adaptive based on input length
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.3,  # Higher penalty for simplification (stronger)
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (skip the input prompt)
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract only the simplified part (after "Simplified:")
    if "Simplified:" in generated_text:
        simplification = generated_text.split("Simplified:")[-1].strip()
    else:
        simplification = generated_text.strip()
    
    # Additional safety: if output is way too long, truncate at first sentence or reasonable length
    # This handles cases where model ignores EOS token
    if len(simplification.split()) > input_word_count * 1.5:  # More than 50% longer than input
        # Try to find first sentence boundary
        sentences = simplification.split('. ')
        if len(sentences) > 0:
            simplification = sentences[0] + '.' if not sentences[0].endswith('.') else sentences[0]
        else:
            # Fallback: truncate to reasonable length
            words = simplification.split()
            simplification = ' '.join(words[:int(input_word_count * 1.2)])
    
    return simplification


def evaluate_dataset(model, tokenizer, test_data_path, output_path, device='cuda', batch_size=8):
    """Evaluate model on entire dataset with batch processing."""
    print(f"Loading test data from {test_data_path}...")
    df = pd.read_csv(test_data_path)
    
    print(f"Evaluating on {len(df)} examples with batch size {batch_size}...")
    predictions = []
    
    # Process in batches for faster evaluation
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        batch_inputs = []
        batch_input_lengths = []
        
        # Prepare batch prompts
        for idx, row in batch_df.iterrows():
            input_text = row['input_text']
            prompt = f"Simplify this text to make it shorter and easier to read: {input_text}\nSimplified:"
            inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=512, padding=False)
            batch_inputs.append(inputs['input_ids'].squeeze(0))
            batch_input_lengths.append(inputs['input_ids'].shape[1])
        
        # Pad batch to same length (left padding for decoder-only models)
        max_len = max(len(ids) for ids in batch_inputs)
        padded_inputs = []
        attention_masks = []
        for ids in batch_inputs:
            padding_length = max_len - len(ids)
            # Left padding: pad tokens go before the actual input
            padded = torch.cat([torch.full((padding_length,), tokenizer.pad_token_id, dtype=ids.dtype), ids])
            padded_inputs.append(padded)
            attention_masks.append(torch.cat([torch.zeros(padding_length, dtype=torch.bool), torch.ones(len(ids), dtype=torch.bool)]))
        
        batch_input_ids = torch.stack(padded_inputs).to(device)
        batch_attention_mask = torch.stack(attention_masks).to(device)
        
        # Generate for batch (use greedy decoding for speed)
        with torch.no_grad():
            outputs = model.generate(
                batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=60,  # Fixed reasonable limit
                num_return_sequences=1,
                do_sample=False,  # Greedy decoding is faster
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode batch results
        for j, (output, input_len) in enumerate(zip(outputs, batch_input_lengths)):
            # With left padding, output starts with padding, then input, then generated
            # Skip padding and input to get only generated tokens
            padding_len = max_len - input_len
            generated_tokens = output[padding_len + input_len:]
            generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            # DEBUG: Print first example to see what model is generating
            if i == 0 and j == 0:
                row = batch_df.iloc[j]
                print(f"\nDEBUG - First example:")
                print(f"  Input: {row['input_text'][:100]}...")
                print(f"  Full generated text: {generated_text[:200]}...")
                print(f"  Generated tokens length: {len(generated_tokens)}")
            
            # Extract simplified part
            if "Simplified:" in generated_text:
                simplification = generated_text.split("Simplified:")[-1].strip()
            else:
                simplification = generated_text.strip()
            
            # Safety truncation
            row = batch_df.iloc[j]
            input_text = row['input_text']
            input_word_count = len(input_text.split())
            if len(simplification.split()) > input_word_count * 1.5:
                sentences = simplification.split('. ')
                if len(sentences) > 0:
                    simplification = sentences[0] + '.' if not sentences[0].endswith('.') else sentences[0]
                else:
                    words = simplification.split()
                    simplification = ' '.join(words[:int(input_word_count * 1.2)])
            
            predictions.append({
                'input_text': row['input_text'],
                'target_text': row['target_text'],
                'prediction': simplification
            })
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return results_df


def calculate_sari(sources, predictions, references):
    """Calculate SARI (System output Against References and Input) score."""
    def sari_score(source, prediction, reference):
        """Calculate SARI for a single example."""
        src_tokens = word_tokenize(source.lower())
        pred_tokens = word_tokenize(prediction.lower())
        ref_tokens = word_tokenize(reference.lower())
        
        # Keep
        keep_pred = set(pred_tokens) & set(src_tokens)
        keep_ref = set(ref_tokens) & set(src_tokens)
        keep_score = len(keep_pred & keep_ref) / max(len(keep_ref), 1) if keep_ref else 1.0
        
        # Add
        add_pred = set(pred_tokens) - set(src_tokens)
        add_ref = set(ref_tokens) - set(src_tokens)
        add_score = len(add_pred & add_ref) / max(len(add_ref), 1) if add_ref else 1.0
        
        # Delete
        del_pred = set(src_tokens) - set(pred_tokens)
        del_ref = set(src_tokens) - set(ref_tokens)
        del_score = len(del_pred & del_ref) / max(len(del_ref), 1) if del_ref else 1.0
        
        # Average of the three scores
        return (keep_score + add_score + del_score) / 3 * 100
    
    scores = []
    for src, pred, ref in zip(sources, predictions, references):
        try:
            score = sari_score(src, pred, ref)
            scores.append(score)
        except:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def calculate_metrics(results_df):
    """Calculate basic evaluation metrics."""
    # Exact match accuracy
    exact_matches = sum(results_df['prediction'] == results_df['target_text'])
    exact_match_acc = exact_matches / len(results_df) * 100
    
    # Calculate average length reduction
    input_lengths = results_df['input_text'].str.split().str.len()
    pred_lengths = results_df['prediction'].str.split().str.len()
    target_lengths = results_df['target_text'].str.split().str.len()
    
    avg_input_len = input_lengths.mean()
    avg_pred_len = pred_lengths.mean()
    avg_target_len = target_lengths.mean()
    
    pred_compression = (1 - avg_pred_len / avg_input_len) * 100
    target_compression = (1 - avg_target_len / avg_input_len) * 100
    
    # Calculate SARI score
    print("\nCalculating SARI scores...")
    sari = calculate_sari(
        results_df['input_text'].tolist(),
        results_df['prediction'].tolist(),
        results_df['target_text'].tolist()
    )
    
    # Calculate Flesch Reading Ease and Flesch-Kincaid Grade Level
    pred_fre_scores = []
    pred_fkgl_scores = []
    target_fre_scores = []
    target_fkgl_scores = []
    
    print("Calculating readability metrics...")
    for idx, row in tqdm(results_df.iterrows(), total=len(results_df), desc="Readability"):
        # Prediction metrics
        try:
            pred_fre = textstat.flesch_reading_ease(row['prediction'])
            pred_fkgl = textstat.flesch_kincaid_grade(row['prediction'])
            pred_fre_scores.append(pred_fre)
            pred_fkgl_scores.append(pred_fkgl)
        except:
            pred_fre_scores.append(0)
            pred_fkgl_scores.append(0)
        
        # Target metrics
        try:
            target_fre = textstat.flesch_reading_ease(row['target_text'])
            target_fkgl = textstat.flesch_kincaid_grade(row['target_text'])
            target_fre_scores.append(target_fre)
            target_fkgl_scores.append(target_fkgl)
        except:
            target_fre_scores.append(0)
            target_fkgl_scores.append(0)
    
    avg_pred_fre = sum(pred_fre_scores) / len(pred_fre_scores)
    avg_pred_fkgl = sum(pred_fkgl_scores) / len(pred_fkgl_scores)
    avg_target_fre = sum(target_fre_scores) / len(target_fre_scores)
    avg_target_fkgl = sum(target_fkgl_scores) / len(target_fkgl_scores)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Total examples: {len(results_df)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_match_acc:.2f}%")
    print(f"\nLength Statistics (in words):")
    print(f"  Average input length: {avg_input_len:.1f}")
    print(f"  Average prediction length: {avg_pred_len:.1f}")
    print(f"  Average target length: {avg_target_len:.1f}")
    print(f"\nCompression Rates:")
    print(f"  Prediction compression: {pred_compression:.1f}%")
    print(f"  Target compression: {target_compression:.1f}%")
    print(f"\nSARI Score: {sari:.2f}")
    print(f"\nReadability Metrics:")
    print(f"  Prediction FRE (Flesch Reading Ease): {avg_pred_fre:.2f}")
    print(f"  Target FRE: {avg_target_fre:.2f}")
    print(f"  Prediction FKGL (Flesch-Kincaid Grade Level): {avg_pred_fkgl:.2f}")
    print(f"  Target FKGL: {avg_target_fkgl:.2f}")
    print("="*50)
    
    # Save metrics
    metrics = {
        'total_examples': len(results_df),
        'exact_matches': int(exact_matches),
        'exact_match_accuracy': float(exact_match_acc),
        'avg_input_length': float(avg_input_len),
        'avg_prediction_length': float(avg_pred_len),
        'avg_target_length': float(avg_target_len),
        'prediction_compression_rate': float(pred_compression),
        'target_compression_rate': float(target_compression),
        'sari_score': float(sari),
        'prediction_flesch_reading_ease': float(avg_pred_fre),
        'target_flesch_reading_ease': float(avg_target_fre),
        'prediction_flesch_kincaid_grade_level': float(avg_pred_fkgl),
        'target_flesch_kincaid_grade_level': float(avg_target_fkgl)
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 on ASSET test data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file (ASSET validation or test)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save predictions CSV (default: auto-generated)')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for evaluation (larger = faster but more memory)')
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        test_name = args.test_data.split('/')[-1].replace('.csv', '')
        args.output_path = f"results_{test_name}_predictions.csv"
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Check if model path exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model path does not exist: {args.model_path}")
    
    # Load model and tokenizer (Auto classes work with GPT-2, GPT-Neo, etc.)
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, local_files_only=True)
    model.to(device)
    model.eval()
    
    # Set pad token and padding side (left for decoder-only models)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'  # Important for decoder-only models
    
    # Evaluate
    results_df = evaluate_dataset(
        model, 
        tokenizer, 
        args.test_data, 
        args.output_path,
        device=device,
        batch_size=args.batch_size
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
