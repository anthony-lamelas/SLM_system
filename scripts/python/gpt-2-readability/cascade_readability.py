"""
Cascading readability improvement: Run multiple models sequentially.
Each model refines the output of the previous model.
Usage: python cascade_readability.py --models model1_path model2_path --test_data hpc_datasets/asset_test.csv
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
import textstat
import nltk
from nltk.tokenize import word_tokenize

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


def generate_simplification(model, tokenizer, input_text, max_length=512, device='cuda'):
    """Generate simplified text for a single input."""
    # Format prompt
    prompt = f"Simplify this text: {input_text}\nSimplified:"
    
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
    
    # Extract only the simplified part (after "Simplified:")
    if "Simplified:" in generated_text:
        simplification = generated_text.split("Simplified:")[-1].strip()
    else:
        simplification = generated_text.strip()
    
    return simplification


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
        
        # Average
        return (keep_score + add_score + del_score) / 3.0
    
    scores = [sari_score(src, pred, ref) for src, pred, ref in zip(sources, predictions, references)]
    return sum(scores) / len(scores) if scores else 0.0


def calculate_metrics(results_df, stage_name="Final"):
    """Calculate evaluation metrics."""
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
    print(f"\nCalculating SARI scores for {stage_name}...")
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
    
    print(f"Calculating readability metrics for {stage_name}...")
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
    
    print(f"\n{stage_name} Metrics:")
    print(f"  Exact match accuracy: {exact_match_acc:.2f}%")
    print(f"  SARI Score: {sari:.2f}")
    print(f"  Average prediction length: {avg_pred_len:.1f} words")
    print(f"  Prediction compression: {pred_compression:.1f}%")
    print(f"  Prediction FRE: {avg_pred_fre:.2f}")
    print(f"  Prediction FKGL: {avg_pred_fkgl:.2f}")
    
    # Save metrics
    metrics = {
        'stage': stage_name,
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
    parser = argparse.ArgumentParser(
        description='Cascade multiple readability models sequentially'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        required=True,
        help='Paths to fine-tuned models (in order: model1, model2, ...)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test CSV file'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='Directory to save results'
    )
    parser.add_argument(
        '--max_length',
        type=int,
        default=512,
        help='Maximum sequence length'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Number of models: {len(args.models)}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    df = pd.read_csv(args.test_data)
    print(f"Loaded {len(df)} examples")
    
    # Initialize results dataframe
    current_df = df.copy()
    current_df['prediction'] = current_df['input_text']  # Start with original input
    
    all_metrics = []
    
    # Run models sequentially
    for stage_idx, model_path in enumerate(args.models, 1):
        print(f"\n{'='*70}")
        print(f"Stage {stage_idx}/{len(args.models)}: Processing with {model_path}")
        print(f"{'='*70}")
        
        # Load model
        print(f"Loading model from {model_path}...")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer.pad_token = tokenizer.eos_token
        
        # Generate predictions using previous stage's output as input
        print(f"Generating simplifications (using Stage {stage_idx-1} output as input)...")
        predictions = []
        
        for idx, row in tqdm(current_df.iterrows(), total=len(current_df), desc=f"Stage {stage_idx}"):
            # Use previous prediction as input (or original input for first stage)
            input_text = row['prediction'] if stage_idx > 1 else row['input_text']
            
            # Generate simplification
            prediction = generate_simplification(model, tokenizer, input_text, args.max_length, device)
            predictions.append(prediction)
        
        # Update dataframe
        current_df['prediction'] = predictions
        
        # Save intermediate results
        stage_name = f"stage_{stage_idx}"
        intermediate_path = os.path.join(args.output_dir, f"cascade_readability_{stage_name}.csv")
        current_df[['input_text', 'target_text', 'prediction']].to_csv(intermediate_path, index=False)
        print(f"Saved intermediate results to {intermediate_path}")
        
        # Calculate metrics for this stage
        stage_metrics = calculate_metrics(current_df, f"Stage {stage_idx}")
        stage_metrics['model_path'] = model_path
        all_metrics.append(stage_metrics)
        
        # Clear model from memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Save final results
    final_path = os.path.join(args.output_dir, "cascade_readability_final.csv")
    current_df[['input_text', 'target_text', 'prediction']].to_csv(final_path, index=False)
    print(f"\nFinal results saved to {final_path}")
    
    # Calculate improvement metrics
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*70}")
        first_metrics = all_metrics[0]
        final_metrics = all_metrics[-1]
        
        sari_improvement = final_metrics['sari_score'] - first_metrics['sari_score']
        exact_match_improvement = final_metrics['exact_match_accuracy'] - first_metrics['exact_match_accuracy']
        fre_improvement = final_metrics['prediction_flesch_reading_ease'] - first_metrics['prediction_flesch_reading_ease']
        
        print(f"SARI improvement: {sari_improvement:+.2f} points")
        print(f"Exact match improvement: {exact_match_improvement:+.2f}%")
        print(f"Flesch Reading Ease improvement: {fre_improvement:+.2f} points")
    
    # Save all metrics
    metrics_path = os.path.join(args.output_dir, "cascade_readability_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'all_stages': all_metrics,
            'final_metrics': all_metrics[-1] if all_metrics else None,
            'improvement': {
                'sari_improvement': sari_improvement if len(all_metrics) > 1 else 0,
                'exact_match_improvement': exact_match_improvement if len(all_metrics) > 1 else 0,
                'fre_improvement': fre_improvement if len(all_metrics) > 1 else 0,
            } if len(all_metrics) > 1 else None
        }, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")
    
    print("\nCascading evaluation completed!")


if __name__ == "__main__":
    main()

