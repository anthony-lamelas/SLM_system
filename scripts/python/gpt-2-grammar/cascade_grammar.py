"""
Cascading grammar correction: Run multiple models sequentially.
Each model refines the output of the previous model.
Usage: python cascade_grammar.py --models model1_path model2_path --test_data hpc_datasets/jfleg_test.csv
"""

import argparse
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
import os
from nltk.translate.gleu_score import sentence_gleu
from nltk.tokenize import word_tokenize


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


def calculate_gleu(sources, predictions, references):
    """Calculate GLEU score."""
    scores = []
    for src, pred, ref in zip(sources, predictions, references):
        try:
            # Tokenize
            pred_tokens = word_tokenize(pred.lower())
            ref_tokens = word_tokenize(ref.lower())
            
            # Calculate GLEU with reference as a list of lists
            score = sentence_gleu([ref_tokens], pred_tokens)
            scores.append(score)
        except:
            scores.append(0.0)
    
    return sum(scores) / len(scores) if scores else 0.0


def calculate_m2_score(sources, predictions, references, results_df):
    """Calculate M² scores using ERRANT."""
    try:
        import errant
        annotator = errant.load('en')
        
        m2_precision_scores = []
        m2_recall_scores = []
        
        for src, pred, ref in zip(sources, predictions, references):
            try:
                # Parse
                src_parse = annotator.parse(src)
                pred_parse = annotator.parse(pred)
                ref_parse = annotator.parse(ref)
                
                # Get edits
                pred_edits = annotator.annotate(src_parse, pred_parse)
                ref_edits = annotator.annotate(src_parse, ref_parse)
                
                # Calculate precision and recall
                if len(pred_edits) == 0:
                    precision = 1.0 if len(ref_edits) == 0 else 0.0
                    recall = 1.0 if len(ref_edits) == 0 else 0.0
                else:
                    # True positives: edits in both pred and ref
                    tp = len([e for e in pred_edits if e in ref_edits])
                    precision = tp / len(pred_edits) if pred_edits else 0.0
                    recall = tp / len(ref_edits) if ref_edits else 0.0
                
                m2_precision_scores.append(precision)
                m2_recall_scores.append(recall)
            except:
                m2_precision_scores.append(0.0)
                m2_recall_scores.append(0.0)
        
        avg_precision = sum(m2_precision_scores) / len(m2_precision_scores) * 100
        avg_recall = sum(m2_recall_scores) / len(m2_recall_scores) * 100
        f05 = (1 + 0.5**2) * (avg_precision * avg_recall) / (0.5**2 * avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return {
            'precision': avg_precision,
            'recall': avg_recall,
            'f0.5': f05
        }
    except ImportError:
        print("Warning: ERRANT not installed. M² scores will not be calculated.")
        return None


def calculate_metrics(results_df, stage_name="Final"):
    """Calculate evaluation metrics."""
    # Exact match accuracy
    exact_matches = sum(results_df['prediction'] == results_df['target_text'])
    exact_match_acc = exact_matches / len(results_df) * 100
    
    # Calculate GLEU score
    print(f"\nCalculating GLEU scores for {stage_name}...")
    gleu = calculate_gleu(
        results_df['input_text'].tolist(),
        results_df['prediction'].tolist(),
        results_df['target_text'].tolist()
    )
    
    # Calculate M^2 F0.5 score
    m2_scores = calculate_m2_score(
        results_df['input_text'].tolist(),
        results_df['prediction'].tolist(),
        results_df['target_text'].tolist(),
        results_df
    )
    
    print(f"\n{stage_name} Metrics:")
    print(f"  Exact match accuracy: {exact_match_acc:.2f}%")
    print(f"  GLEU Score: {gleu * 100:.2f}")
    if m2_scores:
        print(f"  M² Precision: {m2_scores['precision']:.2f}%")
        print(f"  M² Recall: {m2_scores['recall']:.2f}%")
        print(f"  M² F0.5: {m2_scores['f0.5']:.2f}%")
    
    # Save metrics
    metrics = {
        'stage': stage_name,
        'total_examples': len(results_df),
        'exact_matches': int(exact_matches),
        'exact_match_accuracy': float(exact_match_acc),
        'gleu_score': float(gleu * 100)
    }
    
    if m2_scores:
        metrics.update({
            'm2_precision': float(m2_scores['precision']),
            'm2_recall': float(m2_scores['recall']),
            'm2_f0.5': float(m2_scores['f0.5'])
        })
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Cascade multiple grammar correction models sequentially'
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
        print(f"Generating corrections (using Stage {stage_idx-1} output as input)...")
        predictions = []
        
        for idx, row in tqdm(current_df.iterrows(), total=len(current_df), desc=f"Stage {stage_idx}"):
            # Use previous prediction as input (or original input for first stage)
            input_text = row['prediction'] if stage_idx > 1 else row['input_text']
            
            # Generate correction
            prediction = generate_correction(model, tokenizer, input_text, args.max_length, device)
            predictions.append(prediction)
        
        # Update dataframe
        current_df['prediction'] = predictions
        
        # Save intermediate results
        stage_name = f"stage_{stage_idx}"
        intermediate_path = os.path.join(args.output_dir, f"cascade_grammar_{stage_name}.csv")
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
    final_path = os.path.join(args.output_dir, "cascade_grammar_final.csv")
    current_df[['input_text', 'target_text', 'prediction']].to_csv(final_path, index=False)
    print(f"\nFinal results saved to {final_path}")
    
    # Calculate improvement metrics
    if len(all_metrics) > 1:
        print(f"\n{'='*70}")
        print("IMPROVEMENT ANALYSIS")
        print(f"{'='*70}")
        first_metrics = all_metrics[0]
        final_metrics = all_metrics[-1]
        
        gleu_improvement = final_metrics['gleu_score'] - first_metrics['gleu_score']
        exact_match_improvement = final_metrics['exact_match_accuracy'] - first_metrics['exact_match_accuracy']
        
        print(f"GLEU improvement: {gleu_improvement:+.2f} points")
        print(f"Exact match improvement: {exact_match_improvement:+.2f}%")
        
        if 'm2_f0.5' in final_metrics and 'm2_f0.5' in first_metrics:
            m2_improvement = final_metrics['m2_f0.5'] - first_metrics['m2_f0.5']
            print(f"M² F0.5 improvement: {m2_improvement:+.2f} points")
    
    # Save all metrics
    metrics_path = os.path.join(args.output_dir, "cascade_grammar_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump({
            'all_stages': all_metrics,
            'final_metrics': all_metrics[-1] if all_metrics else None,
            'improvement': {
                'gleu_improvement': gleu_improvement if len(all_metrics) > 1 else 0,
                'exact_match_improvement': exact_match_improvement if len(all_metrics) > 1 else 0,
            } if len(all_metrics) > 1 else None
        }, f, indent=2)
    print(f"\nAll metrics saved to {metrics_path}")
    
    print("\nCascading evaluation completed!")


if __name__ == "__main__":
    main()

