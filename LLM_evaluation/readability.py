import argparse
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
import textstat
import nltk
from client import get_openai_client, generate_completion
from prompts import readability_sys_prompt, readability_gen_prompt

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def evaluate_readability(
    model_name,
    test_data_path,
    output_path
):
    
    # Load test data
    df = pd.read_csv(test_data_path)
    print(f"Loaded {len(df)} test examples")
    
    client = get_openai_client(task_type='readability')
    
    print(f"Generating predictions with {model_name}...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Format the user prompt with the input text
        user_prompt = readability_gen_prompt.format(input_text=input_text)
        
        try:
            # Generate simplification
            prediction = generate_completion(
                client=client,
                model_name=model_name,
                system_prompt=readability_sys_prompt,
                user_prompt=user_prompt
            )
        except Exception as e:
            print(f"\nError on example {idx}: {e}")
            prediction = input_text  # Fallback to original text on error
        
        predictions.append({
            'input_text': input_text,
            'target_text': target_text,
            'prediction': prediction
        })
    
    # Create results dataframe
    results_df = pd.DataFrame(predictions)
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return results_df

def calculate_sari(sources, predictions, references):

    from nltk.tokenize import word_tokenize
    
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

    # Exact match accuracy
    exact_matches = sum(results_df['prediction'] == results_df['target_text'])
    exact_match_acc = exact_matches / len(results_df) * 100
    
    input_lengths = results_df['input_text'].str.split().str.len()
    pred_lengths = results_df['prediction'].str.split().str.len()
    target_lengths = results_df['target_text'].str.split().str.len()
    
    avg_input_len = input_lengths.mean()
    avg_pred_len = pred_lengths.mean()
    avg_target_len = target_lengths.mean()
    
    pred_compression = (1 - avg_pred_len / avg_input_len) * 100
    target_compression = (1 - avg_target_len / avg_input_len) * 100
    
    sari = calculate_sari(
        results_df['input_text'].tolist(),
        results_df['prediction'].tolist(),
        results_df['target_text'].tolist()
    )
    
    pred_fre_scores = []
    pred_fkgl_scores = []
    target_fre_scores = []
    target_fkgl_scores = []
    
    for idx, row in results_df.iterrows():
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
    parser = argparse.ArgumentParser(
        description='Evaluate GPT models on ASSET test set for text simplification'
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='gpt-4',
        help='OpenAI model name (e.g., gpt-4, gpt-5, gpt-4-turbo)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        default='../hpc_datasets/asset_test.csv',
        help='Path to ASSET test CSV file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default=None,
        help='Path to save predictions CSV (default: auto-generated)'
    )
    
    args = parser.parse_args()
    
    # Auto-generate output path if not provided
    if args.output_path is None:
        model_slug = args.model_name.replace('-', '_')
        args.output_path = f'results/readability_{model_slug}_asset_test.csv'
    
    # Run evaluation
    results_df = evaluate_readability(
        model_name=args.model_name,
        test_data_path=args.test_data,
        output_path=args.output_path
    )
    
    # Calculate and save metrics
    metrics = calculate_metrics(results_df)
    
    metrics_path = args.output_path.replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")

if __name__ == "__main__":
    main()