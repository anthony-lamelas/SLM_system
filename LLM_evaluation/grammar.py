import argparse
import pandas as pd
import json
from tqdm import tqdm
from pathlib import Path
from nltk.translate.gleu_score import sentence_gleu

from client import get_openai_client, generate_completion
from prompts import grammar_sys_prompt, grammar_gen_prompt


def evaluate_grammar(
    model_name,
    test_data_path,
    output_path
):

    print(f"Evaluating {model_name} on grammar correction task")    
    # Load test data
    df = pd.read_csv(test_data_path)
    print(f"Loaded {len(df)} test examples")
    
    # Initialize OpenAI client
    print("Initializing OpenAI client...")
    client = get_openai_client(task_type='grammar')
    
    # Generate predictions
    print(f"Generating predictions with {model_name}...")
    predictions = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        input_text = row['input_text']
        target_text = row['target_text']
        
        # Format the user prompt with the input text
        user_prompt = grammar_gen_prompt.format(input_text=input_text)
        
        try:
            # Generate correction
            prediction = generate_completion(
                client=client,
                model_name=model_name,
                system_prompt=grammar_sys_prompt,
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
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    
    return results_df


def calculate_gleu(sources, predictions, references):
    from nltk.tokenize import word_tokenize
    
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
    try:
        import errant
        
        # Initialize ERRANT annotator
        annotator = errant.load('en')
        
        total_tp = 0  
        total_fp = 0  
        total_fn = 0  
        
        print("Calculating M^2 scores with ERRANT (optimized with batching)...")
        
        # OPTIMIZATION: Batch process with spaCy's pipe for 3-5x speedup
        # Prepare all texts for batch processing
        all_sources = list(sources)
        all_predictions = list(predictions)
        all_references = list(references)
        
        # Parse all texts in batches (much faster than one-by-one)
        print("  Parsing sources...")
        src_docs = list(annotator.nlp.pipe(all_sources, batch_size=32, n_process=1))
        
        print("  Parsing predictions...")
        pred_docs = list(annotator.nlp.pipe(all_predictions, batch_size=32, n_process=1))
        
        print("  Parsing references...")
        ref_docs = list(annotator.nlp.pipe(all_references, batch_size=32, n_process=1))
        
        print("  Calculating edit alignments...")
        
        # Now process edits with batch-parsed documents
        for src_doc, pred_doc, ref_doc in tqdm(
            zip(src_docs, pred_docs, ref_docs), 
            total=len(src_docs), 
            desc="M^2 Scoring"
        ):
            try:
                # Annotate source 
                pred_edits = annotator.annotate(src_doc, pred_doc)
                
                # Annotate source 
                ref_edits = annotator.annotate(src_doc, ref_doc)
                
                # Convert edits to sets for comparison
                pred_set = set([(e.o_start, e.o_end, e.c_str, e.type) for e in pred_edits])
                ref_set = set([(e.o_start, e.o_end, e.c_str, e.type) for e in ref_edits])
                
                # Calculate TP, FP, FN
                tp = len(pred_set & ref_set)
                fp = len(pred_set - ref_set)
                fn = len(ref_set - pred_set)
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
            except Exception as e:
                continue
        
        # Calculate precision, recall, and F0.5
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        
        # F0.5 = (1 + 0.5^2) * (precision * recall) / (0.5^2 * precision + recall)
        beta = 0.5
        f_beta = ((1 + beta**2) * precision * recall) / (beta**2 * precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            'precision': precision * 100,
            'recall': recall * 100,
            'f0.5': f_beta * 100
        }
    
    except ImportError:
        print("\nWarning: ERRANT not available. Skipping M^2 score calculation.")
        print("Install with: pip install errant")
        print("Then download spaCy model: python -m spacy download en_core_web_sm")
        return None
    except Exception as e:
        print(f"\nError calculating M^2 scores: {e}")
        return None


def calculate_metrics(results_df):
    # Exact match accuracy
    exact_matches = sum(results_df['prediction'] == results_df['target_text'])
    exact_match_acc = exact_matches / len(results_df) * 100
    
    # Calculate GLEU score
    print("\nCalculating GLEU scores...")
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
    
    print(f"Total examples: {len(results_df)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_match_acc:.2f}%")
    print(f"\nGLEU Score: {gleu * 100:.2f}")
    
    if m2_scores:
        print(f"\nM^2 Scores:")
        print(f"  Precision: {m2_scores['precision']:.2f}%")
        print(f"  Recall: {m2_scores['recall']:.2f}%")
        print(f"  F0.5: {m2_scores['f0.5']:.2f}%")
        
    metrics = {
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
        description='Evaluate GPT models on JFLEG test set for grammar correction'
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
        default='../hpc_datasets/jfleg_test.csv',
        help='Path to JFLEG test CSV file'
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
        args.output_path = f'results/grammar_{model_slug}_jfleg_test.csv'
    
    # Run evaluation
    results_df = evaluate_grammar(
        model_name=args.model_name,
        test_data_path=args.test_data,
        output_path=args.output_path
    )
    
    metrics = calculate_metrics(results_df)
    
    metrics_path = args.output_path.replace('.csv', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")
    
if __name__ == "__main__":
    main()
