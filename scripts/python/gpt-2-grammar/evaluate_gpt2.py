"""
Evaluate fine-tuned GPT-2 on BEA validation or test set.
Usage: python evaluate_gpt2.py --model_path ./models/gpt2_bea/final_model --test_data hpc_datasets/bea_dev.csv
"""

import argparse
import os
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import json
from nltk.translate.gleu_score import sentence_gleu
from nltk.tokenize import word_tokenize


def generate_correction(model, tokenizer, input_text, max_length=512, device='cuda'):
    """Generate correction for a single input text."""
    # Format prompt - MUST match training prompt exactly
    prompt = f"Correct this text: {input_text}\nCorrected:"
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=max_length)
    input_length = inputs['input_ids'].shape[1]  # Get actual input length
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate with proper stopping and repetition control
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,  # Limit new tokens, not total length
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,  # Penalize repetition
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens (skip the input prompt)
    generated_tokens = outputs[0][input_length:]
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    # Extract only the corrected part (after "Corrected:")
    if "Corrected:" in generated_text:
        correction = generated_text.split("Corrected:")[-1].strip()
    else:
        correction = generated_text.strip()
    
    return correction


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
            prompt = f"Correct this text: {input_text}\nCorrected:"
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
                max_new_tokens=128,  # Fixed reasonable limit
                num_return_sequences=1,
                do_sample=False,  # Greedy decoding is faster
                repetition_penalty=1.2,
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
            
            # Extract corrected part
            if "Corrected:" in generated_text:
                correction = generated_text.split("Corrected:")[-1].strip()
            else:
                correction = generated_text.strip()
            
            row = batch_df.iloc[j]
            predictions.append({
                'input_text': row['input_text'],
                'target_text': row['target_text'],
                'prediction': correction
            })
    
    # Save results
    results_df = pd.DataFrame(predictions)
    results_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    return results_df


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
        
        # Initialize ERRANT annotator
        annotator = errant.load('en')
        
        total_tp = 0  
        total_fp = 0  
        total_fn = 0  
        
        print("Calculating M^2 scores with ERRANT...")
        
        # Prepare all texts for batch processing and filter out invalid entries
        all_sources = []
        all_predictions = []
        all_references = []
        
        for src, pred, ref in zip(sources, predictions, references):
            # Skip entries with NaN or non-string values
            if isinstance(src, str) and isinstance(pred, str) and isinstance(ref, str):
                all_sources.append(src)
                all_predictions.append(pred)
                all_references.append(ref)
        
        print(f"  Processing {len(all_sources)} valid examples (skipped {len(list(sources)) - len(all_sources)} invalid entries)")
        
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
    """Calculate evaluation metrics including GLEU and M² scores."""
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
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"Total examples: {len(results_df)}")
    print(f"Exact matches: {exact_matches}")
    print(f"Exact match accuracy: {exact_match_acc:.2f}%")
    print(f"\nGLEU Score: {gleu * 100:.2f}")
    
    if m2_scores:
        print(f"\nM^2 Scores:")
        print(f"  Precision: {m2_scores['precision']:.2f}%")
        print(f"  Recall: {m2_scores['recall']:.2f}%")
        print(f"  F0.5: {m2_scores['f0.5']:.2f}%")
    print("="*50)
    
    # Save metrics
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
    parser = argparse.ArgumentParser(description='Evaluate GPT-2 on test data')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to fine-tuned model')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Path to test CSV file')
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
