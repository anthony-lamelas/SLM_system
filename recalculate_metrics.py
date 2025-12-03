#!/usr/bin/env python3
"""
Recalculate GLEU and SARI metrics from existing CSV result files.
"""

import pandas as pd
import json
import os
from pathlib import Path
import nltk
from nltk.translate.gleu_score import sentence_gleu
from nltk.tokenize import word_tokenize
import logging

# Download required NLTK data
print("Downloading NLTK data...")
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def calculate_sari(source, prediction, references):
    """
    Calculate SARI score (simplified version).
    SARI = (F1_add + F1_keep + P_del) / 3
    """
    try:
        source_tokens = set(word_tokenize(source.lower()))
        pred_tokens = set(word_tokenize(prediction.lower()))
        
        # Handle multiple references
        if isinstance(references, str):
            references = [references]
        ref_tokens_list = [set(word_tokenize(ref.lower())) for ref in references]
        
        # Tokens that should be kept (in source and all references)
        keep_tokens = source_tokens.intersection(*ref_tokens_list)
        
        # Tokens that should be added (not in source but in references)
        add_tokens = set()
        for ref_tokens in ref_tokens_list:
            add_tokens.update(ref_tokens - source_tokens)
        
        # Tokens that should be deleted (in source but not in references)
        del_tokens = source_tokens - set().union(*ref_tokens_list)
        
        # Calculate metrics
        # Keep: F1 score
        keep_pred = pred_tokens.intersection(keep_tokens)
        keep_precision = len(keep_pred) / len(pred_tokens) if pred_tokens else 0
        keep_recall = len(keep_pred) / len(keep_tokens) if keep_tokens else 0
        keep_f1 = 2 * keep_precision * keep_recall / (keep_precision + keep_recall) if (keep_precision + keep_recall) > 0 else 0
        
        # Add: F1 score
        add_pred = pred_tokens.intersection(add_tokens)
        add_precision = len(add_pred) / len(pred_tokens) if pred_tokens else 0
        add_recall = len(add_pred) / len(add_tokens) if add_tokens else 0
        add_f1 = 2 * add_precision * add_recall / (add_precision + add_recall) if (add_precision + add_recall) > 0 else 0
        
        # Delete: Precision (how many tokens from source that should be deleted were actually deleted)
        del_pred = source_tokens - pred_tokens
        del_precision = len(del_pred.intersection(del_tokens)) / len(del_tokens) if del_tokens else 0
        
        sari = (keep_f1 + add_f1 + del_precision) / 3 * 100
        
        return sari
    except Exception as e:
        logger.warning(f"Error calculating SARI: {e}")
        return 0.0


def calculate_gleu(reference, prediction):
    """Calculate GLEU score between reference and prediction."""
    try:
        ref_tokens = word_tokenize(reference.lower())
        pred_tokens = word_tokenize(prediction.lower())
        
        if not pred_tokens:
            return 0.0
        
        score = sentence_gleu([ref_tokens], pred_tokens)
        return score * 100  # Convert to percentage
    except Exception as e:
        logger.warning(f"Error calculating GLEU: {e}")
        return 0.0


def process_csv_file(csv_path, task_type):
    """
    Process a CSV file and recalculate metrics.
    
    Args:
        csv_path: Path to CSV file
        task_type: 'grammar' or 'readability'
    """
    logger.info(f"Processing {csv_path}...")
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Check required columns
    required_cols = ['input_text', 'target_text', 'prediction']
    if not all(col in df.columns for col in required_cols):
        logger.error(f"Missing required columns in {csv_path}")
        return None
    
    # Calculate metrics
    gleu_scores = []
    sari_scores = []
    
    for idx, row in df.iterrows():
        input_text = str(row['input_text'])
        target_text = str(row['target_text'])
        prediction = str(row['prediction'])
        
        # Skip empty predictions
        if not prediction or prediction.strip() == '':
            gleu_scores.append(0.0)
            sari_scores.append(0.0)
            continue
        
        # Calculate GLEU
        gleu = calculate_gleu(target_text, prediction)
        gleu_scores.append(gleu)
        
        # Calculate SARI
        sari = calculate_sari(input_text, prediction, target_text)
        sari_scores.append(sari)
        
        if (idx + 1) % 100 == 0:
            logger.info(f"  Processed {idx + 1}/{len(df)} examples...")
    
    # Calculate averages
    avg_gleu = sum(gleu_scores) / len(gleu_scores) if gleu_scores else 0.0
    avg_sari = sum(sari_scores) / len(sari_scores) if sari_scores else 0.0
    
    results = {
        'file': os.path.basename(csv_path),
        'num_examples': len(df),
        'average_gleu': round(avg_gleu, 2),
        'average_sari': round(avg_sari, 2),
        'task_type': task_type
    }
    
    logger.info(f"  Results: GLEU={avg_gleu:.2f}, SARI={avg_sari:.2f}")
    
    return results


def main():
    backup_dir = Path('/home/dell/Coding/SLM_system/hpc_results_backup')
    
    # Find all test CSV files
    test_csvs = list(backup_dir.glob('*test*.csv'))
    
    logger.info(f"Found {len(test_csvs)} test CSV files")
    
    all_results = []
    
    for csv_path in sorted(test_csvs):
        filename = csv_path.name
        
        # Determine task type
        if 'grammar' in filename:
            task_type = 'grammar'
        elif 'readability' in filename:
            task_type = 'readability'
        else:
            logger.warning(f"Unknown task type for {filename}, skipping")
            continue
        
        # Process file
        result = process_csv_file(csv_path, task_type)
        if result:
            all_results.append(result)
    
    # Save summary
    summary_path = backup_dir / 'recalculated_metrics_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    logger.info(f"\nSummary saved to {summary_path}")
    
    # Print summary table
    print("\n" + "="*80)
    print("RECALCULATED METRICS SUMMARY")
    print("="*80)
    print(f"{'File':<50} {'GLEU':>8} {'SARI':>8}")
    print("-"*80)
    
    for result in all_results:
        print(f"{result['file']:<50} {result['average_gleu']:>8.2f} {result['average_sari']:>8.2f}")
    
    print("="*80)


if __name__ == '__main__':
    main()

