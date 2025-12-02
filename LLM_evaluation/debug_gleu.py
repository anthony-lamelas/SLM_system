"""Debug script to identify why GLEU scores are 0 for some models."""
import pandas as pd
from nltk.translate.gleu_score import sentence_gleu
from nltk.tokenize import word_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading punkt tokenizer...")
    nltk.download('punkt')

# Load GPT-3.5-turbo results
print("Loading GPT-3.5-turbo predictions...")
df = pd.read_csv('results/grammar_gpt_3.5_turbo_jfleg_test.csv')

print(f"\nDataset shape: {df.shape}")
print(f"Null predictions: {df['prediction'].isnull().sum()}")
print(f"Empty predictions: {(df['prediction'] == '').sum()}")

# Test GLEU calculation on first few examples
print("\n" + "="*80)
print("Testing GLEU calculation on first 5 examples:")
print("="*80)

scores = []
for idx in range(min(5, len(df))):
    row = df.iloc[idx]
    src = row['input_text']
    pred = row['prediction']
    ref = row['target_text']
    
    print(f"\n--- Example {idx+1} ---")
    print(f"Source: {src}")
    print(f"Prediction: {pred}")
    print(f"Reference: {ref}")
    
    try:
        pred_tokens = word_tokenize(pred.lower())
        ref_tokens = word_tokenize(ref.lower())
        
        print(f"Pred tokens: {pred_tokens[:10]}...")
        print(f"Ref tokens: {ref_tokens[:10]}...")
        
        # Calculate GLEU
        score = sentence_gleu([ref_tokens], pred_tokens)
        scores.append(score)
        print(f"✓ GLEU score: {score:.4f}")
        
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        scores.append(0.0)

print("\n" + "="*80)
print(f"Average GLEU (first 5): {sum(scores)/len(scores) if scores else 0:.4f}")
print("="*80)

# Now test on all examples
print("\nCalculating GLEU on full dataset...")
all_scores = []
error_count = 0

for idx, row in df.iterrows():
    try:
        pred_tokens = word_tokenize(row['prediction'].lower())
        ref_tokens = word_tokenize(row['target_text'].lower())
        score = sentence_gleu([ref_tokens], pred_tokens)
        all_scores.append(score)
    except Exception as e:
        if error_count < 3:  # Only print first 3 errors
            print(f"Error at index {idx}: {type(e).__name__}: {e}")
        error_count += 1
        all_scores.append(0.0)

print(f"\nTotal examples: {len(df)}")
print(f"Successful calculations: {len(all_scores) - error_count}")
print(f"Errors: {error_count}")
print(f"Average GLEU: {sum(all_scores)/len(all_scores) * 100:.2f}")
print(f"Non-zero scores: {sum(1 for s in all_scores if s > 0)}")
