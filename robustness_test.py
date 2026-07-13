import pandas as pd
import numpy as np
import joblib
import re
import random
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import accuracy_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

print("=" * 60)
print("ROBUSTNESS TESTING: Perturbed AI Text Analysis")
print("=" * 60)

# Load data and models
test_df = pd.read_csv('data/processed/test.csv')
preprocessing = joblib.load('models/preprocessing.pkl')
all_models = joblib.load('models/all_models.pkl')

tfidf = preprocessing['word_tfidf']
char_tfidf = preprocessing['char_tfidf']
scaler = preprocessing['scaler']

# Select only AI essays for robustness analysis
ai_essays = test_df[test_df['label'] == 1]['text'].tolist()
print(f"Total AI essays in test set: {len(ai_essays)}")

def inject_typos(text, rate=0.05):
    """Simulate human typos/errors by swapping/deleting random characters."""
    chars = list(text)
    n_changes = int(len(chars) * rate)
    for _ in range(n_changes):
        idx = random.randint(0, len(chars) - 1)
        if chars[idx].isalpha():
            # either delete or change case
            if random.random() < 0.5:
                chars[idx] = ''
            else:
                chars[idx] = chars[idx].swapcase()
    return "".join(chars)

def extract_text_features(text):
    text = str(text)
    words = text.split()
    unique_words = set(w.lower() for w in words)
    import textstat
    return {
        'text_length': len(text),
        'word_count': len(words),
        'sentence_count': max(len(text.split('.')), 1),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'unique_words': len(unique_words),
        'type_token_ratio': len(unique_words) / len(words) if words else 0,
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'punctuation_count': len(re.findall(r'[^\w\s]', text)),
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'digit_count': sum(1 for c in text if c.isdigit()),
        'avg_sentence_length': len(words) / max(len(text.split('.')), 1),
    }

def evaluate_robustness(model, X_text, scaler, y_true):
    # Extract text stats
    stats = pd.DataFrame([extract_text_features(t) for t in X_text])
    stats_scaled = scaler.transform(stats)
    stats_sparse = csr_matrix(stats_scaled)
    
    # Extract text features
    X_word = tfidf.transform(X_text)
    X_char = char_tfidf.transform(X_text)
    X_comb = hstack([X_word, X_char, stats_sparse])
    
    # Naive bayes requires dense
    if hasattr(model, 'classes_') and type(model).__name__ == 'MultinomialNB':
        X_comb = X_comb.toarray()
        
    y_pred = model.predict(X_comb)
    # Since we test ONLY on AI texts, recall is equivalent to accuracy
    recall = recall_score(y_true, y_pred)
    return recall

rates = [0.0, 0.05, 0.10, 0.20]  # Perturbation rates (0% to 20% human edit simulation)
results = []

y_true_ai = [1] * len(ai_essays)

for name, model in all_models.items():
    print(f"\nEvaluating {name}...")
    for rate in rates:
        if rate == 0.0:
            perturbed_texts = ai_essays
        else:
            perturbed_texts = [inject_typos(t, rate=rate) for t in ai_essays]
            
        recall = evaluate_robustness(model, perturbed_texts, scaler, y_true_ai)
        print(f"  Perturbation Rate {rate*100:.0f}% -> Recall: {recall:.4f}")
        results.append({
            'Model': name,
            'Perturbation Rate': f"{rate*100:.0f}%",
            'Recall (AI Detection)': recall
        })

robustness_df = pd.DataFrame(results)
robustness_df.to_csv('reports/robustness_comparison.csv', index=False)
print("\nSaved: reports/robustness_comparison.csv")
print(robustness_df.to_string(index=False))
