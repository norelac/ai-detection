import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

print("Loading models and data...")
model = joblib.load('models/best_model.pkl')
preprocessing = joblib.load('models/preprocessing.pkl')

tfidf = preprocessing['word_tfidf']
char_tfidf = preprocessing['char_tfidf']
scaler = preprocessing['scaler']

test_df = pd.read_csv('data/processed/test.csv')
X_test_text = test_df['text']
y_test = test_df['label']

# Text stats
import textstat
import re
def extract_text_features(text):
    text = str(text)
    words = text.split()
    unique_words = set(w.lower() for w in words)
    return {
        'text_length': len(text), 'word_count': len(words), 'sentence_count': max(len(text.split('.')), 1),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0, 'unique_words': len(unique_words),
        'type_token_ratio': len(unique_words) / len(words) if words else 0,
        'flesch_reading_ease': textstat.flesch_reading_ease(text), 'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'punctuation_count': len(re.findall(r'[^\w\s]', text)), 'uppercase_count': sum(1 for c in text if c.isupper()),
        'digit_count': sum(1 for c in text if c.isdigit()),
        'avg_sentence_length': len(words) / max(len(text.split('.')), 1),
    }

print("Transforming test data...")
X_test_word = tfidf.transform(X_test_text.astype(str))
X_test_char = char_tfidf.transform(X_test_text.astype(str))
X_test_stats = scaler.transform(pd.DataFrame([extract_text_features(t) for t in X_test_text]))
X_test_combined = hstack([X_test_word, X_test_char, csr_matrix(X_test_stats)])

# We will use a subsample for SHAP to save memory and time
sample_idx = np.random.choice(len(X_test_text), 1000, replace=False)
X_shap = X_test_combined[sample_idx].toarray()

# Get feature names
word_features = list(tfidf.get_feature_names_out())
char_features = list(char_tfidf.get_feature_names_out())
stat_features = list(preprocessing['stats_columns'])
all_features = word_features + char_features + stat_features

print(f"Computing SHAP values for {len(sample_idx)} samples...")
# Use LinearExplainer for Logistic Regression
explainer = shap.LinearExplainer(model, X_shap)
shap_values = explainer.shap_values(X_shap)

print("Creating SHAP summary plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_shap, feature_names=all_features, max_display=20, show=False)
plt.tight_layout()
plt.savefig('app/assets/shap_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: app/assets/shap_summary.png")

print("Computing feature importance...")
mean_shap = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({'feature': all_features[:len(mean_shap)], 'importance': mean_shap})
feature_importance = feature_importance.sort_values('importance', ascending=False)

print("Creating feature importance plot...")
plt.figure(figsize=(10, 8))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature', palette='viridis')
plt.title('Top 20 Features by SHAP Importance (Logistic Regression)')
plt.xlabel('Mean |SHAP Value|')
plt.tight_layout()
plt.savefig('app/assets/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: app/assets/feature_importance.png")

print('Saving feature importance to CSV...')
feature_importance.to_csv('reports/feature_importance.csv', index=False)

print("SHAP Analysis Complete!")
