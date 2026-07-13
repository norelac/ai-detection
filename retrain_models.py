import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
import xgboost as xgb
from scipy.sparse import hstack, csr_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import textstat
import re
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/val.csv')
test_df = pd.read_csv('data/processed/test.csv')

X_train_text, y_train = train_df['text'], train_df['label']
X_val_text, y_val = val_df['text'], val_df['label']
X_test_text, y_test = test_df['text'], test_df['label']

print(f"Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")


print("\n" + "=" * 60)
print("STEP 2: Extracting Text Statistics Features")
print("=" * 60)

def extract_text_features(text):
    text = str(text)
    words = text.split()
    unique_words = set(w.lower() for w in words)
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

print("Extracting text statistics for train set...")
train_stats = pd.DataFrame([extract_text_features(t) for t in X_train_text])
print(f"Extracting text statistics for val set...")
val_stats = pd.DataFrame([extract_text_features(t) for t in X_val_text])
print(f"Extracting text statistics for test set...")
test_stats = pd.DataFrame([extract_text_features(t) for t in X_test_text])

print(f"Text statistics features: {list(train_stats.columns)}")

# Normalize stats features
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_stats_scaled = scaler.fit_transform(train_stats)
val_stats_scaled = scaler.transform(val_stats)
test_stats_scaled = scaler.transform(test_stats)

train_stats_sparse = csr_matrix(train_stats_scaled)
val_stats_sparse = csr_matrix(val_stats_scaled)
test_stats_sparse = csr_matrix(test_stats_scaled)


print("\n" + "=" * 60)
print("STEP 3: TF-IDF + Char N-gram Features")
print("=" * 60)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=5, max_df=0.9, stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train_text.astype(str))
X_val_tfidf = tfidf.transform(X_val_text.astype(str))
X_test_tfidf = tfidf.transform(X_test_text.astype(str))
print(f"TF-IDF features: {X_train_tfidf.shape[1]}")

char_tfidf = TfidfVectorizer(max_features=3000, ngram_range=(2, 4), min_df=5, max_df=0.9, analyzer='char')
X_train_char = char_tfidf.fit_transform(X_train_text.astype(str))
X_val_char = char_tfidf.transform(X_val_text.astype(str))
X_test_char = char_tfidf.transform(X_test_text.astype(str))
print(f"Char n-gram features: {X_train_char.shape[1]}")

print(f"Text stats features: {train_stats_sparse.shape[1]}")


print("\n" + "=" * 60)
print("STEP 4: Combining All Features")
print("=" * 60)

X_train_combined = hstack([X_train_tfidf, X_train_char, train_stats_sparse])
X_val_combined = hstack([X_val_tfidf, X_val_char, val_stats_sparse])
X_test_combined = hstack([X_test_tfidf, X_test_char, test_stats_sparse])

print(f"Total features: {X_train_combined.shape[1]}")
print(f"  - TF-IDF: {X_train_tfidf.shape[1]}")
print(f"  - Char n-gram: {X_train_char.shape[1]}")
print(f"  - Text stats: {train_stats_sparse.shape[1]}")


print("\n" + "=" * 60)
print("STEP 5: Training All Models")
print("=" * 60)

models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42, n_jobs=-1),
    'Naive Bayes': MultinomialNB(alpha=0.1),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1, eval_metric='logloss')
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    if name == 'Naive Bayes':
        X_tr = X_train_combined.toarray()
        X_vl = X_val_combined.toarray()
        X_ts = X_test_combined.toarray()
    else:
        X_tr, X_vl, X_ts = X_train_combined, X_val_combined, X_test_combined

    model.fit(X_tr, y_train)

    y_pred_val = model.predict(X_vl)
    y_prob_val = model.predict_proba(X_vl)[:, 1]

    y_pred_test = model.predict(X_ts)
    y_prob_test = model.predict_proba(X_ts)[:, 1]

    val_metrics = {
        'accuracy': accuracy_score(y_val, y_pred_val),
        'precision': precision_score(y_val, y_pred_val),
        'recall': recall_score(y_val, y_pred_val),
        'f1_score': f1_score(y_val, y_pred_val),
        'roc_auc': roc_auc_score(y_val, y_prob_val)
    }

    test_metrics = {
        'accuracy': accuracy_score(y_test, y_pred_test),
        'precision': precision_score(y_test, y_pred_test),
        'recall': recall_score(y_test, y_pred_test),
        'f1_score': f1_score(y_test, y_pred_test),
        'roc_auc': roc_auc_score(y_test, y_prob_test)
    }

    results[name] = {
        'model': model,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'y_pred_test': y_pred_test,
        'y_prob_test': y_prob_test
    }

    print(f"  Val  - F1: {val_metrics['f1_score']:.4f}, ROC-AUC: {val_metrics['roc_auc']:.4f}")
    print(f"  Test - F1: {test_metrics['f1_score']:.4f}, ROC-AUC: {test_metrics['roc_auc']:.4f}")


print("\n" + "=" * 60)
print("STEP 6: Selecting Best Model")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]['test_metrics']['f1_score'])
best_model = results[best_name]['model']
print(f"Best model: {best_name} (F1: {results[best_name]['test_metrics']['f1_score']:.4f})")


print("\n" + "=" * 60)
print("STEP 7: Generating Evaluation Plots")
print("=" * 60)

# Model comparison table
comparison_data = []
for name, res in results.items():
    m = res['test_metrics']
    comparison_data.append({
        'Model': name,
        'Accuracy': m['accuracy'],
        'Precision': m['precision'],
        'Recall': m['recall'],
        'F1-Score': m['f1_score'],
        'ROC-AUC': m['roc_auc']
    })

comparison_df = pd.DataFrame(comparison_data)
comparison_df = comparison_df.sort_values('F1-Score', ascending=False)
comparison_df.to_csv('reports/model_comparison.csv', index=False)
print("Saved: reports/model_comparison.csv")
print(comparison_df.to_string(index=False))

# Confusion Matrix (best model)
cm = confusion_matrix(y_test, results[best_name]['y_pred_test'])
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix - {best_name}')
plt.tight_layout()
plt.savefig('app/assets/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: app/assets/confusion_matrix.png")

# ROC Curve (all models)
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#2563eb', '#16a34a', '#ea580c', '#8b5cf6']
for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_prob_test'])
    auc = res['test_metrics']['roc_auc']
    ax.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', color=color, linewidth=2)

ax.plot([0, 1], [0, 1], 'k--', label='Random', alpha=0.5)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve Comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('app/assets/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: app/assets/roc_curve.png")

# Feature importance (text stats correlation heatmap)
feature_names = list(train_stats.columns)
corr_matrix = train_stats.corr()
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax)
ax.set_title('Text Statistics Features Correlation')
plt.tight_layout()
plt.savefig('app/assets/text_stats_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: app/assets/text_stats_correlation.png")


print("\n" + "=" * 60)
print("STEP 8: Saving Models")
print("=" * 60)

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump({
    'word_tfidf': tfidf,
    'char_tfidf': char_tfidf,
    'scaler': scaler,
    'stats_columns': list(train_stats.columns)
}, 'models/preprocessing.pkl')

# Save all models for the app
joblib.dump({name: res['model'] for name, res in results.items()}, 'models/all_models.pkl')

print("Saved: models/best_model.pkl")
print("Saved: models/preprocessing.pkl")
print("Saved: models/all_models.pkl")


print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
