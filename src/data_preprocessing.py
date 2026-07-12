import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack
import textstat
import re


def extract_text_features(text):
    text = str(text)
    
    features = {
        'text_length': len(text),
        'word_count': len(text.split()),
        'sentence_count': len(text.split('.')),
        'avg_word_length': np.mean([len(word) for word in text.split()]) if text.split() else 0,
        'unique_words': len(set(text.lower().split())),
        'type_token_ratio': len(set(text.lower().split())) / len(text.split()) if text.split() else 0,
        'flesch_reading_ease': textstat.flesch_reading_ease(text),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(text),
        'punctuation_count': len(re.findall(r'[^\w\s]', text)),
        'uppercase_count': sum(1 for c in text if c.isupper()),
        'digit_count': sum(1 for c in text if c.isdigit())
    }
    
    return features


def create_tfidf_features(X_train, X_val, X_test, max_features=10000, ngram_range=(1, 3)):
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        stop_words='english',
        analyzer='word'
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train.astype(str))
    X_val_tfidf = tfidf.transform(X_val.astype(str))
    X_test_tfidf = tfidf.transform(X_test.astype(str))
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf


def create_char_ngram_features(X_train, X_val, X_test, max_features=5000, ngram_range=(2, 5)):
    char_tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        analyzer='char'
    )
    
    X_train_char = char_tfidf.fit_transform(X_train.astype(str))
    X_val_char = char_tfidf.transform(X_val.astype(str))
    X_test_char = char_tfidf.transform(X_test.astype(str))
    
    return X_train_char, X_val_char, X_test_char, char_tfidf


def combine_features(tfidf_features, char_features, text_stats=None):
    if text_stats is not None:
        return hstack([tfidf_features, char_features, text_stats])
    return hstack([tfidf_features, char_features])


def preprocess_pipeline(X_train, X_val, X_test):
    X_train_tfidf, X_val_tfidf, X_test_tfidf, tfidf = create_tfidf_features(X_train, X_val, X_test)
    X_train_char, X_val_char, X_test_char, char_tfidf = create_char_ngram_features(X_train, X_val, X_test)
    
    X_train_combined = combine_features(X_train_tfidf, X_train_char)
    X_val_combined = combine_features(X_val_tfidf, X_val_char)
    X_test_combined = combine_features(X_test_tfidf, X_test_char)
    
    preprocessing_objects = {
        'word_tfidf': tfidf,
        'char_tfidf': char_tfidf
    }
    
    return X_train_combined, X_val_combined, X_test_combined, preprocessing_objects
