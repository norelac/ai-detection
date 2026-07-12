import re
import string


def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def remove_urls(text):
    return re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)


def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


def format_metrics(metrics):
    formatted = {}
    for key, value in metrics.items():
        formatted[key] = f"{value:.4f}"
    return formatted


def load_data_splits(base_path='../data/processed/'):
    import pandas as pd
    
    train_df = pd.read_csv(f'{base_path}train.csv')
    val_df = pd.read_csv(f'{base_path}val.csv')
    test_df = pd.read_csv(f'{base_path}test.csv')
    
    return train_df, val_df, test_df
