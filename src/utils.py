"""
utils.py - Helper functions for the Toxic Chat Detector project.

This module contains utility functions for:
- Loading and preprocessing data
- Text cleaning
- File I/O operations
"""

import os
import re
import json
import pandas as pd
from typing import List, Tuple, Optional


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load a CSV file and ensure it has the required columns.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        DataFrame with 'text' and 'label' columns
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        ValueError: If required columns are missing
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Try to adapt columns if they have different names
    # Common alternatives for text column
    text_alternatives = ['text', 'message', 'comment', 'comment_text', 'content']
    # Common alternatives for label column
    label_alternatives = ['label', 'toxic', 'is_toxic', 'class', 'target']
    
    # Find and rename text column
    text_col = None
    for col in text_alternatives:
        if col in df.columns:
            text_col = col
            break
    
    # Find and rename label column
    label_col = None
    for col in label_alternatives:
        if col in df.columns:
            label_col = col
            break
    
    # Validate we found the columns
    if text_col is None:
        raise ValueError(f"Could not find text column. Available: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find label column. Available: {list(df.columns)}")
    
    # Create standardized DataFrame
    result = pd.DataFrame({
        'text': df[text_col],
        'label': df[label_col]
    })
    
    # Ensure label is integer (0 or 1)
    result['label'] = result['label'].astype(int)
    
    return result


def clean_text(text: str) -> str:
    """
    Basic text cleaning for chat messages.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def preprocess_texts(texts: List[str], clean: bool = True) -> List[str]:
    """
    Preprocess a list of text strings.
    
    Args:
        texts: List of raw text strings
        clean: Whether to apply text cleaning
        
    Returns:
        List of preprocessed text strings
    """
    if clean:
        return [clean_text(t) for t in texts]
    return list(texts)


def ensure_dir(directory: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Path to the directory
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def save_json(data: dict, filepath: str) -> None:
    """
    Save a dictionary to a JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save the JSON file
    """
    # Ensure the directory exists
    ensure_dir(os.path.dirname(filepath))
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved JSON to: {filepath}")


def load_json(filepath: str) -> dict:
    """
    Load a dictionary from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary loaded from the file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test sets.
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        test_size: Fraction of data for test set (default: 0.2)
        random_state: Random seed for reproducibility
        stratify: Whether to stratify by label (default: True)
        
    Returns:
        Tuple of (train_df, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    stratify_col = df['label'] if stratify else None
    
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_col
    )
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    return train_df, test_df


def get_class_distribution(labels: List[int]) -> dict:
    """
    Get the distribution of classes in the labels.
    
    Args:
        labels: List of integer labels
        
    Returns:
        Dictionary with class counts and percentages
    """
    total = len(labels)
    counts = {}
    
    for label in set(labels):
        count = labels.count(label) if isinstance(labels, list) else sum(labels == label)
        counts[label] = {
            'count': int(count),
            'percentage': round(count / total * 100, 2)
        }
    
    return counts


if __name__ == "__main__":
    # Quick test of the utilities
    print("Testing utils.py...")
    
    # Test text cleaning
    test_text = "  Check out http://example.com   THIS IS A TEST  "
    cleaned = clean_text(test_text)
    print(f"Original: '{test_text}'")
    print(f"Cleaned: '{cleaned}'")
    
    # Test data loading
    try:
        df = load_data("data/sample_data.csv")
        print(f"\nLoaded {len(df)} samples")
        print(f"Columns: {list(df.columns)}")
        print(f"Class distribution: {get_class_distribution(df['label'].tolist())}")
    except FileNotFoundError as e:
        print(f"Note: {e}")
    
    print("\nutils.py tests complete!")

