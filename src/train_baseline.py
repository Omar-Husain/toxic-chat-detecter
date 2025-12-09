"""
train_baseline.py - Train a TF-IDF + Logistic Regression baseline model.

This script:
1. Loads the dataset from a CSV file
2. Splits into train/test sets (80/20 with stratification)
3. Builds a Pipeline with TfidfVectorizer and LogisticRegression
4. Trains the model and evaluates on the test set
5. Saves the model, vectorizer, metrics, and confusion matrix

Usage:
    python src/train_baseline.py --csv data/sample_data.csv

The trained model and artifacts are saved to the artifacts/ folder.
"""

import argparse
import os
import sys

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from utils import load_data, split_data, ensure_dir, save_json, clean_text
from metrics import (
    compute_metrics,
    get_classification_report,
    plot_confusion_matrix,
    find_misclassified,
    print_misclassified,
    create_metrics_summary
)


def create_baseline_pipeline() -> Pipeline:
    """
    Create the TF-IDF + Logistic Regression pipeline.
    
    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        # Step 1: Convert text to TF-IDF features
        ('tfidf', TfidfVectorizer(
            max_features=5000,      # Limit vocabulary size
            ngram_range=(1, 2),     # Use unigrams and bigrams
            min_df=2,               # Ignore very rare words
            stop_words='english'    # Remove common English words
        )),
        # Step 2: Logistic Regression classifier
        ('classifier', LogisticRegression(
            max_iter=1000,          # Ensure convergence
            random_state=42,        # Reproducibility
            C=1.0,                  # Regularization strength
            class_weight='balanced' # Handle class imbalance
        ))
    ])
    
    return pipeline


def train_baseline(
    csv_path: str,
    artifacts_dir: str = "artifacts",
    test_size: float = 0.2
) -> dict:
    """
    Train the baseline model and save artifacts.
    
    Args:
        csv_path: Path to the CSV file with 'text' and 'label' columns
        artifacts_dir: Directory to save model artifacts
        test_size: Fraction of data for testing
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("TRAINING BASELINE MODEL (TF-IDF + Logistic Regression)")
    print("="*60)
    
    # --- Step 1: Load the data ---
    print(f"\n[1/6] Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"Total samples: {len(df)}")
    print(f"Class distribution:")
    print(df['label'].value_counts())
    
    # --- Step 2: Clean the text ---
    print("\n[2/6] Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)
    
    # --- Step 3: Split the data ---
    print(f"\n[3/6] Splitting data ({int((1-test_size)*100)}/{int(test_size*100)} train/test)...")
    train_df, test_df = split_data(df, test_size=test_size, stratify=True)
    
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    
    # --- Step 4: Create and train the pipeline ---
    print("\n[4/6] Training the model...")
    pipeline = create_baseline_pipeline()
    pipeline.fit(X_train, y_train)
    print("Training complete!")
    
    # --- Step 5: Evaluate the model ---
    print("\n[5/6] Evaluating on test set...")
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of toxic class
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, verbose=True)
    
    # Print detailed classification report
    print("\nDetailed Classification Report:")
    print(get_classification_report(y_test, y_pred))
    
    # Find and print misclassified examples
    false_positives, false_negatives = find_misclassified(
        X_test, y_test, y_pred, y_proba, n_examples=5
    )
    print_misclassified(false_positives, false_negatives)
    
    # --- Step 6: Save artifacts ---
    print(f"\n[6/6] Saving artifacts to: {artifacts_dir}/")
    ensure_dir(artifacts_dir)
    
    # Save the complete pipeline (includes both vectorizer and classifier)
    pipeline_path = os.path.join(artifacts_dir, "baseline_pipeline.joblib")
    joblib.dump(pipeline, pipeline_path)
    print(f"  - Pipeline saved: {pipeline_path}")
    
    # Also save them separately for flexibility
    vectorizer_path = os.path.join(artifacts_dir, "vectorizer.joblib")
    model_path = os.path.join(artifacts_dir, "model.joblib")
    joblib.dump(pipeline.named_steps['tfidf'], vectorizer_path)
    joblib.dump(pipeline.named_steps['classifier'], model_path)
    print(f"  - Vectorizer saved: {vectorizer_path}")
    print(f"  - Model saved: {model_path}")
    
    # Save confusion matrix plot
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")
    cm = plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Save metrics to JSON
    metrics_summary = create_metrics_summary(
        metrics, cm, false_positives, false_negatives
    )
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    save_json(metrics_summary, metrics_path)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nArtifacts saved to: {artifacts_dir}/")
    print("  - baseline_pipeline.joblib (complete pipeline)")
    print("  - vectorizer.joblib (TF-IDF vectorizer)")
    print("  - model.joblib (Logistic Regression model)")
    print("  - confusion_matrix.png")
    print("  - metrics.json")
    
    return metrics


def main():
    """Main function to run training from command line."""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Train the TF-IDF + Logistic Regression baseline model"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/sample_data.csv",
        help="Path to the CSV file (default: data/sample_data.csv)"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Directory to save artifacts (default: artifacts)"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (default: 0.2)"
    )
    
    args = parser.parse_args()
    
    # Run training
    metrics = train_baseline(
        csv_path=args.csv,
        artifacts_dir=args.artifacts,
        test_size=args.test_size
    )
    
    return metrics


if __name__ == "__main__":
    main()

