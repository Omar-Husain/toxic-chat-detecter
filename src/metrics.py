"""
metrics.py - Evaluation metrics for the Toxic Chat Detector project.

This module contains functions for:
- Computing classification metrics (accuracy, precision, recall, F1)
- Creating confusion matrices
- Generating classification reports
- Finding misclassified examples (false positives/negatives)
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


def compute_metrics(
    y_true: List[int],
    y_pred: List[int],
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        verbose: Whether to print the metrics
        
    Returns:
        Dictionary with accuracy, precision, recall, and F1 score
    """
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # Print if verbose
    if verbose:
        print("\n" + "="*50)
        print("CLASSIFICATION METRICS")
        print("="*50)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print("="*50)
    
    return metrics


def get_classification_report(
    y_true: List[int],
    y_pred: List[int],
    target_names: Optional[List[str]] = None
) -> str:
    """
    Generate a detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        target_names: Names for each class (default: ['Non-Toxic', 'Toxic'])
        
    Returns:
        Classification report as a string
    """
    if target_names is None:
        target_names = ['Non-Toxic', 'Toxic']
    
    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        zero_division=0
    )
    
    return report


def plot_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
    save_path: Optional[str] = None,
    labels: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create and optionally save a confusion matrix plot.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        save_path: Path to save the plot (optional)
        labels: Class labels for the axes (default: ['Non-Toxic', 'Toxic'])
        
    Returns:
        The confusion matrix as a numpy array
    """
    if labels is None:
        labels = ['Non-Toxic', 'Toxic']
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        annot_kws={'size': 14}
    )
    
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")
    
    plt.close()
    
    return cm


def find_misclassified(
    texts: List[str],
    y_true: List[int],
    y_pred: List[int],
    y_proba: Optional[List[float]] = None,
    n_examples: int = 5
) -> Tuple[List[Dict], List[Dict]]:
    """
    Find false positives and false negatives.
    
    Args:
        texts: Original text samples
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_proba: Prediction probabilities (optional)
        n_examples: Number of examples to return for each type
        
    Returns:
        Tuple of (false_positives, false_negatives) as lists of dicts
    """
    false_positives = []  # Predicted toxic, but actually non-toxic
    false_negatives = []  # Predicted non-toxic, but actually toxic
    
    for i, (text, true_label, pred_label) in enumerate(zip(texts, y_true, y_pred)):
        # Get probability if available
        prob = y_proba[i] if y_proba is not None else None
        
        example = {
            'text': text,
            'true_label': int(true_label),
            'predicted_label': int(pred_label),
            'probability': float(prob) if prob is not None else None
        }
        
        # False positive: predicted 1, actual 0
        if pred_label == 1 and true_label == 0:
            false_positives.append(example)
        
        # False negative: predicted 0, actual 1
        elif pred_label == 0 and true_label == 1:
            false_negatives.append(example)
    
    # Limit to n_examples
    false_positives = false_positives[:n_examples]
    false_negatives = false_negatives[:n_examples]
    
    return false_positives, false_negatives


def print_misclassified(
    false_positives: List[Dict],
    false_negatives: List[Dict]
) -> None:
    """
    Print misclassified examples in a readable format.
    
    Args:
        false_positives: List of false positive examples
        false_negatives: List of false negative examples
    """
    print("\n" + "="*60)
    print("MISCLASSIFIED EXAMPLES")
    print("="*60)
    
    print("\n--- FALSE POSITIVES (predicted toxic, actually non-toxic) ---")
    if false_positives:
        for i, fp in enumerate(false_positives, 1):
            prob_str = f" (prob: {fp['probability']:.3f})" if fp['probability'] else ""
            print(f"{i}. \"{fp['text']}\"{prob_str}")
    else:
        print("No false positives found.")
    
    print("\n--- FALSE NEGATIVES (predicted non-toxic, actually toxic) ---")
    if false_negatives:
        for i, fn in enumerate(false_negatives, 1):
            prob_str = f" (prob: {fn['probability']:.3f})" if fn['probability'] else ""
            print(f"{i}. \"{fn['text']}\"{prob_str}")
    else:
        print("No false negatives found.")
    
    print("="*60)


def create_metrics_summary(
    metrics: Dict[str, float],
    cm: np.ndarray,
    false_positives: List[Dict],
    false_negatives: List[Dict]
) -> Dict:
    """
    Create a comprehensive metrics summary for saving.
    
    Args:
        metrics: Dictionary of computed metrics
        cm: Confusion matrix
        false_positives: List of false positive examples
        false_negatives: List of false negative examples
        
    Returns:
        Complete summary dictionary
    """
    summary = {
        'metrics': {k: round(v, 4) for k, v in metrics.items()},
        'confusion_matrix': {
            'true_negatives': int(cm[0, 0]),
            'false_positives': int(cm[0, 1]),
            'false_negatives': int(cm[1, 0]),
            'true_positives': int(cm[1, 1])
        },
        'misclassified_examples': {
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    }
    
    return summary


if __name__ == "__main__":
    # Quick test of the metrics functions
    print("Testing metrics.py...")
    
    # Create sample data
    y_true = [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
    y_pred = [0, 0, 1, 1, 0, 1, 0, 1, 0, 1]
    texts = [
        "Hello friend",
        "Great game",
        "Nice try",  # FP
        "You're terrible",
        "Good match",  # FN
        "I hate you",
        "Let's play",
        "Go away loser",
        "Thanks!",
        "You suck"
    ]
    y_proba = [0.1, 0.2, 0.6, 0.9, 0.3, 0.95, 0.15, 0.85, 0.1, 0.8]
    
    # Test compute_metrics
    metrics = compute_metrics(y_true, y_pred)
    
    # Test classification report
    print("\nClassification Report:")
    print(get_classification_report(y_true, y_pred))
    
    # Test confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, save_path=None)
    print(f"\nConfusion Matrix:\n{cm}")
    
    # Test finding misclassified
    fp, fn = find_misclassified(texts, y_true, y_pred, y_proba, n_examples=3)
    print_misclassified(fp, fn)
    
    print("\nmetrics.py tests complete!")

