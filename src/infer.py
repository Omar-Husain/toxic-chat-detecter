"""
infer.py - Inference script for the Toxic Chat Detector.

This script loads a saved model and makes predictions on:
- A single text input (--text argument)
- A CSV file with a 'text' column (--file argument)

Usage:
    # Single text prediction
    python src/infer.py --text "you are awesome"
    
    # Batch prediction from file
    python src/infer.py --file input.csv --output predictions.csv
    
    # Use PyTorch model instead of baseline
    python src/infer.py --text "hello" --model torch

Output:
    Prints the predicted label (Toxic/Non-Toxic) and probability.
"""

import argparse
import os
import sys

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import joblib
import pandas as pd
import numpy as np
import torch

from utils import load_json, clean_text


def load_baseline_model(artifacts_dir: str = "artifacts"):
    """
    Load the TF-IDF + Logistic Regression baseline model.
    
    Args:
        artifacts_dir: Directory containing the saved artifacts
        
    Returns:
        The loaded sklearn Pipeline
    """
    pipeline_path = os.path.join(artifacts_dir, "baseline_pipeline.joblib")
    
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"Baseline model not found at {pipeline_path}. "
            "Please run train_baseline.py first."
        )
    
    pipeline = joblib.load(pipeline_path)
    print(f"Loaded baseline model from: {pipeline_path}")
    return pipeline


def load_torch_model(artifacts_dir: str = "artifacts"):
    """
    Load the PyTorch model.
    
    Args:
        artifacts_dir: Directory containing the saved artifacts
        
    Returns:
        Tuple of (model, vocab, config)
    """
    # Import here to avoid circular imports
    from train_torch import TinyToxicClassifier, tokenize_and_encode
    
    model_path = os.path.join(artifacts_dir, "torch_model.pt")
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    config_path = os.path.join(artifacts_dir, "torch_config.json")
    
    # Check files exist
    for path in [model_path, vocab_path, config_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"PyTorch model file not found: {path}. "
                "Please run train_torch.py first."
            )
    
    # Load vocabulary and config
    vocab = load_json(vocab_path)
    config = load_json(config_path)
    
    # Create model and load weights
    model = TinyToxicClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    print(f"Loaded PyTorch model from: {model_path}")
    return model, vocab, config


def predict_baseline(texts: list, pipeline) -> tuple:
    """
    Make predictions using the baseline model.
    
    Args:
        texts: List of text strings
        pipeline: Loaded sklearn Pipeline
        
    Returns:
        Tuple of (labels, probabilities)
    """
    # Clean texts
    cleaned_texts = [clean_text(t) for t in texts]
    
    # Get predictions
    labels = pipeline.predict(cleaned_texts)
    probabilities = pipeline.predict_proba(cleaned_texts)[:, 1]
    
    return labels, probabilities


def predict_torch(texts: list, model, vocab, config) -> tuple:
    """
    Make predictions using the PyTorch model.
    
    Args:
        texts: List of text strings
        model: Loaded PyTorch model
        vocab: Vocabulary dictionary
        config: Model configuration
        
    Returns:
        Tuple of (labels, probabilities)
    """
    from train_torch import tokenize_and_encode
    
    model.eval()
    max_length = config['max_length']
    
    # Clean and encode texts
    cleaned_texts = [clean_text(t) for t in texts]
    encoded = [tokenize_and_encode(t, vocab, max_length) for t in cleaned_texts]
    
    # Convert to tensor
    input_ids = torch.tensor(encoded, dtype=torch.long)
    
    # Get predictions
    with torch.no_grad():
        probabilities = model(input_ids).numpy()
    
    labels = (probabilities > 0.5).astype(int)
    
    return labels, probabilities


def format_prediction(text: str, label: int, probability: float) -> str:
    """
    Format a single prediction for display.
    
    Args:
        text: Input text
        label: Predicted label (0 or 1)
        probability: Prediction probability
        
    Returns:
        Formatted string
    """
    label_str = "TOXIC" if label == 1 else "NON-TOXIC"
    confidence = probability if label == 1 else 1 - probability
    
    output = f"""
{'='*60}
Text: "{text}"
{'='*60}
Prediction: {label_str}
Probability of toxicity: {probability:.4f}
Confidence: {confidence:.2%}
{'='*60}
"""
    return output


def predict_single(text: str, model_type: str = "baseline", artifacts_dir: str = "artifacts"):
    """
    Predict toxicity for a single text input.
    
    Args:
        text: Input text string
        model_type: "baseline" or "torch"
        artifacts_dir: Directory containing model artifacts
    """
    if model_type == "baseline":
        pipeline = load_baseline_model(artifacts_dir)
        labels, probs = predict_baseline([text], pipeline)
    else:
        model, vocab, config = load_torch_model(artifacts_dir)
        labels, probs = predict_torch([text], model, vocab, config)
    
    # Format and print result
    output = format_prediction(text, labels[0], probs[0])
    print(output)
    
    return labels[0], probs[0]


def predict_file(
    input_path: str,
    output_path: str,
    model_type: str = "baseline",
    artifacts_dir: str = "artifacts"
):
    """
    Predict toxicity for all texts in a CSV file.
    
    Args:
        input_path: Path to input CSV with 'text' column
        output_path: Path to save predictions CSV
        model_type: "baseline" or "torch"
        artifacts_dir: Directory containing model artifacts
    """
    # Load input data
    print(f"Loading input file: {input_path}")
    df = pd.read_csv(input_path)
    
    # Find text column
    text_col = None
    for col in ['text', 'message', 'comment', 'comment_text', 'content']:
        if col in df.columns:
            text_col = col
            break
    
    if text_col is None:
        raise ValueError(f"No text column found. Available columns: {list(df.columns)}")
    
    texts = df[text_col].tolist()
    print(f"Found {len(texts)} texts to predict")
    
    # Load model and predict
    if model_type == "baseline":
        pipeline = load_baseline_model(artifacts_dir)
        labels, probs = predict_baseline(texts, pipeline)
    else:
        model, vocab, config = load_torch_model(artifacts_dir)
        labels, probs = predict_torch(texts, model, vocab, config)
    
    # Create output dataframe
    output_df = pd.DataFrame({
        'text': texts,
        'predicted_label': labels,
        'toxic_probability': probs,
        'prediction': ['TOXIC' if l == 1 else 'NON-TOXIC' for l in labels]
    })
    
    # Save to file
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")
    
    # Print summary
    toxic_count = sum(labels)
    print(f"\nSummary:")
    print(f"  Total predictions: {len(labels)}")
    print(f"  Toxic: {toxic_count} ({toxic_count/len(labels)*100:.1f}%)")
    print(f"  Non-toxic: {len(labels) - toxic_count} ({(len(labels)-toxic_count)/len(labels)*100:.1f}%)")
    
    return output_df


def main():
    """Main function to run inference from command line."""
    parser = argparse.ArgumentParser(
        description="Make predictions with the Toxic Chat Detector"
    )
    
    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--text",
        type=str,
        help="Single text to classify"
    )
    input_group.add_argument(
        "--file",
        type=str,
        help="CSV file with 'text' column to classify"
    )
    
    # Other options
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Output file for batch predictions (default: predictions.csv)"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["baseline", "torch"],
        default="baseline",
        help="Model to use: 'baseline' (TF-IDF+LR) or 'torch' (default: baseline)"
    )
    parser.add_argument(
        "--artifacts",
        type=str,
        default="artifacts",
        help="Directory containing model artifacts (default: artifacts)"
    )
    
    args = parser.parse_args()
    
    # Run prediction
    if args.text:
        predict_single(args.text, args.model, args.artifacts)
    else:
        predict_file(args.file, args.output, args.model, args.artifacts)


if __name__ == "__main__":
    main()

