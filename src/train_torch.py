"""
train_torch.py - Train a simple PyTorch model for toxic chat detection.

This script implements a tiny neural network:
- Embedding layer: converts word indices to dense vectors
- Mean pooling: averages all word embeddings in a message
- Linear layer: maps to output with sigmoid activation

This is an optional, more advanced model compared to the TF-IDF baseline.

Usage:
    python src/train_torch.py --csv data/sample_data.csv

The trained model and vocab are saved to the artifacts/ folder.
"""

import argparse
import os
import sys
import json

# Add the src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from utils import load_data, split_data, ensure_dir, save_json, clean_text
from metrics import compute_metrics, get_classification_report, print_misclassified, find_misclassified


# ============================================================
# VOCABULARY BUILDING
# ============================================================

def build_vocab(texts, min_freq=1, max_vocab_size=10000):
    """
    Build a vocabulary from a list of texts.
    
    Args:
        texts: List of text strings
        min_freq: Minimum frequency for a word to be included
        max_vocab_size: Maximum vocabulary size
        
    Returns:
        Dictionary mapping words to indices
    """
    # Count word frequencies
    word_counts = Counter()
    for text in texts:
        words = text.lower().split()
        word_counts.update(words)
    
    # Filter by frequency and limit size
    common_words = [
        word for word, count in word_counts.most_common(max_vocab_size)
        if count >= min_freq
    ]
    
    # Create vocabulary with special tokens
    vocab = {
        '<PAD>': 0,  # Padding token
        '<UNK>': 1   # Unknown token
    }
    
    for word in common_words:
        vocab[word] = len(vocab)
    
    print(f"Vocabulary size: {len(vocab)}")
    return vocab


def tokenize_and_encode(text, vocab, max_length=50):
    """
    Tokenize and encode a text string.
    
    Args:
        text: Input text string
        vocab: Vocabulary dictionary
        max_length: Maximum sequence length (pad/truncate)
        
    Returns:
        List of word indices
    """
    # Tokenize by whitespace
    words = text.lower().split()
    
    # Convert to indices (use <UNK> for unknown words)
    indices = [vocab.get(word, vocab['<UNK>']) for word in words]
    
    # Pad or truncate to max_length
    if len(indices) < max_length:
        indices = indices + [vocab['<PAD>']] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    
    return indices


# ============================================================
# DATASET CLASS
# ============================================================

class ToxicDataset(Dataset):
    """PyTorch Dataset for toxic chat data."""
    
    def __init__(self, texts, labels, vocab, max_length=50):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text strings
            labels: List of labels (0 or 1)
            vocab: Vocabulary dictionary
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Encode the text
        encoded = tokenize_and_encode(text, self.vocab, self.max_length)
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }


# ============================================================
# MODEL DEFINITION
# ============================================================

class TinyToxicClassifier(nn.Module):
    """
    A simple neural network for text classification.
    
    Architecture:
    - Embedding: word indices -> dense vectors
    - Mean pooling: average all word embeddings
    - Linear: map to single output
    - Sigmoid: convert to probability
    """
    
    def __init__(self, vocab_size, embedding_dim=64, hidden_dim=32):
        """
        Initialize the model.
        
        Args:
            vocab_size: Size of the vocabulary
            embedding_dim: Dimension of word embeddings
            hidden_dim: Dimension of hidden layer
        """
        super(TinyToxicClassifier, self).__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # <PAD> token
        )
        
        # Simple classifier
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids):
        """
        Forward pass.
        
        Args:
            input_ids: Tensor of shape (batch_size, seq_length)
            
        Returns:
            Tensor of shape (batch_size,) with probabilities
        """
        # Get embeddings: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(input_ids)
        
        # Create mask for padding (1 for real tokens, 0 for padding)
        mask = (input_ids != 0).float().unsqueeze(-1)
        
        # Mean pooling (ignore padding)
        # Sum embeddings and divide by number of real tokens
        summed = (embedded * mask).sum(dim=1)
        lengths = mask.sum(dim=1).clamp(min=1)  # Avoid division by zero
        pooled = summed / lengths
        
        # Classify
        output = self.classifier(pooled)
        
        return output.squeeze(-1)


# ============================================================
# TRAINING FUNCTIONS
# ============================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """Evaluate the model and return predictions."""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(input_ids)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).long().cpu().numpy()
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_probs, all_labels


# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

def train_torch_model(
    csv_path: str,
    artifacts_dir: str = "artifacts",
    test_size: float = 0.2,
    num_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    max_length: int = 50,
    embedding_dim: int = 64
) -> dict:
    """
    Train the PyTorch model and save artifacts.
    
    Args:
        csv_path: Path to the CSV file
        artifacts_dir: Directory to save artifacts
        test_size: Fraction of data for testing
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        max_length: Maximum sequence length
        embedding_dim: Embedding dimension
        
    Returns:
        Dictionary with evaluation metrics
    """
    print("="*60)
    print("TRAINING PYTORCH MODEL (Embedding + Mean Pooling + Linear)")
    print("="*60)
    
    # Determine device (CPU for simplicity)
    device = torch.device('cpu')
    print(f"\nUsing device: {device}")
    
    # --- Step 1: Load and prepare data ---
    print(f"\n[1/6] Loading data from: {csv_path}")
    df = load_data(csv_path)
    df['text'] = df['text'].apply(clean_text)
    print(f"Total samples: {len(df)}")
    
    # Split data
    print(f"\n[2/6] Splitting data...")
    train_df, test_df = split_data(df, test_size=test_size, stratify=True)
    
    X_train = train_df['text'].tolist()
    y_train = train_df['label'].tolist()
    X_test = test_df['text'].tolist()
    y_test = test_df['label'].tolist()
    
    # --- Step 2: Build vocabulary ---
    print("\n[3/6] Building vocabulary...")
    vocab = build_vocab(X_train, min_freq=1, max_vocab_size=5000)
    
    # --- Step 3: Create datasets and dataloaders ---
    print("\n[4/6] Creating datasets...")
    train_dataset = ToxicDataset(X_train, y_train, vocab, max_length)
    test_dataset = ToxicDataset(X_test, y_test, vocab, max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # --- Step 4: Create model ---
    print("\n[5/6] Training model...")
    model = TinyToxicClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=32
    )
    model = model.to(device)
    
    # Print model summary
    print(f"\nModel architecture:")
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Setup training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"\nTraining for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate on test set
        test_preds, test_probs, test_labels = evaluate(model, test_loader, device)
        test_metrics = compute_metrics(test_labels, test_preds, verbose=False)
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Loss: {train_loss:.4f} | "
              f"Test F1: {test_metrics['f1']:.4f}")
    
    # --- Step 5: Final evaluation ---
    print("\n[6/6] Final evaluation...")
    y_pred, y_proba, y_true = evaluate(model, test_loader, device)
    
    metrics = compute_metrics(y_true, y_pred, verbose=True)
    print("\nDetailed Classification Report:")
    print(get_classification_report(y_true, y_pred))
    
    # Find misclassified
    false_positives, false_negatives = find_misclassified(
        X_test, y_true, y_pred, y_proba, n_examples=3
    )
    print_misclassified(false_positives, false_negatives)
    
    # --- Step 6: Save artifacts ---
    print(f"\nSaving artifacts to: {artifacts_dir}/")
    ensure_dir(artifacts_dir)
    
    # Save model
    model_path = os.path.join(artifacts_dir, "torch_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"  - Model saved: {model_path}")
    
    # Save vocabulary
    vocab_path = os.path.join(artifacts_dir, "vocab.json")
    save_json(vocab, vocab_path)
    
    # Save model config
    config = {
        'vocab_size': len(vocab),
        'embedding_dim': embedding_dim,
        'hidden_dim': 32,
        'max_length': max_length
    }
    config_path = os.path.join(artifacts_dir, "torch_config.json")
    save_json(config, config_path)
    
    # Save metrics
    torch_metrics_path = os.path.join(artifacts_dir, "torch_metrics.json")
    save_json({
        'metrics': {k: round(v, 4) for k, v in metrics.items()},
        'training': {
            'epochs': num_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, torch_metrics_path)
    
    print("\n" + "="*60)
    print("PYTORCH TRAINING COMPLETE!")
    print("="*60)
    print(f"\nArtifacts saved to: {artifacts_dir}/")
    print("  - torch_model.pt (model weights)")
    print("  - vocab.json (vocabulary)")
    print("  - torch_config.json (model configuration)")
    print("  - torch_metrics.json (evaluation metrics)")
    
    return metrics


def main():
    """Main function to run training from command line."""
    parser = argparse.ArgumentParser(
        description="Train the tiny PyTorch model for toxic chat detection"
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
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )
    
    args = parser.parse_args()
    
    metrics = train_torch_model(
        csv_path=args.csv,
        artifacts_dir=args.artifacts,
        num_epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    return metrics


if __name__ == "__main__":
    main()

