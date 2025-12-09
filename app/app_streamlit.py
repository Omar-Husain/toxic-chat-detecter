"""
Toxic Chat Detector - Streamlit Web App

Supports both models:
- Baseline: TF-IDF + Logistic Regression
- Neural Network: PyTorch (Embedding + Linear)

Usage:
    streamlit run app/app_streamlit.py
"""

import streamlit as st
import os
import sys
import re
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib
import torch

# Page config
st.set_page_config(page_title="Toxic Chat Detector", page_icon="🛡️")


def get_artifacts_dir():
    return os.path.join(os.path.dirname(__file__), '..', 'artifacts')


@st.cache_resource
def load_baseline_model():
    """Load the TF-IDF + Logistic Regression model."""
    path = os.path.join(get_artifacts_dir(), 'baseline_pipeline.joblib')
    if os.path.exists(path):
        return joblib.load(path)
    return None


@st.cache_resource
def load_pytorch_model():
    """Load the PyTorch neural network model."""
    artifacts_dir = get_artifacts_dir()
    model_path = os.path.join(artifacts_dir, 'torch_model.pt')
    vocab_path = os.path.join(artifacts_dir, 'vocab.json')
    config_path = os.path.join(artifacts_dir, 'torch_config.json')
    
    # Check if all files exist
    if not all(os.path.exists(p) for p in [model_path, vocab_path, config_path]):
        return None, None, None
    
    # Load vocab and config
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Import and create model
    from train_torch import TinyToxicClassifier
    
    model = TinyToxicClassifier(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    return model, vocab, config


def clean_text(text: str) -> str:
    """Clean text before prediction."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def predict_baseline(text, model):
    """Predict using baseline model."""
    cleaned = clean_text(text)
    label = model.predict([cleaned])[0]
    prob = model.predict_proba([cleaned])[0][1]
    return label, prob


def predict_pytorch(text, model, vocab, config):
    """Predict using PyTorch model."""
    from train_torch import tokenize_and_encode
    
    cleaned = clean_text(text)
    encoded = tokenize_and_encode(cleaned, vocab, config['max_length'])
    input_ids = torch.tensor([encoded], dtype=torch.long)
    
    with torch.no_grad():
        prob = model(input_ids).item()
    
    label = 1 if prob > 0.5 else 0
    return label, prob


def main():
    st.title("🛡️ Toxic Chat Detector")
    
    # Load both models
    baseline_model = load_baseline_model()
    pytorch_model, vocab, config = load_pytorch_model()
    
    # Check which models are available
    baseline_available = baseline_model is not None
    pytorch_available = pytorch_model is not None
    
    if not baseline_available and not pytorch_available:
        st.error("⚠️ No models found! Train a model first:")
        st.code("python src/train_baseline.py --csv data/gaming_toxic.csv")
        st.code("python src/train_torch.py --csv data/gaming_toxic.csv")
        return
    
    # Model selector
    model_options = []
    if baseline_available:
        model_options.append("Baseline (TF-IDF + Logistic Regression)")
    if pytorch_available:
        model_options.append("Neural Network (PyTorch)")
    
    selected_model = st.selectbox("Select Model:", model_options)
    
    # Show model status
    if "Baseline" in selected_model:
        st.success("✅ Baseline model loaded")
        st.caption("Traditional ML: TF-IDF vectorization + Logistic Regression")
    else:
        st.success("✅ PyTorch model loaded")
        st.caption("Neural Network: Word Embeddings + Mean Pooling + Linear layers")
    
    st.markdown("---")
    
    # Text input
    message = st.text_input("Enter a message:", placeholder="Type a message to analyze...")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary"):
        if message.strip():
            # Get prediction based on selected model
            if "Baseline" in selected_model:
                label, prob = predict_baseline(message, baseline_model)
            else:
                label, prob = predict_pytorch(message, pytorch_model, vocab, config)
            
            # Show results
            st.markdown("---")
            st.subheader("Results")
            
            if label == 1:
                st.error(f"⚠️ **TOXIC** (Confidence: {prob:.1%})")
            else:
                st.success(f"✅ **Not Toxic** (Confidence: {1-prob:.1%})")
            
            # Toxicity bar
            st.write("Toxicity Score:")
            st.progress(prob)
        else:
            st.warning("Please enter a message to analyze.")


if __name__ == "__main__":
    main()
