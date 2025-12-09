"""
Toxic Chat Detector - Streamlit Web App (ML Version)

Usage:
    streamlit run app/app_streamlit.py
"""

import streamlit as st
import os
import sys
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import joblib

# Page config
st.set_page_config(page_title="Toxic Chat Detector", page_icon="🛡️")


@st.cache_resource
def load_model():
    """Load the baseline ML model."""
    artifacts_dir = os.path.join(os.path.dirname(__file__), '..', 'artifacts')
    pipeline_path = os.path.join(artifacts_dir, 'baseline_pipeline.joblib')
    
    if os.path.exists(pipeline_path):
        return joblib.load(pipeline_path)
    return None


def clean_text(text: str) -> str:
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def main():
    st.title("🛡️ Toxic Chat Detector")
    st.write("Detect toxic messages using Machine Learning (TF-IDF + Logistic Regression)")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Model not found! Please train the model first:")
        st.code("python src/train_baseline.py --csv data/your_data.csv", language="bash")
        st.info("You need a CSV file with 'text' and 'label' columns (0=non-toxic, 1=toxic)")
        return
    
    st.success("✅ ML Model loaded successfully!")
    
    # Text input
    message = st.text_input("Enter a message:", placeholder="Type a message to analyze...")
    
    # Analyze button
    if st.button("🔍 Analyze", type="primary"):
        if message.strip():
            cleaned = clean_text(message)
            label = model.predict([cleaned])[0]
            prob = model.predict_proba([cleaned])[0][1]
            
            st.markdown("---")
            st.subheader("Results")
            
            if label == 1:
                st.error(f"⚠️ **TOXIC** (Confidence: {prob:.1%})")
            else:
                st.success(f"✅ **Not Toxic** (Confidence: {1-prob:.1%})")
            
            # Show probability bar
            st.write("Toxicity Score:")
            st.progress(prob)
        else:
            st.warning("Please enter a message to analyze.")


if __name__ == "__main__":
    main()
