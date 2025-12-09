# 🛡️ Toxic Chat Detector

By Omar Husain 100491847

**CSCI 4050U - Machine Learning Final Project**

## 📋 What It Does

This project uses machine learning to classify chat messages as **toxic** or **not toxic**. It can detect insults, harassment, and hate speech commonly found in online gaming.

**Example:**
- "great game everyone!" → ✅ Not Toxic
- "you are trash at this game" → ⚠️ Toxic

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd toxic-chat-detector
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python src/train_baseline.py --csv data/gaming_toxic.csv
```

### 3. Run the Web App

```bash
streamlit run app/app_streamlit.py
```

Then open http://localhost:8501 in your browser.

## 📁 Project Structure

```
toxic-chat-detector/
├── app/
│   └── app_streamlit.py    # Web app interface
├── data/
│   ├── gaming_toxic.csv    # Gaming-specific dataset (~2,200 samples)
│   ├── twitter_hate.csv    # Twitter hate speech (~25,000 samples)
│   └── README.md           # Dataset documentation
├── src/
│   ├── train_baseline.py   # Train TF-IDF + Logistic Regression model
│   ├── train_torch.py      # Train PyTorch neural network (optional)
│   ├── infer.py            # Make predictions from command line
│   ├── utils.py            # Helper functions
│   └── metrics.py          # Evaluation metrics
├── artifacts/              # Saved models (created after training)
├── notebooks/
│   └── exploration.ipynb   # Data exploration notebook
└── requirements.txt
```

## 🧠 How It Works

### Model: TF-IDF + Logistic Regression

1. **TF-IDF**: Converts text into numbers by counting word frequencies
2. **Logistic Regression**: Classifies the numbers as toxic (1) or not toxic (0)

```
"you are trash" → [0.2, 0.1, 0.8, ...] → Model → 0.95 → TOXIC
```

## 📊 Results

Trained on gaming data (~2,200 samples):

| Metric    | Score |
|-----------|-------|
| Accuracy  | ~94%  |
| Precision | ~95%  |
| Recall    | ~94%  |
| F1 Score  | ~94%  |

## 💻 Commands

```bash
# Train the model
python src/train_baseline.py --csv data/gaming_toxic.csv

# Make a single prediction
python src/infer.py --text "you are awesome"

# Predict from a file
python src/infer.py --file input.csv --output predictions.csv

# Run the web app
streamlit run app/app_streamlit.py

# Train PyTorch model (optional)
python src/train_torch.py --csv data/gaming_toxic.csv
```

## 📚 Datasets

| Dataset | Samples | Best For |
|---------|---------|----------|
| `gaming_toxic.csv` | ~2,200 | Gaming chat detection |
| `twitter_hate.csv` | ~25,000 | General hate speech |

See `data/README.md` for more details.

## 🌐 Web App Features

- Enter any message and get instant classification
- Shows confidence score and toxicity bar
- Simple, clean interface

## 🔮 Future Improvements

- Use BERT for better accuracy
- Detect specific types of toxicity (insults, threats, etc.)
- Add multi-language support
- Build an API for game integration

## 👤 Author
Omar Husain 100491847
CSCI 4050U - Machine Learning  
Ontario Tech University

## 📄 License

MIT License

