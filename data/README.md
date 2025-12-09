# Dataset Information

## Overview

This folder contains datasets for training the Toxic Chat Detector model.

## Available Datasets

### 1. `gaming_toxic.csv` (Recommended for Gaming)
- **Samples**: ~2,200
- **Source**: Custom gaming-specific toxic phrases
- **Best for**: Detecting toxicity in video game chats
- **Contains**: Gaming insults, toxic behavior, and positive messages

### 2. `twitter_hate.csv`
- **Samples**: ~25,000
- **Source**: [Twitter Hate Speech Dataset](https://github.com/t-davidson/hate-speech-and-offensive-language)
- **Best for**: General hate speech detection
- **Contains**: Tweets labeled as hate speech, offensive, or neither

### 3. `sample_data.csv`
- **Samples**: 30
- **Source**: Hand-crafted examples
- **Best for**: Quick testing only (too small for real training)

### 4. `combined_data.csv`
- **Samples**: ~25,000
- **Source**: Combination of twitter_hate.csv + sample_data.csv + extra gaming examples
- **Best for**: General toxicity detection with some gaming context

## Data Format

All CSV files use the same format:

| Column | Type | Description |
|--------|------|-------------|
| text   | str  | The message text |
| label  | int  | 0 = non-toxic, 1 = toxic |

## How to Train

Use any of these datasets to train the model:

```bash
# Train with gaming data (recommended)
python src/train_baseline.py --csv data/gaming_toxic.csv

# Train with Twitter data
python src/train_baseline.py --csv data/twitter_hate.csv

# Train with combined data
python src/train_baseline.py --csv data/combined_data.csv
```

## Adding Your Own Data

Create a CSV file with two columns:
1. `text` - The message to classify
2. `label` - 0 for non-toxic, 1 for toxic

Example:
```csv
text,label
"great game everyone!",0
"you are terrible at this",1
"nice shot!",0
"uninstall the game noob",1
```

## Data Sources

- **Twitter Dataset**: Davidson et al. (2017) - "Automated Hate Speech Detection and the Problem of Offensive Language"
- **Gaming Dataset**: Custom curated for this project
