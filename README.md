# Bengali Hate Speech Detection with BERT

A multi-label classification system for detecting hate speech in Bengali social media content using fine-tuned BERT models.

## 🎯 Project Overview

This project addresses hate speech detection in Bengali, one of the world's most spoken languages with 260+ million speakers. The system can identify multiple types of toxic content simultaneously, making it practical for real-world social media moderation.

### Problem Statement
- Bengali hate speech detection is understudied compared to English
- Social media platforms need automated tools for Bengali content moderation
- Multi-label classification allows detecting multiple toxic behaviors in single posts

## 📊 Results Summary

### Model Performance
| Metric | Score |
|--------|-------|
| **Macro F1-Score** | **67.49%** |
| **Accuracy** | **70.84%** |
| **Micro F1-Score** | **67.91%** |

### Per-Category Performance
| Category | F1-Score | Performance Level |
|----------|----------|------------------|
| Vulgar | 82.03% | Excellent |
| Religious Hate | 77.69% | Excellent |
| Threat | 68.53% | Good |
| Insult | 67.25% | Good |
| Hate Speech | 60.81% | Decent |
| Trolling | 48.66% | Needs Improvement |

## 🔧 Technical Implementation

### Dataset
- **Total Samples:** 16,068 Bengali social media comments
- **Training:** 11,253 samples (70%)
- **Validation:** 2,404 samples (15%)  
- **Test:** 2,411 samples (15%)
- **Source:** Multi-labeled toxic comments from Facebook, YouTube, news platforms

### Model Architecture
- **Base Model:** [sagorsarker/bangla-bert-base](https://huggingface.co/sagorsarker/bangla-bert-base)
- **Task:** Multi-label sequence classification
- **Labels:** 6 categories (vulgar, hate, religious, threat, troll, insult)
- **Max Sequence Length:** 256 tokens

### Training Configuration
- **Epochs:** 3 (stopped early due to validation loss increase)
- **Batch Size:** 8 per device
- **Learning Rate:** 2e-5 with warmup
- **Hardware:** Kaggle Tesla P100 GPU
- **Training Time:** ~17 minutes

## 📁 Project Structure

bengali-hate-speech-detection/
├── notebooks/
│   ├── 01_data_exploration.ipynb          # Dataset analysis and visualization
│   ├── 02_preprocessing_dataset1.ipynb    # Text cleaning and data preparation
│   ├── 03_model_training.ipynb           # Baseline models and setup
│   └── 04_kaggle_bert_training.ipynb     # BERT fine-tuning on Kaggle
├── data/
│   ├── train_dataset1.csv               # Training data
│   ├── val_dataset1.csv                 # Validation data
│   ├── test_dataset1.csv                # Test data
├── requirements.txt                      # Python dependencies
└── README.md                            # Project documentation

## 🚀 Getting Started

### Prerequisites
```bash
# Python 3.8+ required
pip install torch transformers datasets scikit-learn pandas numpy matplotlib seaborn accelerate
```
### Installation  
```bash
git clone https://github.com/ridwanahmed002/bengali-hate-speech-detection.git  # Clone the repository
cd bengali-hate-speech-detection
pip install -r requirements.txt  # Install dependencies
``` 
