# Complete Project Guide: Sentiment Analysis with LSTM and Transformer Models

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Creating the Project from Scratch](#creating-the-project-from-scratch)
4. [Installation & Setup](#installation--setup)
5. [Running the Project](#running-the-project)
6. [Project Architecture](#project-architecture)
7. [Expected Results](#expected-results)
8. [Troubleshooting](#troubleshooting)

---

## 🎯 Project Overview

A comprehensive sentiment analysis system that implements and compares three state-of-the-art deep learning architectures:
- **Custom LSTM** - Recurrent neural network built from scratch
- **AWD-LSTM (ULMFiT)** - Pretrained language model with transfer learning
- **BERT** - Transformer-based architecture with self-attention

**Dataset:** IMDb Movie Reviews (50,000 labeled samples)  
**Task:** Binary sentiment classification (Positive/Negative)

---

## ✨ Features

### Core Features
✅ **Three Complete Model Implementations**
- Custom Bidirectional LSTM with dropout regularization
- AWD-LSTM with ULMFiT transfer learning approach
- BERT transformer with fine-tuning

✅ **Comprehensive Data Pipeline**
- Automatic dataset download from HuggingFace
- Custom vocabulary builder (25,000 words)
- Tokenization and padding utilities
- Separate preprocessing for LSTM and BERT

✅ **Advanced Training System**
- Early stopping to prevent overfitting
- Learning rate scheduling
- Gradient clipping for stable training
- Model checkpointing (saves best model)
- Training history tracking

✅ **Extensive Evaluation Metrics**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC curves
- Confusion matrices
- Classification reports
- Training/validation loss curves

✅ **Comparative Analysis**
- Side-by-side model comparison
- Performance visualization (bar charts, radar plots)
- Training time analysis
- Convergence comparison
- Automated report generation

✅ **Visualization Suite**
- Training loss and accuracy curves
- Validation performance plots
- Multi-model ROC curves
- Confusion matrices for each model
- Comprehensive comparison dashboard
- Convergence analysis plots

✅ **Production-Ready Code**
- Modular architecture
- Configurable hyperparameters
- Error handling
- Progress bars for long operations
- Reproducible results (fixed random seeds)
- GPU/CPU support

---

## 🛠️ Creating the Project from Scratch

### Step 1: Create Project Directory Structure

```bash
# Create main project folder
mkdir G_P1
cd G_P1

# Create subdirectories
mkdir models
mkdir training
mkdir utils
mkdir data
mkdir saved_models
mkdir results
mkdir results\plots
```

### Step 2: Create Configuration File

Create `config.py`:
```python
import torch
import os

# Device and seed
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Dataset settings
MAX_SEQUENCE_LENGTH = 500

# Custom LSTM configuration
CUSTOM_LSTM_CONFIG = {
    'vocab_size': 25000,
    'embedding_dim': 300,
    'hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.5,
    'bidirectional': True,
    'batch_size': 64,
    'learning_rate': 0.001,
    'num_epochs': 20,
    'weight_decay': 1e-5,
    'clip_grad': 5.0,
}

# AWD-LSTM configuration
AWD_LSTM_CONFIG = {
    'batch_size': 32,
    'learning_rate_lm': 1e-3,
    'learning_rate_classifier': 1e-2,
    'num_epochs_lm': 5,
    'num_epochs_classifier': 10,
    'dropout_mult': 0.5,
}

# BERT configuration
BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
}

# Training settings
EARLY_STOPPING_PATIENCE = 3
SAVE_BEST_MODEL = True
LOG_INTERVAL = 100
```

### Step 3: Implement Data Preprocessing

Create `utils/data_preprocessing.py` with:
- `Vocabulary` class for building word vocabulary
- `IMDbDataset` class for custom PyTorch dataset
- `load_imdb_data()` function
- `prepare_custom_lstm_data()` function
- `get_bert_dataloaders()` function

### Step 4: Implement Model Architectures

**Create `models/custom_lstm.py`:**
- `CustomLSTM` class with bidirectional LSTM
- Embedding layer → LSTM → Dropout → Fully Connected layers
- Support for variable sequence lengths

**Create `models/awd_lstm.py`:**
- `AWDLSTMSentimentClassifier` wrapper class
- Uses fastai's ULMFiT implementation
- Two-stage training: LM fine-tuning + classifier training

**Create `models/bert_model.py`:**
- `BERTSentimentClassifier` using HuggingFace
- `BERTTrainer` class for training loop
- AdamW optimizer with warmup

### Step 5: Create Training Scripts

**Create `training/train_custom_lstm.py`:**
- `CustomLSTMTrainer` class
- Training loop with validation
- Early stopping and checkpointing

**Create `training/train_awd_lstm.py`:**
- AWD-LSTM training function
- Two-stage training process

**Create `training/train_bert.py`:**
- BERT fine-tuning script
- Gradient accumulation support

### Step 6: Implement Evaluation Tools

**Create `utils/evaluate.py`:**
- `evaluate_model()` - Calculate all metrics
- `plot_confusion_matrix()` - Visualization
- `plot_training_history()` - Loss/accuracy curves
- `plot_roc_curve()` - ROC analysis

**Create `utils/comparative_analysis.py`:**
- `load_all_results()` - Load saved results
- `create_comparison_table()` - Generate comparison DataFrame
- `plot_metrics_comparison()` - Bar charts
- `plot_comprehensive_comparison()` - Dashboard
- `generate_comparison_report()` - Text report

### Step 7: Create Main Execution Script

Create `main.py`:
- Command-line argument parsing
- Model selection logic
- Orchestrates training and evaluation
- Runs comparative analysis

### Step 8: Create Package Initialization Files

Create `__init__.py` in each package:
- `models/__init__.py`
- `training/__init__.py`
- `utils/__init__.py`

### Step 9: Create Documentation Files

**Create `requirements.txt`:**
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
fastai>=2.7.12
```

**Create `README.md`** with project overview

**Create `.gitignore`** for version control

---

## 💻 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 8GB RAM minimum (16GB recommended)
- 5GB free disk space
- GPU optional (recommended for BERT)

### Step-by-Step Installation

#### 1. Clone or Download Project
```bash
cd path/to/project
cd G_P1
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**If you encounter numpy/pandas compatibility issues:**
```bash
pip uninstall numpy pandas -y
pip install numpy==1.24.3 pandas==2.0.3
```

#### 4. Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import datasets; print('Datasets: OK')"
```

---

## 🚀 Running the Project

### Quick Start - Train All Models
```bash
python main.py --all
```

This will:
1. Download IMDb dataset (~80MB)
2. Train Custom LSTM (20 epochs)
3. Train AWD-LSTM (15 epochs total)
4. Train BERT (4 epochs)
5. Generate comparative analysis
6. Create visualizations
7. Save all results

**Expected Total Time:**
- CPU: 2-3 hours
- GPU: 30-60 minutes

### Train Individual Models

#### Custom LSTM Only
```bash
python main.py --model custom_lstm
```
- Trains from scratch
- ~20 epochs
- CPU: 30-60 min
- GPU: 10-20 min

#### AWD-LSTM Only
```bash
python main.py --model awd_lstm
```
- Uses pretrained weights
- ~15 epochs total
- CPU: 40-50 min
- GPU: 15-25 min

#### BERT Only
```bash
python main.py --model bert
```
- Fine-tunes pretrained BERT
- ~4 epochs
- CPU: 60-90 min (not recommended)
- GPU: 20-30 min

### Advanced Options

#### Use GPU (if available)
```bash
python main.py --model custom_lstm --device cuda
```

#### Force CPU Usage
```bash
python main.py --model bert --device cpu
```

#### Custom Random Seed
```bash
python main.py --all --seed 123
```

#### Run Evaluation Only (Skip Training)
```bash
python main.py --evaluate
```
*Requires trained models in `saved_models/` directory*

### Command-Line Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `--model` | `custom_lstm`, `awd_lstm`, `bert`, `all` | `all` | Which model to train |
| `--evaluate` | flag | False | Run evaluation only |
| `--seed` | integer | 42 | Random seed |
| `--device` | `cuda`, `cpu`, `auto` | `auto` | Training device |

---

## 🏗️ Project Architecture

### Directory Structure
```
G_P1/
│
├── main.py                    # Main execution script
├── config.py                  # Configuration & hyperparameters
├── requirements.txt           # Python dependencies
├── README.md                  # Project overview
├── PROJECT_GUIDE.md          # This file
├── DOCUMENTATION.md          # Detailed documentation
│
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── custom_lstm.py        # Custom LSTM implementation
│   ├── awd_lstm.py           # AWD-LSTM (ULMFiT)
│   └── bert_model.py         # BERT transformer
│
├── training/                  # Training scripts
│   ├── __init__.py
│   ├── train_custom_lstm.py  # LSTM training
│   ├── train_awd_lstm.py     # AWD-LSTM training
│   └── train_bert.py         # BERT training
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_preprocessing.py # Data loading & preprocessing
│   ├── evaluate.py           # Evaluation metrics
│   └── comparative_analysis.py # Model comparison
│
├── data/                      # Dataset storage (auto-created)
│   ├── imdb/                 # IMDb dataset
│   └── vocabulary.pkl        # Vocabulary file
│
├── saved_models/             # Model checkpoints (auto-created)
│   ├── custom_lstm_best.pt
│   ├── awd_lstm_best.pkl
│   └── bert_best/
│
└── results/                  # Results & visualizations (auto-created)
    ├── custom_lstm_results.pkl
    ├── awd_lstm_results.pkl
    ├── bert_results.pkl
    ├── comparison_results.csv
    ├── comparison_report.txt
    └── plots/
        ├── metrics_comparison.png
        ├── training_time_comparison.png
        ├── convergence_comparison.png
        ├── comprehensive_comparison.png
        └── (confusion matrices, ROC curves, etc.)
```

### Data Flow

```
IMDb Dataset (HuggingFace)
    ↓
Data Preprocessing
    ├── Custom LSTM: Tokenization → Vocabulary → Padding
    ├── AWD-LSTM: fastai DataBlock → Tokenization
    └── BERT: WordPiece Tokenization → Attention Masks
    ↓
Model Training
    ├── Custom LSTM: 20 epochs with early stopping
    ├── AWD-LSTM: LM fine-tuning → Classifier training
    └── BERT: 4 epochs with warmup
    ↓
Evaluation
    ├── Test set predictions
    ├── Metrics calculation
    └── Visualization generation
    ↓
Comparative Analysis
    ├── Performance comparison
    ├── Statistical analysis
    └── Report generation
```

---

## 📊 Expected Results

### Performance Benchmarks

| Model | Accuracy | Precision | Recall | F1-Score | Training Time (GPU) |
|-------|----------|-----------|--------|----------|---------------------|
| Custom LSTM | 85-88% | 0.85-0.88 | 0.84-0.87 | 0.85-0.88 | 10-20 min |
| AWD-LSTM | 88-92% | 0.88-0.92 | 0.87-0.91 | 0.88-0.92 | 15-25 min |
| BERT | 92-94% | 0.92-0.94 | 0.91-0.93 | 0.92-0.94 | 20-30 min |

### Output Files Generated

**Model Checkpoints:**
- `saved_models/custom_lstm_best.pt` - Best Custom LSTM model
- `saved_models/awd_lstm_best.pkl` - Best AWD-LSTM model
- `saved_models/bert_best/` - Best BERT model directory

**Results Files:**
- `results/custom_lstm_results.pkl` - Custom LSTM metrics
- `results/awd_lstm_results.pkl` - AWD-LSTM metrics
- `results/bert_results.pkl` - BERT metrics
- `results/comparison_results.csv` - Comparison table
- `results/comparison_report.txt` - Detailed text report

**Visualizations:**
- Metrics comparison bar charts
- Training/validation loss curves
- Training/validation accuracy curves
- Convergence comparison plots
- ROC curves for all models
- Confusion matrices
- Comprehensive comparison dashboard

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'torch'
```
**Solution:**
```bash
pip install -r requirements.txt
```

#### Issue 2: NumPy/Pandas Compatibility
```
ValueError: numpy.dtype size changed
```
**Solution:**
```bash
pip uninstall numpy pandas -y
pip install numpy==1.24.3 pandas==2.0.3
```

#### Issue 3: CUDA Out of Memory
```
RuntimeError: CUDA out of memory
```
**Solution:**
- Reduce batch size in `config.py`
- Use CPU: `python main.py --device cpu`
- Close other GPU applications

#### Issue 4: Dataset Download Fails
```
ConnectionError: Failed to download dataset
```
**Solution:**
- Check internet connection
- Try again (HuggingFace auto-retry)
- Manual download: https://huggingface.co/datasets/imdb

#### Issue 5: Slow Training on CPU
**Solution:**
- Normal for CPU training (2-3 hours for all models)
- Use GPU if available
- Train individual models instead of all
- Reduce epochs in `config.py`

#### Issue 6: fastai Import Error
```
ImportError: cannot import name 'AWD_LSTM'
```
**Solution:**
```bash
pip install fastai==2.7.12
```
Or skip AWD-LSTM:
```bash
python main.py --model custom_lstm
python main.py --model bert
```

---

## 📚 Additional Resources

### Project Files
- `README.md` - Quick start guide
- `DOCUMENTATION.md` - Detailed technical documentation
- `config.py` - Hyperparameter reference

### Key Concepts
- **LSTM**: Long Short-Term Memory networks for sequence modeling
- **ULMFiT**: Universal Language Model Fine-tuning
- **BERT**: Bidirectional Encoder Representations from Transformers
- **Transfer Learning**: Using pretrained models for new tasks

### References
- IMDb Dataset: https://huggingface.co/datasets/imdb
- PyTorch Documentation: https://pytorch.org/docs/
- HuggingFace Transformers: https://huggingface.co/docs/transformers/
- fastai: https://docs.fast.ai/

---

## 📝 Summary

This project provides a complete, production-ready implementation of sentiment analysis using three different deep learning approaches. It demonstrates:

1. **Classical RNN Approach** (Custom LSTM)
2. **Transfer Learning** (AWD-LSTM with ULMFiT)
3. **Transformer Architecture** (BERT)

All with comprehensive evaluation, comparison, and visualization tools.

### Quick Commands Cheat Sheet
```bash
# Install
pip install -r requirements.txt

# Train all models
python main.py --all

# Train specific model
python main.py --model custom_lstm

# Use GPU
python main.py --device cuda

# Evaluate only
python main.py --evaluate
```

---

**Created by:** NLP Deep Learning Project Team  
**Date:** December 2025  
**Version:** 1.0  
**License:** MIT
