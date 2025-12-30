"""
Configuration file for Sentiment Analysis Project
Contains all hyperparameters and settings for LSTM, AWD-LSTM, and BERT models
"""

import torch
import os

# ============================================================================
# GENERAL SETTINGS
# ============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ============================================================================
# DATASET SETTINGS
# ============================================================================

DATASET_NAME = 'imdb'
MAX_SEQUENCE_LENGTH = 500  # Maximum length for padding/truncation
TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

# ============================================================================
# CUSTOM LSTM SETTINGS
# ============================================================================

CUSTOM_LSTM_CONFIG = {
    'vocab_size': 25000,  # Will be updated after tokenization
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

# ============================================================================
# AWD-LSTM (ULMFiT) SETTINGS
# ============================================================================

AWD_LSTM_CONFIG = {
    'base_arch': 'AWD_LSTM',
    'pretrained': True,
    'batch_size': 32,
    'learning_rate_lm': 1e-3,  # Language model fine-tuning
    'learning_rate_classifier': 1e-2,  # Classifier training
    'num_epochs_lm': 5,  # Language model fine-tuning epochs
    'num_epochs_classifier': 10,  # Classifier training epochs
    'dropout_mult': 0.5,
    'max_length': 500,
}

# ============================================================================
# BERT SETTINGS
# ============================================================================

BERT_CONFIG = {
    'model_name': 'bert-base-uncased',
    'max_length': 512,
    'batch_size': 16,  # Smaller batch size due to memory constraints
    'learning_rate': 2e-5,
    'num_epochs': 4,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'gradient_accumulation_steps': 2,
}

# ============================================================================
# TRAINING SETTINGS
# ============================================================================

EARLY_STOPPING_PATIENCE = 3
SAVE_BEST_MODEL = True
LOG_INTERVAL = 100  # Log every N batches

# ============================================================================
# EVALUATION SETTINGS
# ============================================================================

EVAL_BATCH_SIZE = 128
CONFUSION_MATRIX = True
CLASSIFICATION_REPORT = True
SAVE_PREDICTIONS = True

# ============================================================================
# COMPARATIVE ANALYSIS SETTINGS
# ============================================================================

PLOT_TRAINING_CURVES = True
PLOT_COMPARISON_CHARTS = True
GENERATE_REPORT = True

# Model names for comparison
MODEL_NAMES = ['Custom LSTM', 'AWD-LSTM', 'BERT']

# ============================================================================
# MODEL SAVE PATHS
# ============================================================================

CUSTOM_LSTM_PATH = os.path.join(MODELS_DIR, 'custom_lstm_best.pt')
AWD_LSTM_PATH = os.path.join(MODELS_DIR, 'awd_lstm_best.pkl')
BERT_PATH = os.path.join(MODELS_DIR, 'bert_best')

# ============================================================================
# RESULTS PATHS
# ============================================================================

RESULTS_CSV = os.path.join(RESULTS_DIR, 'comparison_results.csv')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)
