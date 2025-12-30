"""
Main Execution Script
Orchestrates training and evaluation of all sentiment analysis models
"""

import argparse
import os
import sys
import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from utils.comparative_analysis import analyze_all_models


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_custom_lstm_model():
    """Train Custom LSTM model"""
    print("\n" + "=" * 80)
    print("TRAINING CUSTOM LSTM MODEL")
    print("=" * 80)
    
    try:
        from training.train_custom_lstm import train_custom_lstm
        results, model, trainer = train_custom_lstm()
        return results
    except Exception as e:
        print(f"Error training Custom LSTM: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_awd_lstm_model():
    """Train AWD-LSTM (ULMFiT) model"""
    print("\n" + "=" * 80)
    print("TRAINING AWD-LSTM (ULMFiT) MODEL")
    print("=" * 80)
    
    try:
        from training.train_awd_lstm import train_awd_lstm
        results, classifier = train_awd_lstm()
        return results
    except Exception as e:
        print(f"Error training AWD-LSTM: {e}")
        print("\nNote: AWD-LSTM requires fastai library.")
        print("You can skip this model if fastai is not available.")
        import traceback
        traceback.print_exc()
        return None


def train_bert_model():
    """Train BERT model"""
    print("\n" + "=" * 80)
    print("TRAINING BERT MODEL")
    print("=" * 80)
    
    try:
        from training.train_bert import train_bert
        results, model, trainer = train_bert()
        return results
    except Exception as e:
        print(f"Error training BERT: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_comparative_analysis():
    """Run comparative analysis on all trained models"""
    print("\n" + "=" * 80)
    print("RUNNING COMPARATIVE ANALYSIS")
    print("=" * 80)
    
    try:
        df, results = analyze_all_models(config.RESULTS_DIR, config.PLOTS_DIR)
        return df, results
    except Exception as e:
        print(f"Error in comparative analysis: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Sentiment Analysis: Custom LSTM vs AWD-LSTM vs BERT'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['custom_lstm', 'awd_lstm', 'bert', 'all'],
        default='all',
        help='Which model to train (default: all)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Run evaluation and comparison only (skip training)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cuda', 'cpu', 'auto'],
        default='auto',
        help='Device to use for training (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    config.RANDOM_SEED = args.seed
    
    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    config.DEVICE = torch.device(device)
    
    print("=" * 80)
    print("SENTIMENT ANALYSIS PROJECT")
    print("Custom LSTM vs AWD-LSTM (ULMFiT) vs BERT")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Random Seed: {config.RANDOM_SEED}")
    print(f"Results Directory: {config.RESULTS_DIR}")
    print("=" * 80)
    
    if args.evaluate:
        # Run evaluation only
        print("\nRunning evaluation and comparative analysis only...")
        run_comparative_analysis()
    else:
        # Train models
        if args.model == 'all':
            # Train all models
            print("\nTraining all models sequentially...\n")
            
            # 1. Custom LSTM
            custom_lstm_results = train_custom_lstm_model()
            
            # 2. AWD-LSTM
            awd_lstm_results = train_awd_lstm_model()
            
            # 3. BERT
            bert_results = train_bert_model()
            
            # Run comparative analysis
            print("\n" + "=" * 80)
            print("ALL MODELS TRAINED - RUNNING COMPARATIVE ANALYSIS")
            print("=" * 80)
            
            run_comparative_analysis()
            
        elif args.model == 'custom_lstm':
            train_custom_lstm_model()
            
        elif args.model == 'awd_lstm':
            train_awd_lstm_model()
            
        elif args.model == 'bert':
            train_bert_model()
    
    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)
    print(f"\nResults saved in: {config.RESULTS_DIR}")
    print(f"Plots saved in: {config.PLOTS_DIR}")
    print(f"Models saved in: {config.MODELS_DIR}")
    print("\nTo run comparative analysis on existing results:")
    print("  python main.py --evaluate")
    print("\nTo train a specific model:")
    print("  python main.py --model custom_lstm")
    print("  python main.py --model awd_lstm")
    print("  python main.py --model bert")
    print("=" * 80)


if __name__ == '__main__':
    main()
