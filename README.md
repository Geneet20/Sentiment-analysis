# Sentiment Analysis Using LSTM and Transformer Models

A comprehensive comparison of sentiment analysis models: Custom LSTM, AWD-LSTM (ULMFiT), and BERT.

## Project Overview

This project implements and compares three different deep learning architectures for sentiment analysis:
1. **Custom LSTM**: Built from scratch in PyTorch
2. **AWD-LSTM (ULMFiT)**: Pretrained language model with fine-tuning
3. **BERT**: Transformer-based model using HuggingFace

## Dataset

**IMDb Movie Reviews Dataset**
- 50,000 labeled movie reviews
- Binary sentiment labels (positive/negative)
- Split: 25,000 training, 25,000 testing

## Project Structure

```
G_P1/
├── data/                      # Dataset storage
├── models/                    # Model architectures
│   ├── custom_lstm.py
│   ├── awd_lstm.py
│   └── bert_model.py
├── training/                  # Training scripts
│   ├── train_custom_lstm.py
│   ├── train_awd_lstm.py
│   └── train_bert.py
├── utils/                     # Utilities
│   ├── data_preprocessing.py
│   ├── evaluate.py
│   └── comparative_analysis.py
├── saved_models/             # Trained model checkpoints
├── results/                  # Evaluation results and plots
├── config.py                 # Configuration parameters
├── main.py                   # Main execution script
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Run Complete Pipeline
```bash
python main.py --all
```

### Train Individual Models
```bash
# Custom LSTM
python main.py --model custom_lstm

# AWD-LSTM
python main.py --model awd_lstm

# BERT
python main.py --model bert
```

### Evaluation Only
```bash
python main.py --evaluate
```

## Configuration

Edit `config.py` to modify:
- Hyperparameters (learning rate, batch size, epochs)
- Model architectures
- Data preprocessing settings
- Paths and random seeds

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Training/Validation Loss
- Epochs to Convergence
- Training Time

## Expected Results

| Model | Accuracy | Training Time | Convergence |
|-------|----------|---------------|-------------|
| Custom LSTM | ~85% | Slowest | Slow |
| AWD-LSTM | ~90% | Medium | Fast |
| BERT | ~93%+ | GPU required | Fastest |

## Business Applications

- Customer sentiment analysis
- Social media opinion mining
- Brand reputation monitoring
- E-commerce feedback analysis
- Call-center transcript classification

## Technical Stack

- **Framework**: PyTorch
- **Pretrained Models**: fastai (ULMFiT), HuggingFace Transformers (BERT)
- **Dataset**: IMDb via HuggingFace datasets
- **Evaluation**: scikit-learn, matplotlib, seaborn

## Timeline

- **Day 1**: Dataset preprocessing and exploration
- **Day 2**: Custom LSTM implementation and training
- **Day 3**: AWD-LSTM fine-tuning and BERT implementation
- **Day 4**: Evaluation and comparative analysis

## Authors

NLP Deep Learning Project - 2025

## License

MIT License
