# Project Documentation

## Sentiment Analysis Using LSTM and Transformer Models

### Project Overview

This project implements and compares three state-of-the-art deep learning architectures for sentiment analysis on the IMDb movie reviews dataset:

1. **Custom LSTM** - Recurrent neural network built from scratch
2. **AWD-LSTM (ULMFiT)** - Pretrained language model with transfer learning
3. **BERT** - Transformer-based architecture with self-attention

### Key Features

- Complete implementation of three different architectures
- Comprehensive evaluation metrics (Accuracy, Precision, Recall, F1-Score)
- Comparative analysis with visualizations
- Modular and extensible codebase
- Reproducible results with fixed random seeds

### Installation

```bash
# Clone or download the project
cd G_P1

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# Train all models and run comparative analysis
python main.py --all

# Train a specific model
python main.py --model custom_lstm
python main.py --model awd_lstm
python main.py --model bert

# Run evaluation only (requires trained models)
python main.py --evaluate
```

### Project Structure

```
G_P1/
├── main.py                    # Main execution script
├── config.py                  # Configuration and hyperparameters
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
│
├── models/                    # Model architectures
│   ├── __init__.py
│   ├── custom_lstm.py        # Custom LSTM implementation
│   ├── awd_lstm.py           # AWD-LSTM (ULMFiT) implementation
│   └── bert_model.py         # BERT implementation
│
├── training/                  # Training scripts
│   ├── __init__.py
│   ├── train_custom_lstm.py  # Custom LSTM training
│   ├── train_awd_lstm.py     # AWD-LSTM training
│   └── train_bert.py         # BERT training
│
├── utils/                     # Utility functions
│   ├── __init__.py
│   ├── data_preprocessing.py # Data loading and preprocessing
│   ├── evaluate.py           # Evaluation metrics
│   └── comparative_analysis.py # Model comparison
│
├── data/                      # Dataset storage (auto-created)
├── saved_models/             # Trained model checkpoints (auto-created)
└── results/                  # Results and plots (auto-created)
    └── plots/
```

### Models

#### 1. Custom LSTM
- Bidirectional LSTM with 2 layers
- 300-dimensional word embeddings
- 256 hidden units
- Trained from scratch on task-specific data
- ~20 epochs to convergence

**Architecture:**
```
Embedding → BiLSTM → Dropout → FC → ReLU → Dropout → Output
```

#### 2. AWD-LSTM (ULMFiT)
- Uses pretrained language model
- Fine-tuned on IMDb domain
- Transfer learning approach
- Faster convergence (~10 epochs)
- Better generalization

**Training Stages:**
1. Language model fine-tuning
2. Classifier training with gradual unfreezing

#### 3. BERT
- Transformer architecture with self-attention
- Pretrained on large text corpus
- Fine-tuned on sentiment classification
- Best accuracy but computationally expensive
- ~4 epochs sufficient

**Features:**
- WordPiece tokenization
- [CLS] token for classification
- Attention masks for variable-length inputs

### Dataset

**IMDb Movie Reviews**
- 50,000 movie reviews
- Binary sentiment labels (positive/negative)
- 25,000 training samples
- 25,000 test samples
- Average review length: ~200 words

### Hyperparameters

All hyperparameters are defined in `config.py`:

**Custom LSTM:**
- Vocabulary: 25,000 words
- Embedding dim: 300
- Hidden dim: 256
- Batch size: 64
- Learning rate: 0.001
- Epochs: 20

**AWD-LSTM:**
- Batch size: 32
- LM learning rate: 0.001
- Classifier learning rate: 0.01
- LM epochs: 5
- Classifier epochs: 10

**BERT:**
- Model: bert-base-uncased
- Max length: 512
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 4
- Warmup steps: 500

### Evaluation Metrics

All models are evaluated using:
- **Accuracy** - Overall correctness
- **Precision** - Positive predictive value
- **Recall** - True positive rate
- **F1-Score** - Harmonic mean of precision and recall
- **ROC-AUC** - Area under ROC curve
- **Confusion Matrix** - Detailed error analysis
- **Training Time** - Computational efficiency
- **Convergence** - Epochs to best performance

### Results

Expected performance ranges:

| Model | Accuracy | F1-Score | Training Time |
|-------|----------|----------|---------------|
| Custom LSTM | 85-88% | 0.85-0.88 | ~60 min (CPU) |
| AWD-LSTM | 88-92% | 0.88-0.92 | ~45 min (CPU) |
| BERT | 92-94% | 0.92-0.94 | ~30 min (GPU) |

### Visualizations

The project generates several comparison plots:

1. **Metrics Comparison** - Bar charts for all metrics
2. **Training Time** - Computational efficiency comparison
3. **Convergence Curves** - Loss and accuracy over epochs
4. **ROC Curves** - Model discrimination ability
5. **Confusion Matrices** - Per-model error analysis
6. **Comprehensive Dashboard** - All metrics in one view

All plots are saved in `results/plots/`

### Configuration

Edit `config.py` to modify:
- Model architectures
- Hyperparameters
- Batch sizes
- Learning rates
- File paths
- Device settings (GPU/CPU)

### Usage Examples

**Train Custom LSTM only:**
```bash
python main.py --model custom_lstm --device cuda
```

**Train with custom seed:**
```bash
python main.py --seed 123
```

**Force CPU usage:**
```bash
python main.py --device cpu
```

**Analyze existing results:**
```bash
python main.py --evaluate
```

### Requirements

**Core Dependencies:**
- Python 3.8+
- PyTorch 2.0+
- Transformers (HuggingFace)
- fastai (for AWD-LSTM)
- datasets (HuggingFace)
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- tqdm

**Hardware:**
- CPU: Sufficient for all models (slower)
- GPU: Recommended for BERT
- RAM: 8GB minimum, 16GB recommended
- Storage: ~5GB for datasets and models

### Troubleshooting

**Issue: CUDA out of memory**
- Reduce batch size in `config.py`
- Use CPU instead: `--device cpu`
- Enable gradient accumulation

**Issue: fastai import error**
- AWD-LSTM requires fastai: `pip install fastai`
- Or skip AWD-LSTM: `python main.py --model custom_lstm`

**Issue: Dataset download fails**
- Check internet connection
- HuggingFace datasets downloads automatically
- Manual download: https://huggingface.co/datasets/imdb

**Issue: Slow training**
- Use GPU if available
- Reduce number of epochs
- Reduce batch size
- Use smaller models

### Extensions

Possible improvements:
1. Add attention mechanism to LSTM
2. Try other pretrained models (RoBERTa, DistilBERT)
3. Ensemble methods
4. Cross-validation
5. Hyperparameter tuning
6. Multi-class sentiment (3+ classes)
7. Aspect-based sentiment analysis

### Business Applications

- **E-commerce**: Product review analysis
- **Social Media**: Brand sentiment monitoring
- **Customer Service**: Ticket priority classification
- **Market Research**: Consumer opinion mining
- **Content Moderation**: Negative content detection

### References

- ULMFiT: Howard & Ruder (2018)
- BERT: Devlin et al. (2018)
- IMDb Dataset: Maas et al. (2011)
- PyTorch: https://pytorch.org/
- HuggingFace: https://huggingface.co/

### License

MIT License

### Authors

NLP Deep Learning Project - 2025

### Contact

For questions or issues, please refer to the project documentation.

---

**Note**: This is an educational project demonstrating comparative analysis of sentiment analysis architectures. Results may vary based on hardware, random seeds, and hyperparameter choices.
