# LLM Pretraining

A PyTorch-based implementation for pretraining large language models with a custom transformer architecture featuring Rotary Position Embeddings (RoPE) and optimized training workflows.

## Overview

This project provides a complete pipeline for pretraining transformer-based language models, including:
- Custom transformer architecture with multi-head attention and RoPE
- Training with mixed precision and gradient checkpointing
- Inference with top-k sampling
- Dataset tokenization and preprocessing
- Experiment tracking with Weights & Biases
- Checkpoint management for resuming training

## Features

- **Advanced Attention Mechanism**: Multi-head attention with Rotary Position Embeddings for better positional awareness
- **Efficient Training**: Mixed precision training with gradient accumulation and gradient clipping
- **Checkpoint Management**: Automatic checkpointing and ability to resume from saved states
- **Experiment Tracking**: Integration with Weights & Biases for monitoring training metrics
- **Flexible Configuration**: Centralized config system for easy hyperparameter tuning
- **Inference Pipeline**: Sampling-based text generation with temperature and top-k controls

## Project Structure

```
llm_pretraining/
├── main.py                 # Entry point
├── train.py               # Training loop with validation
├── infer.py               # Inference and text generation
├── model.py               # Transformer architecture implementation
├── config.py              # Hyperparameters and configuration
├── dataset.py             # Dataset loading and preprocessing
├── helpers.py             # Utility functions for logging and model saving
├── tokenizer.json         # Tokenizer configuration (GPT-2)
├── rope.py                # Rotary Position Embeddings implementation
├── README.md              # This file
├── data_processing/       # Data preparation pipeline
│   ├── dl_dataset.py      # Dataset downloading
│   ├── tokenize.py        # Text tokenization
│   └── create_tok_data.py # Tokenized data creation
└── model_checkpoints/     # Saved model checkpoints
```

## Configuration

Edit `config.py` to customize hyperparameters:

- **Vocab & Sequence**: `N_VOCAB=50257` (GPT-2 vocab), `SEQ_LEN=1024`
- **Training**: `N_EPOCHS=2`, `BATCH_SIZE=16`, `GRAD_ACCUMULATION_STEPS=8`
- **Optimization**: `OPTIM_LR=1e-4`, `GRAD_CLIP=1.0`
- **Model Architecture**:
  - `d_emb=768` (embedding dimension)
  - `n_heads=12` (attention heads)
  - `d_val=64`, `d_qk=64` (attention dimensions)
  - `n_blocks=10` (transformer blocks)

## Installation

```bash
# Clone the repository
git clone https://github.com/xyphoes0727/llm-pretraining.git

cd llm_pretraining

# Install dependencies
pip install -r requirements.txt

# Set up environment variables (for Weights & Biases)
echo "WANDB_KEY=your_wandb_key" > .env
```

## Usage

### Data Preparation

```bash
# 1. Download and prepare dataset
python data_processing/dl_dataset.py

# 2. Tokenize the data
python data_processing/tokenize.py

# 3. Create tokenized dataset splits
python data_processing/create_tok_data.py
```

### Training

```bash
# Start training from scratch
python train.py

# Resume from a checkpoint (set RESUME_FROM_CHECKPOINT=True in config.py)
python train.py
```

Monitor training progress using WandB

### Inference

Generate text using a trained model:

```bash
# With command-line arguments
python infer.py \
    --prompt "Once upon a time" \
    --checkpoint /path/to/checkpoint.pt \
    --length 100 \
    --temperature 0.7 \
    --top_k 50

# Or use default checkpoint from config.py
python infer.py --prompt "Your text here"
```

## Model Architecture

The transformer architecture includes:

- **Token Embedding**: Maps tokens to embedding space (vocab_size → d_emb)
- **Rotary Position Embeddings**: RoPE for efficient absolute position encoding
- **Multi-Head Attention**: 
  - Parallel attention heads for capturing different representation subspaces
  - Masking and dropout for regularization
- **Feed-Forward Networks**: Dense layers with activation functions
- **Layer Normalization**: Stabilizes training
- **Residual Connections**: Facilitates gradient flow

## Training Details

The training pipeline includes:

1. **Distributed Loss Calculation**: Cross-entropy loss on next-token prediction
2. **Gradient Accumulation**: Larger effective batch size without memory overflow
3. **Mixed Precision**: Automatic mixed precision for faster training with lower memory
4. **Checkpointing**: Periodic model snapshots every N steps
5. **Evaluation**: Validation loss computed on test set
6. **Logging**: Training metrics logged to console and Weights & Biases

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Datasets library
- Tokenizers
- Weights & Biases
- python-dotenv