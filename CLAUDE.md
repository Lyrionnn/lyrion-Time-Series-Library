# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TSLib (Time Series Library) is a comprehensive Python library for deep time series analysis. It provides a unified codebase for evaluating and developing time series models across five main tasks: long-term forecasting, short-term forecasting, imputation, anomaly detection, and classification.

## Development Commands

### Running Experiments
All experiments are executed using the main entry point `run.py` with task-specific arguments:

```bash
# Basic experiment command
python run.py \
  --task_name [task_type] \
  --is_training 1 \
  --model [model_name] \
  --data [dataset_name] \
  --model_id [experiment_id] \
  [additional_parameters]
```

### Task Types
- `long_term_forecast` - Long-term time series forecasting
- `short_term_forecast` - Short-term forecasting (M4 dataset)
- `imputation` - Missing value imputation
- `anomaly_detection` - Anomaly detection
- `classification` - Time series classification

### Common Parameters
- `--seq_len` - Input sequence length (default: 96)
- `--pred_len` - Prediction sequence length (default: 96)
- `--batch_size` - Training batch size (default: 32)
- `--learning_rate` - Learning rate (default: 0.0001)
- `--train_epochs` - Number of training epochs (default: 10)
- `--d_model` - Model dimension (default: 512)

### Example Commands
```bash
# Long-term forecasting with TimesNet
python run.py --task_name long_term_forecast --is_training 1 --model TimesNet --data ETTh1 --model_id test

# Classification task
python run.py --task_name classification --is_training 1 --model TimesNet --data UEA --model_id classification_test

# Testing only (no training)
python run.py --task_name long_term_forecast --is_training 0 --model TimesNet --data ETTh1 --model_id test
```

### Using Scripts
Pre-configured experiment scripts are available in the `scripts/` directory:
```bash
# Run classification experiments
bash ./scripts/classification/TimesNet.sh

# Run long-term forecasting
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh

# Run short-term forecasting
bash ./scripts/short_term_forecast/TimesNet_M4.sh
```

## Code Architecture

### Core Components

**Main Entry Point**
- `run.py` - Main experiment runner with argument parsing and task dispatch

**Experiment Classes** (`exp/`)
- `exp_basic.py` - Base experiment class with model registry
- `exp_long_term_forecasting.py` - Long-term forecasting experiments
- `exp_short_term_forecasting.py` - Short-term forecasting experiments
- `exp_imputation.py` - Imputation task experiments
- `exp_anomaly_detection.py` - Anomaly detection experiments
- `exp_classification.py` - Classification task experiments

**Models** (`models/`)
The library includes 35+ time series models including:
- TimesNet, Autoformer, Transformer, DLinear
- iTransformer, PatchTST, TimeXer, TimeMixer
- Mamba (requires `mamba_ssm` package installation)

**Data Handling** (`data_provider/`)
- `data_factory.py` - Data loader factory with dataset registry
- `data_loader.py` - Dataset implementations for various formats
- Supports: ETT datasets, M4, custom CSV data, UEA classification datasets

**Model Layers** (`layers/`)
- Reusable components like attention mechanisms, embeddings, normalization
- Specialized layers for different model architectures

### Debugging and Testing
```bash
# Test single experiment run
python run.py --task_name long_term_forecast --is_training 1 --model TimesNet --data ETTh1 --model_id debug_test --train_epochs 1

# Run with different GPU configurations
python run.py [...args] --use_gpu 0  # CPU only
python run.py [...args] --gpu 0      # Single GPU
python run.py [...args] --use_multi_gpu --devices 0,1  # Multi-GPU
```

### Model Development Workflow
1. **Adding New Models**: Create model in `models/`, register in `exp_basic.py:model_dict`, create experiment scripts
2. **Custom Datasets**: Add data loader to `data_provider/data_loader.py`, register in `data_factory.py`
3. **Architecture Pattern**: Models inherit from `nn.Module`, implement `forward()` method with standardized input/output shapes
4. **Experiment Integration**: Each task has dedicated experiment class in `exp/` that handles training/validation loops

## Core Architecture Patterns

### Task-Experiment Mapping
The codebase uses a task-based architecture where each time series task has a dedicated experiment class:
- `run.py` → Task dispatcher and argument parser
- `exp_*.py` → Task-specific training/evaluation logic  
- `models/` → Model implementations (35+ models)
- `data_provider/` → Unified data loading interface
- `layers/` → Reusable model components

### Model Registry System
Models are registered in `exp_basic.py:model_dict` for dynamic instantiation. Special handling exists for models with external dependencies (e.g., Mamba requires `mamba_ssm`).

### Data Pipeline Architecture  
- `data_factory.py` provides unified data loader interface across tasks
- `data_loader.py` implements dataset-specific preprocessing
- Automatic feature engineering and normalization per task type
- Support for time features, target encoding, and sequence padding

### Dataset Structure
- Training data should be placed in `./dataset/` directory
- Supports multiple formats: ETT (electricity), M4 (forecasting competition), custom CSV, UEA (classification)
- Data preprocessing is handled automatically by data loaders

### GPU Configuration
- Supports CUDA, MPS (Apple Silicon), and CPU execution
- Multi-GPU training available with `--use_multi_gpu --devices 0,1,2,3`
- Automatic device selection based on availability

## Environment Setup

### Installation
```bash
# Install Python 3.8+ and dependencies
pip install -r requirements.txt

# For Mamba model support (optional)
pip install mamba_ssm
```

### Data Setup
Download datasets from [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2) or [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) and place in `./dataset/` directory.

### Development Environment
- Python 3.8+
- PyTorch 1.7.1+
- CUDA support (optional, auto-detected)
- MPS support for Apple Silicon (auto-detected)

## Important Notes

- Fixed random seed (2021) is set for reproducibility
- Checkpoints are saved to `./checkpoints/` directory
- Results and logs are generated automatically during training
- The codebase uses a unified interface across all tasks for consistency
- Model performance varies by task - refer to README leaderboard for benchmarks