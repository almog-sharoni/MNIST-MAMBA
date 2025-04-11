# MNIST Mamba

A high-performance implementation of the Mamba SSM (State Space Model) architecture for MNIST digit classification, featuring PyTorch training and optimized C inference.

## Overview

This project implements the Mamba architecture for handwritten digit recognition using the MNIST dataset. It features:
- PyTorch-based training pipeline
- Efficient C implementation for inference
- Model export functionality for trained models
- OpenMP support for parallel processing

## Project Structure

```
├── models/
│   └── mamba_model.py     # PyTorch Mamba model implementation
├── utils/
│   └── data_utils.py      # Data loading utilities
├── train.py               # Training script
├── export.py             # Model export utilities
├── mamba.c              # C inference implementation
├── config.json          # Model configuration
└── makefile            # Build system
```

## Requirements

Python dependencies:
```
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
mamba-ssm>=1.0.0
tqdm
matplotlib
```

System requirements:
- GCC or compatible C compiler
- OpenMP (optional, for parallel processing)
- CMake (for building)

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Build the C inference engine:
```bash
make              # Basic build
make omp         # Build with OpenMP support
make debug       # Debug build
```

## Usage

### Training

Train a new model:
```bash
python train.py
```

Training options:
- `batch_size`: Batch size (default: 128)
- `epochs`: Number of training epochs (default: 10)
- `learning_rate`: Learning rate (default: 0.001)
- `save_dir`: Directory to save checkpoints (default: 'checkpoints')

### Model Export

Export a trained model for C inference:
```bash
python export.py model --checkpoint checkpoints/best_model.pth --output mnist_mamba.bin
```

Export a test image:
```bash
python export.py image --index 0 --output input.bin
```

### Inference

Run inference using the C implementation:
```bash
./mamba mnist_mamba.bin input.bin
```

## Model Architecture

The implementation uses a Mamba-based architecture with the following key components:

- Input size: 784 (28x28 MNIST images)
- Selective SSM mechanism
- Configurable number of layers and dimensions
- RMSNorm for layer normalization
- Cross-entropy loss for classification

Key hyperparameters (configurable in config.json):
- Hidden dimension size
- Number of layers
- State dimension
- Convolution kernel size
- Delta (timestep) parameters

## Performance

The model achieves competitive accuracy on MNIST digit classification while maintaining efficient inference in C. The C implementation includes optimizations for:
- Memory mapping for efficient weight loading
- OpenMP parallelization support
- Efficient matrix operations
- Minimal memory allocations during inference

## Build Options

The makefile provides several build targets:
- `make`: Standard optimized build
- `make omp`: Build with OpenMP support
- `make debug`: Debug build with symbols
- `make fast`: Maximum optimization (-Ofast)
- `make clean`: Clean build artifacts

## License

[Add your license information here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation is based on the Mamba architecture paper:
[Add paper reference here]