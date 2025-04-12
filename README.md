# MNIST Mamba Model

This repository contains a C implementation of the Mamba SSM model for MNIST digit classification.

## Export Tools

The `export.py` script provides commands to prepare data for the C model:

### Export Trained Model
Convert a PyTorch checkpoint to binary format for C inference:
```bash
python export.py model [--checkpoint PATH] [--output mnist_mamba.bin]
```

### Export Single Test Image
Export a single MNIST test image for inference:
```bash
python export.py image [--index N] [--output input.bin]
```

## C Model Usage

### Single Image Inference
```bash
./mamba model.bin input.bin
```

## Model Architecture

- Input size: 784 (28x28 grayscale image)
- Output: 10 classes (digits 0-9)
- Uses Mamba SSM layers with RMSNorm and residual connections