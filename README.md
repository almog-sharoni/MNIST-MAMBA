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

# MNIST Mamba Evaluation

This project contains tools for evaluating a Mamba model on the MNIST dataset.

## Evaluation

To evaluate a trained model on the MNIST test set, use the `evaluate.py` script:

```bash
python evaluate.py <model.bin>
```

For example:
```bash
python evaluate.py mnist_mamba.bin
```

The script will:
- Download the MNIST test dataset if not present
- Process all 10,000 test images
- Show a progress bar with current accuracy
- Print final results including:
  - Overall accuracy
  - Number of correct predictions
  - Total evaluation time
  - Average time per image

### Requirements
- Python with torchvision and numpy
- Compiled mamba executable in current directory
- Valid model.bin file