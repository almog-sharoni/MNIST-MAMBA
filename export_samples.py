#!/usr/bin/env python3
import torch
import numpy as np
import argparse
import os
import struct
from pathlib import Path
import matplotlib.pyplot as plt
from utils.data_utils import get_mnist_loaders

def export_mnist_sample(index=0, output_path="input.raw", visualize=False):
    """
    Export a single MNIST image as raw float32 data for C inference
    
    Args:
        index: Index of the image to export from the test set
        output_path: Path to save the raw data
        visualize: Whether to visualize the image
    """
    # Get test loader with batch size 1 for easy indexing
    _, test_loader = get_mnist_loaders(batch_size=1, num_workers=0)
    
    # Convert to list for easy indexing
    test_data = []
    test_labels = []
    for images, labels in test_loader:
        test_data.append(images)
        test_labels.append(labels)
    
    # Ensure index is valid
    if index >= len(test_data):
        print(f"Error: index {index} out of range, max is {len(test_data)-1}")
        return
    
    # Get the image and label
    image = test_data[index][0]  # Shape: [1, 28, 28]
    label = test_labels[index].item()
    
    # Reshape to row-wise format expected by the C model
    # Each row becomes a feature vector (28 pixels per row)
    image = image.reshape(28, 28).float()
    
    # Visualize if requested
    if visualize:
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.title(f"MNIST Digit: {label}")
        plt.savefig(f"{output_path}.png")
        plt.show()
        print(f"Image saved as {output_path}.png")
    
    # Save raw data as binary file with float32 values
    with open(output_path, 'wb') as f:
        # Write as row-wise float32 values
        # C model expects rows as inputs (28 features per row)
        image_np = image.numpy().astype(np.float32)
        f.write(image_np.tobytes())
    
    print(f"Exported MNIST digit {label} to {output_path}")
    print(f"Shape: {image.shape}, Size: {image.numel()} float32 values")
    print(f"File size: {os.path.getsize(output_path)} bytes")
    
    return label

def export_mnist_samples(num_samples=10, output_dir="samples"):
    """
    Export multiple MNIST samples for testing
    
    Args:
        num_samples: Number of samples to export
        output_dir: Directory to save the samples
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Export samples
    labels = []
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"sample_{i}.raw")
        label = export_mnist_sample(index=i, output_path=output_path, visualize=True)
        labels.append(label)
    
    # Save labels as text file
    with open(os.path.join(output_dir, "labels.txt"), 'w') as f:
        for i, label in enumerate(labels):
            f.write(f"sample_{i}.raw: {label}\n")
    
    print(f"Exported {num_samples} samples to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export MNIST samples as raw data for C inference')
    parser.add_argument('--index', type=int, default=0, help='Index of the image to export')
    parser.add_argument('--output', type=str, default="input.raw", help='Output path for the raw data')
    parser.add_argument('--visualize', action='store_true', help='Visualize the image')
    parser.add_argument('--batch', action='store_true', help='Export multiple samples')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to export in batch mode')
    parser.add_argument('--output-dir', type=str, default="samples", help='Output directory for batch mode')
    
    args = parser.parse_args()
    
    if args.batch:
        export_mnist_samples(args.num_samples, args.output_dir)
    else:
        export_mnist_sample(args.index, args.output, args.visualize)