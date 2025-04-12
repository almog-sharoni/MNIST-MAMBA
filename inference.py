import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
import os
import numpy as np
from models.mamba_model import MambaMNIST
from utils.data_utils import get_mnist_loaders

def inference(model, test_loader, device):
    """
    Perform inference on the test dataset
    
    Args:
        model: Trained MambaMNIST model
        test_loader: DataLoader for test dataset
        device: Device to run inference on
    
    Returns:
        accuracy: Overall accuracy on test set
        predictions: List of predictions for each image
        true_labels: List of true labels
    """
    model.eval()
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Running inference"):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and true labels
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = 100 * correct / total
    return accuracy, predictions, true_labels

def debug_model_weights(model, output_file="weights_python.txt"):
    """Debug model weights by writing them to a file"""
    with open(output_file, "w") as f:
        # Input projection
        f.write("=== Input Projection ===\n")
        weights = model.input_proj.weight.detach().numpy()
        f.write(f"Shape: {weights.shape}\n")
        f.write(f"Mean: {weights.mean():.6f}, Std: {weights.std():.6f}\n")
        f.write(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}\n")
        f.write("\n")

        # Mamba layers
        for i, layer_dict in enumerate(model.layers):
            f.write(f"=== Layer {i} ===\n")
            
            # Norm weights
            norm_weights = layer_dict['norm'].weight.detach().numpy()
            f.write(f"Norm weights - Shape: {norm_weights.shape}\n")
            f.write(f"Mean: {norm_weights.mean():.6f}, Std: {norm_weights.std():.6f}\n")
            f.write(f"Min: {norm_weights.min():.6f}, Max: {norm_weights.max():.6f}\n")
            
            # Mamba layer weights
            mamba = layer_dict['mamba']
            for name, param in mamba.named_parameters():
                if param.requires_grad:
                    weights = param.detach().numpy()
                    f.write(f"\n{name} - Shape: {weights.shape}\n")
                    f.write(f"Mean: {weights.mean():.6f}, Std: {weights.std():.6f}\n")
                    f.write(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}\n")
            f.write("\n")

        # Final norm
        f.write("=== Final Norm ===\n")
        weights = model.norm_f.weight.detach().numpy()
        f.write(f"Shape: {weights.shape}\n")
        f.write(f"Mean: {weights.mean():.6f}, Std: {weights.std():.6f}\n")
        f.write(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}\n")
        f.write("\n")

        # Output head
        f.write("=== Output Head ===\n")
        weights = model.output_head.weight.detach().numpy()
        f.write(f"Shape: {weights.shape}\n")
        f.write(f"Mean: {weights.mean():.6f}, Std: {weights.std():.6f}\n")
        f.write(f"Min: {weights.min():.6f}, Max: {weights.max():.6f}\n")

def print_model_weights(model, output_file="weights_debug_python.txt"):
    """Print model weights to a file for debugging"""
    with open(output_file, "w") as f:
        # Input projection
        f.write("=== Input Projection ===\n")
        w = model.input_proj.weight.detach().numpy()
        f.write(f"Weight shape: {w.shape}\n")
        # Print first few weights as sample
        f.write("First 10 weights of first 3 rows:\n")
        for i in range(min(3, w.shape[0])):
            f.write(f"Row {i}: {w[i, :10]}\n")
        f.write("\n")

        # Mamba layers
        for i, layer_dict in enumerate(model.layers):
            f.write(f"\n=== Layer {i} ===\n")
            mamba = layer_dict['mamba']
            
            # Print in_proj weights
            w = mamba.in_proj.weight.detach().numpy()
            f.write(f"\nin_proj weight shape: {w.shape}\n")
            f.write("First 10 weights of first 3 rows:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Row {j}: {w[j, :10]}\n")
            
            # Print conv1d weights
            w = mamba.conv1d.weight.detach().numpy()
            f.write(f"\nconv1d weight shape: {w.shape}\n")
            f.write("First 10 weights of first 3 channels:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Channel {j}: {w[j, 0, :min(10, w.shape[2])]}\n")
            
            # Print x_proj weights
            w = mamba.x_proj.weight.detach().numpy()
            f.write(f"\nx_proj weight shape: {w.shape}\n")
            f.write("First 10 weights of first 3 rows:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Row {j}: {w[j, :10]}\n")
            
            # Print dt_proj weights
            w = mamba.dt_proj.weight.detach().numpy()
            f.write(f"\ndt_proj weight shape: {w.shape}\n")
            f.write("First 10 weights of first 3 rows:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Row {j}: {w[j, :10]}\n")
            
            # Print A_log
            w = mamba.A_log.detach().numpy()
            f.write(f"\nA_log shape: {w.shape}\n")
            f.write("First 10 weights of first 3 rows:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Row {j}: {w[j, :10]}\n")
            
            # Print D
            w = mamba.D.detach().numpy()
            f.write(f"\nD shape: {w.shape}\n")
            f.write("First 10 values:\n")
            f.write(f"{w[:10]}\n")
            
            # Print out_proj weights
            w = mamba.out_proj.weight.detach().numpy()
            f.write(f"\nout_proj weight shape: {w.shape}\n")
            f.write("First 10 weights of first 3 rows:\n")
            for j in range(min(3, w.shape[0])):
                f.write(f"Row {j}: {w[j, :10]}\n")
            f.write("\n")

        # Output head
        w = model.output_head.weight.detach().numpy()
        f.write("\n=== Output Head ===\n")
        f.write(f"Weight shape: {w.shape}\n")
        f.write("First 10 weights of first 3 rows:\n")
        for i in range(min(3, w.shape[0])):
            f.write(f"Row {i}: {w[i, :10]}\n")

def main():
    parser = argparse.ArgumentParser(description='Run inference on MNIST test set using trained Mamba model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load data
    _, test_loader = get_mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Load model
    print(f'Loading model from {args.checkpoint}')
    model, _ = MambaMNIST.load_from_checkpoint(args.checkpoint, map_location=device)
    
    # Print model weights before moving to GPU
    print_model_weights(model)
    
    # Move model to device
    model = model.to(device)
    
    # Run inference
    accuracy, predictions, true_labels = inference(model, test_loader, device)
    
    print(f'\nTest Accuracy: {accuracy:.2f}%')
    
    # Save results
    results = {
        'accuracy': accuracy,
        'predictions': predictions,
        'true_labels': true_labels
    }
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save results
    output_file = 'results/inference_results.npz'
    np.savez(output_file, **results)
    print(f'\nResults saved to {output_file}')

if __name__ == '__main__':
    main()