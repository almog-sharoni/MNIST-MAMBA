#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.serialization
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
import time
import copy

from models.mamba_model import MambaMNIST, MambaConfig, MambaLayer
from utils.data_utils import get_mnist_loaders


def evaluate_model(model, test_loader, device, criterion=None):
    """
    Evaluate model on the test set
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        device: Device to evaluate on
        criterion: Loss function (optional)
        
    Returns:
        accuracy: Accuracy on the test set
        avg_loss: Average loss on the test set (if criterion provided)
        class_correct: List of correct predictions per class
        class_total: List of total samples per class
        inference_time: Total inference time
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # For per-class accuracy
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
    # For timing inference
    start_time = time.time()
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Calculate per-class accuracy
            c = (predicted == targets).squeeze()
            for i in range(targets.size(0)):
                label = targets[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    inference_time = time.time() - start_time
    accuracy = 100. * correct / total
    avg_loss = total_loss / total if criterion is not None else None
    
    return accuracy, avg_loss, class_correct, class_total, inference_time


def visualize_predictions(model, test_loader, device, num_samples=10):
    """
    Visualize predictions on random samples
    
    Args:
        model: Model to use for predictions
        test_loader: DataLoader for test data
        device: Device to run predictions on
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Get a batch of data
    dataiter = iter(test_loader)
    images, labels = next(dataiter)
    
    # Move to device and get predictions
    images, labels = images.to(device), labels.to(device)
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # Move back to CPU for visualization
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    predicted = predicted.cpu().numpy()
    
    # Create a grid to display images
    fig = plt.figure(figsize=(15, 5))
    for i in range(min(num_samples, len(images))):
        ax = fig.add_subplot(2, 5, i + 1, xticks=[], yticks=[])
        # Reshape and denormalize image
        img = images[i][0]
        img = (img * 0.3081) + 0.1307  # Denormalize using MNIST mean and std
        ax.imshow(img, cmap='gray')
        
        # Add color-coded title based on correct/incorrect prediction
        title = f"True: {labels[i]}\nPred: {predicted[i]}"
        if labels[i] == predicted[i]:
            ax.set_title(title, color='green')
        else:
            ax.set_title(title, color='red')
    
    plt.tight_layout()
    plt.savefig('predictions_quantized.png')
    print(f"Predictions visualization saved to 'predictions_quantized.png'")


def visualize_confusion_matrix(model, test_loader, device):
    """
    Generate and visualize confusion matrix
    
    Args:
        model: Model to use for predictions
        test_loader: DataLoader for test data
        device: Device to run predictions on
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Computing confusion matrix'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Quantized Model)')
    plt.savefig('confusion_matrix_quantized.png')
    print(f"Confusion matrix saved to 'confusion_matrix_quantized.png'")


def quantize_weights_to_int8(model):
    """
    Quantize model weights to int8 precision for core Mamba components only
    
    Args:
        model: The model to quantize
        
    Returns:
        quantized_model: The model with quantized weights
    """
    # Create a copy of the model to avoid modifying the original
    quantized_model = copy.deepcopy(model)
    
    # Record statistics for each layer type
    total_params = 0
    quantized_params = 0
    
    # Iterate through all modules and quantize core Mamba components only
    for name, module in quantized_model.named_modules():
        # Skip MNIST-specific layers (input projection, row_proj, classifier)
        if any(mnist_layer in name for mnist_layer in ['input_proj', 'row_proj', 'classifier']):
            # Count parameters but don't quantize
            if hasattr(module, 'weight'):
                total_params += module.weight.numel()
            continue
            
        # Target only core Mamba components for quantization
        if isinstance(module, nn.Linear) and any(target in name for target in 
                                              ['in_proj', 'x_proj', 'dt_proj', 'out_proj']):
            # Scale factor for weight quantization
            weight_scale = module.weight.abs().max() / 127.0
            
            # Quantize weights to int8
            weight_int8 = torch.round(module.weight / weight_scale).to(torch.int8)
            
            # Store the quantized weights and scale factor
            module.register_buffer('weight_scale', weight_scale)
            module.register_buffer('weight_int8', weight_int8)
            
            # In inference, we'll use the quantized weights, so keep track of the fp32 weights
            module.weight_fp32 = module.weight.clone()
            
            # Replace the forward method
            module._original_forward = module.forward
            module.forward = lambda x, mod=module: mod._original_forward(x) if not hasattr(mod, 'weight_int8') else \
                                                  nn.functional.linear(x, (mod.weight_int8.float() * mod.weight_scale), 
                                                                     mod.bias)
            
            # Update statistics
            param_size = module.weight.numel()
            total_params += param_size
            quantized_params += param_size
            
            print(f"Quantized {name} (Linear): {module.weight.shape}, scale={weight_scale.item():.6f}")
            
        elif isinstance(module, nn.Conv1d) and 'conv1d' in name:
            # Scale factor for weight quantization
            weight_scale = module.weight.abs().max() / 127.0
            
            # Quantize weights to int8
            weight_int8 = torch.round(module.weight / weight_scale).to(torch.int8)
            
            # Store the quantized weights and scale factor
            module.register_buffer('weight_scale', weight_scale)
            module.register_buffer('weight_int8', weight_int8)
            
            # In inference, we'll use the quantized weights, so keep track of the fp32 weights
            module.weight_fp32 = module.weight.clone()
            
            # Replace the forward method with quantized version
            module._original_forward = module.forward
            
            def quantized_conv_forward(self, x):
                # Dequantize weights for computation
                dequantized_weight = self.weight_int8.float() * self.weight_scale
                return nn.functional.conv1d(x, dequantized_weight, self.bias, 
                                         self.stride, self.padding, 
                                         self.dilation, self.groups)
            
            module.forward = lambda x, mod=module: mod._original_forward(x) if not hasattr(mod, 'weight_int8') else \
                                                quantized_conv_forward(mod, x)
            
            # Update statistics
            param_size = module.weight.numel()
            total_params += param_size
            quantized_params += param_size
            
            print(f"Quantized {name} (Conv1d): {module.weight.shape}, scale={weight_scale.item():.6f}")
        
        # Count parameters for other layers with weights
        elif hasattr(module, 'weight'):
            total_params += module.weight.numel()
    
    # Print quantization statistics
    print(f"\nQuantization Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Quantized parameters: {quantized_params:,} ({quantized_params/total_params*100:.2f}%)")
    print(f"  Memory reduction: {(quantized_params*3)/1024/1024:.2f} MB")  # 3 bytes saved per parameter (4->1)
    
    return quantized_model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference with Quantized Mamba model on MNIST')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--confusion', action='store_true', help='Generate confusion matrix')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--weights_only', action='store_true', help='Use weights_only=True for loading')
    parser.add_argument('--compare', action='store_true', help='Compare with original model performance')
    parser.add_argument('--save_model', action='store_true', help='Save the quantized model')
    parser.add_argument('--output_path', type=str, default='quantized_model.pt', help='Path to save quantized model')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}')
    try:
        # Register MambaConfig as a safe global for unpickling
        torch.serialization.add_safe_globals([MambaConfig])
        
        # Load the checkpoint first
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=args.weights_only)
        # Extract config
        config = checkpoint.get('config', None)
        if config is None:
            # Try to load from separate config file
            config_path = args.checkpoint.replace('.pt', '_config.pt').replace('.pth', '_config.pth')
            if os.path.exists(config_path):
                config = torch.load(config_path, map_location=device)
        
        if config is None:
            raise ValueError("Could not find model configuration")
        
        # Create model with just the config
        model = MambaMNIST(config=config)
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    except Exception as e:
        print(f"Failed to load with specified options: {e}")
        print("Attempting fallback loading method...")
        try:
            # Simple fallback - load config from config file directly
            config_path = args.checkpoint.replace('.pt', '_config.pt')
            config = torch.load(config_path, map_location=device)
            model = MambaMNIST(config=config)
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        except Exception as e2:
            print(f"Fallback loading also failed: {e2}")
            raise e2
    
    model = model.to(device)
    
    # Get test loader
    _, test_loader = get_mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Evaluate original model if requested
    if args.compare:
        print("\n===== Original Model Performance =====")
        criterion = nn.CrossEntropyLoss()
        accuracy, avg_loss, class_correct, class_total, inference_time = evaluate_model(
            model, test_loader, device, criterion
        )
        
        # Print results
        print(f'Test Accuracy: {accuracy:.2f}%')
        print(f'Test Loss: {avg_loss:.4f}')
        print(f'Inference Time: {inference_time:.2f} seconds')
        
        # Print per-class accuracy
        print('\nPer-Class Accuracy:')
        for i in range(10):
            acc = 100 * class_correct[i] / class_total[i]
            print(f'  Digit {i}: {acc:.2f}%')
        
        # Save original model metrics
        original_accuracy = accuracy
        original_inference_time = inference_time
    
    # Quantize model weights to int8
    print("\n===== Quantizing Model to INT8 =====")
    quantized_model = quantize_weights_to_int8(model)
    quantized_model = quantized_model.to(device)
    
    # Evaluate quantized model
    print("\n===== Quantized Model Performance =====")
    criterion = nn.CrossEntropyLoss()
    accuracy, avg_loss, class_correct, class_total, inference_time = evaluate_model(
        quantized_model, test_loader, device, criterion
    )
    
    # Print results
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    print(f'Inference Time: {inference_time:.2f} seconds')
    
    # Print per-class accuracy
    print('\nPer-Class Accuracy:')
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'  Digit {i}: {acc:.2f}%')
    
    # Compare with original model if requested
    if args.compare:
        print("\n===== Performance Comparison =====")
        print(f'Original Accuracy: {original_accuracy:.2f}%')
        print(f'Quantized Accuracy: {accuracy:.2f}%')
        print(f'Accuracy Difference: {accuracy - original_accuracy:.2f}%')
        print(f'Original Inference Time: {original_inference_time:.2f} seconds')
        print(f'Quantized Inference Time: {inference_time:.2f} seconds')
        print(f'Speed Improvement: {(original_inference_time/inference_time - 1)*100:.2f}%')
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(quantized_model, test_loader, device, args.num_samples)
    
    # Generate confusion matrix if requested
    if args.confusion:
        visualize_confusion_matrix(quantized_model, test_loader, device)
    
    # Save quantized model if requested
    if args.save_model:
        print(f"\nSaving quantized model to {args.output_path}")
        torch.save({
            'model_state_dict': quantized_model.state_dict(),
            'config': config,
            'model_args': {'config': config},
            'quantized': True
        }, args.output_path)
        print("Quantized model saved successfully")


if __name__ == '__main__':
    main()