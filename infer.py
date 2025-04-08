import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from models.mamba_model import MambaMNIST
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
    """
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    
    # For per-class accuracy
    class_correct = list(0. for _ in range(10))
    class_total = list(0. for _ in range(10))
    
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
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / total if criterion is not None else None
    
    return accuracy, avg_loss, class_correct, class_total


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
    plt.savefig('predictions.png')
    print(f"Predictions visualization saved to 'predictions.png'")


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
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print(f"Confusion matrix saved to 'confusion_matrix.png'")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Inference with Mamba model on MNIST')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--visualize', action='store_true', help='Visualize predictions')
    parser.add_argument('--confusion', action='store_true', help='Generate confusion matrix')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Load model
    print(f'Loading model from {args.checkpoint}')
    model, checkpoint = MambaMNIST.load_from_checkpoint(args.checkpoint, map_location=device)
    model = model.to(device)
    
    # Get test loader
    _, test_loader = get_mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Evaluate model
    criterion = nn.CrossEntropyLoss()
    accuracy, avg_loss, class_correct, class_total = evaluate_model(
        model, test_loader, device, criterion
    )
    
    # Print results
    print(f'Test Accuracy: {accuracy:.2f}%')
    print(f'Test Loss: {avg_loss:.4f}')
    
    # Print per-class accuracy
    print('\nPer-Class Accuracy:')
    for i in range(10):
        acc = 100 * class_correct[i] / class_total[i]
        print(f'  Digit {i}: {acc:.2f}%')
    
    # Visualize predictions if requested
    if args.visualize:
        visualize_predictions(model, test_loader, device, args.num_samples)
    
    # Generate confusion matrix if requested
    if args.confusion:
        visualize_confusion_matrix(model, test_loader, device)


if __name__ == '__main__':
    main()