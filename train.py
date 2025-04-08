import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse

from models.mamba_model import MambaConfig, MambaMNIST
from utils.data_utils import get_mnist_loaders


def train(model, train_loader, test_loader, optimizer, criterion, device, 
          num_epochs=10, save_dir='checkpoints', save_prefix='model'):
    """
    Train the model
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        optimizer: Optimizer to use
        criterion: Loss function
        device: Device to train on
        num_epochs: Number of epochs to train for
        save_dir: Directory to save checkpoints
        save_prefix: Prefix for checkpoint filenames
    
    Returns:
        best_acc: Best accuracy achieved during training
    """
    # Create directory for saving checkpoints
    os.makedirs(save_dir, exist_ok=True)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        # Training with progress bar
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for inputs, targets in train_pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update metrics
            total_loss += loss.item() * inputs.size(0)
            train_acc = 100. * correct / total
            train_loss = total_loss / total
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{train_loss:.4f}',
                'acc': f'{train_acc:.2f}%'
            })
        
        # Evaluation phase
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        test_pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Test]')
        with torch.no_grad():
            for inputs, targets in test_pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Calculate accuracy
                _, predicted = outputs.max(1)
                test_total += targets.size(0)
                test_correct += predicted.eq(targets).sum().item()
                
                # Update metrics
                test_loss += loss.item() * inputs.size(0)
                test_acc = 100. * test_correct / test_total
                avg_test_loss = test_loss / test_total
                
                # Update progress bar
                test_pbar.set_postfix({
                    'loss': f'{avg_test_loss:.4f}',
                    'acc': f'{test_acc:.2f}%'
                })
        
        # Print epoch summary
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            print(f'  New best accuracy: {best_acc:.2f}% - Saving model...')
            model.save_checkpoint(
                f'{save_dir}/best_{save_prefix}.pt',
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc
            )
        
        # Save latest model
        model.save_checkpoint(
            f'{save_dir}/latest_{save_prefix}.pt',
            optimizer=optimizer,
            epoch=epoch,
            best_acc=best_acc
        )
    
    return best_acc


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train Mamba model on MNIST')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train for')
    parser.add_argument('--dim', type=int, default=128, help='Hidden dimension size')
    parser.add_argument('--n_layers', type=int, default=4, help='Number of Mamba layers')
    parser.add_argument('--d_state', type=int, default=16, help='State dimension')
    parser.add_argument('--d_conv', type=int, default=4, help='Convolution kernel size')
    parser.add_argument('--expand', type=int, default=2, help='Expansion factor')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save models')
    parser.add_argument('--save_prefix', type=str, default='mamba_mnist', help='Prefix for saved model files')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--resume', type=str, default='', help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=args.batch_size, num_workers=args.num_workers)
    
    # Create model
    if args.resume:
        print(f'Loading model from {args.resume}')
        model, checkpoint = MambaMNIST.load_from_checkpoint(args.resume, map_location=device)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Resumed from epoch {start_epoch} with best accuracy {best_acc:.2f}%')
    else:
        # Create new model
        config = MambaConfig(
            input_size=28*28,  # MNIST image size
            dim=args.dim,
            n_layers=args.n_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            num_classes=10    # 10 digits
        )
        model = MambaMNIST(config)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Move model to device
    model = model.to(device)
    
    # Set up loss function
    criterion = nn.CrossEntropyLoss()
    
    # Train the model
    best_acc = train(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        save_prefix=args.save_prefix
    )
    
    print(f'Training completed! Best accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()