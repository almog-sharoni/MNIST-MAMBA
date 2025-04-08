import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm


def silu(x):
    """SiLU activation function"""
    return x * torch.sigmoid(x)


def softplus(x):
    """Softplus activation function"""
    return F.softplus(x)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        """Apply RMS normalization to input x."""
        # Calculate RMS
        rms = torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)
        # Normalize and scale
        x = x / rms * self.weight
        return x


class MambaConfig:
    """Configuration class for Mamba model for MNIST"""
    def __init__(self, 
                 input_size=28*28,  # MNIST image flattened size
                 dim=128,          # Hidden dimension size
                 n_layers=4,       # Number of Mamba layers
                 d_state=16,       # State dimension
                 d_conv=4,         # Convolution kernel size
                 expand=2,         # Expansion factor for inner dimension
                 num_classes=10,   # Number of output classes (digits 0-9)
                 dt_rank=None,     # Rank of delta (timestep) projection
                 dt_min=0.001,     # Minimum delta timestep
                 dt_max=0.1,       # Maximum delta timestep
                 dt_init="random", # Delta timestep initialization
                 dt_scale=1.0,     # Scale of delta timestep
                 dt_init_floor=1e-4):
        self.input_size = input_size
        self.dim = dim
        self.n_layers = n_layers
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(expand * dim)
        self.dt_rank = math.ceil(dim / 16) if dt_rank is None else dt_rank
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.num_classes = num_classes


class MambaLayer(nn.Module):
    """Single Mamba layer implementing the selective state space model"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        dim = config.dim
        d_inner = config.d_inner
        d_state = config.d_state
        d_conv = config.d_conv
        dt_rank = config.dt_rank
        
        # Input projection
        self.in_proj = nn.Linear(dim, 2 * d_inner, bias=True)
        
        # Convolution
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv-1,
            bias=True
        )
        
        # Projections for SSM parameters
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        
        # SSM parameters
        # Initialize A log as arange(1, d_state+1) for each d_inner
        A_log = torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
        A_log = A_log.expand(d_inner, -1).clone()  # (d_inner, d_state)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        
        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True
        
        # Output projection
        self.out_proj = nn.Linear(d_inner, dim, bias=True)
        
        # Initialize dt_proj to ensure good initial distribution
        dt_init_std = dt_rank**-0.5 * config.dt_scale
        if config.dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif config.dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        
        # Initialize dt bias so that softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner) * (math.log(config.dt_max) - math.log(config.dt_min))
            + math.log(config.dt_min)
        ).clamp(min=config.dt_init_floor)
        # Inverse of softplus: ln(exp(x) - 1)
        inv_dt = torch.log(torch.exp(dt) - 1)
        self.dt_proj.bias.data.copy_(inv_dt)
    
    def forward(self, hidden_state):
        """
        Forward pass for a single layer
        
        Args:
            hidden_state: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            output: Tensor of same shape as hidden_state
        """
        batch_size, seq_len, dim = hidden_state.shape
        d_inner = self.config.d_inner
        d_state = self.config.d_state
        
        # Input projection: hidden_state (batch, seq, dim) -> xz (batch, seq, 2*d_inner)
        xz = self.in_proj(hidden_state)
        
        # Split into x and z components
        x, z = torch.chunk(xz, 2, dim=-1)  # Each (batch, seq, d_inner)
        
        # For convolutional part, we need to transpose: (batch, d_inner, seq)
        x_conv = x.transpose(1, 2)
        
        # Apply causal convolution
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]
        
        # Apply SiLU activation and transpose back: (batch, seq, d_inner)
        x_conv = F.silu(x_conv).transpose(1, 2)
        
        # Project x to obtain dt, B, C parameters
        x_db = self.x_proj(x_conv)  # (batch, seq, dt_rank + 2*d_state)
        
        # Split the projection into dt, B, C components
        dt, B, C = torch.split(
            x_db, [self.config.dt_rank, d_state, d_state], dim=-1
        )
        
        # Project dt from dt_rank to d_inner dimension and apply softplus
        dt = self.dt_proj(dt)  # (batch, seq, d_inner)
        dt = F.softplus(dt)  # Ensure dt is positive
        
        # Compute discretized A
        # A is a negative value obtained by exponentiating A_log
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # Initialize SSM state - shape: (batch_size, d_inner, d_state)
        ssm_state = torch.zeros(batch_size, d_inner, d_state, device=hidden_state.device)
        
        # Prepare output tensor
        y = torch.zeros(batch_size, seq_len, d_inner, device=hidden_state.device)
        
        # Process sequence elements
        for t in range(seq_len):
            # Current input at position t
            xt = x_conv[:, t]  # (batch, d_inner)
            dt_t = dt[:, t]    # (batch, d_inner)
            B_t = B[:, t]      # (batch, d_state)
            C_t = C[:, t]      # (batch, d_state)
            
            # Compute discretized A for current timestep: dA = exp(outer(dt, A))
            # We need to be careful with broadcasting dimensions:
            # dt_t is (batch, d_inner), A is (d_inner, d_state)
            # Reshape dt_t to (batch, d_inner, 1) for proper broadcasting with A
            dA = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (batch, d_inner, d_state)
            
            # Compute discretized B for current timestep
            # dt_t is (batch, d_inner), B_t is (batch, d_state)
            # Reshape for outer product: (batch, d_inner, 1) * (batch, 1, d_state)
            dB = dt_t.unsqueeze(-1) * B_t.unsqueeze(1)  # (batch, d_inner, d_state)
            
            # Update SSM state
            # ssm_state: (batch, d_inner, d_state)
            # dA: (batch, d_inner, d_state)
            # xt: (batch, d_inner) -> (batch, d_inner, 1)
            # dB: (batch, d_inner, d_state)
            ssm_state = ssm_state * dA + xt.unsqueeze(-1) * dB
            
            # Compute output
            # ssm_state: (batch, d_inner, d_state)
            # C_t: (batch, d_state) -> (batch, 1, d_state)
            # Result should be (batch, d_inner)
            y[:, t] = torch.sum(ssm_state * C_t.unsqueeze(1), dim=-1)
        
        # Apply skip connection with D and activation with z
        y = y + self.D.unsqueeze(0).unsqueeze(0) * x_conv
        y = y * F.silu(z)
        
        # Output projection
        out = self.out_proj(y)
        
        return out


class MambaMNIST(nn.Module):
    """Mamba model adapted for MNIST classification"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection from MNIST image to model dimension
        self.input_proj = nn.Linear(config.input_size, config.dim)
        
        # Layer normalizations
        self.norm_layers = nn.ModuleList([RMSNorm(config.dim) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.dim)
        
        # Mamba layers
        self.layers = nn.ModuleList([MambaLayer(config) for _ in range(config.n_layers)])
        
        # Classification head
        self.classifier = nn.Linear(config.dim, config.num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Standard initialization for linear layers
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass for MNIST classification
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
                
        Returns:
            logits: Output logits (batch_size, num_classes)
        """
        batch_size = x.shape[0]
        
        # Reshape MNIST images to sequence format:
        # Original shape: (batch_size, 1, 28, 28)
        # We want: (batch_size, seq_len=28, feature_dim=28)
        # Each row of pixels becomes a token in our sequence
        x = x.view(batch_size, 1, 28, 28).squeeze(1)  # Remove channel dim
        x = x.float()  # Ensure float type
        
        # Now shape is (batch_size, 28, 28) - each row is a "token" with 28 features
        
        # Project each row (each with 28 features) to model dimension
        # First reshape to (batch_size * 28, 28) to apply projection to each row
        x_reshaped = x.reshape(-1, 28)
        
        # Create a new projection layer specifically for row features
        if not hasattr(self, 'row_proj'):
            self.row_proj = nn.Linear(28, self.config.dim).to(x.device)
        
        # Apply projection to each row
        x_projected = self.row_proj(x_reshaped)  # (batch_size * 28, dim)
        
        # Reshape back to (batch_size, 28, dim)
        x = x_projected.view(batch_size, 28, self.config.dim)
        
        # Forward through each Mamba layer with residual connections
        hidden_states = x
        
        # Rest of the forward pass remains the same
        for i, layer in enumerate(self.layers):
            # Apply layer normalization
            normed_states = self.norm_layers[i](hidden_states)
            
            # Forward through Mamba layer
            layer_output = layer(normed_states)
            
            # Residual connection
            hidden_states = hidden_states + layer_output
        
        # Final normalization
        hidden_states = self.final_norm(hidden_states)
        
        # Take the representation from the last "token" (last row of the image)
        # Alternative: could use mean pooling across all positions
        final_representation = hidden_states[:, -1]  # (batch_size, dim)
        
        # Classify
        logits = self.classifier(final_representation)
        
        return logits
    
    def train_step(self, batch, optimizer, criterion):
        """
        Perform a single training step
        
        Args:
            batch: Tuple of (inputs, targets)
            optimizer: PyTorch optimizer
            criterion: Loss function
            
        Returns:
            loss: Training loss for this batch
        """
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = self(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def evaluate(self, data_loader, criterion=None):
        """
        Evaluate the model on a data loader
        
        Args:
            data_loader: PyTorch DataLoader
            criterion: Loss function (optional)
            
        Returns:
            accuracy: Accuracy on the dataset
            avg_loss: Average loss on the dataset (if criterion provided)
        """
        self.eval()
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(next(self.parameters()).device), targets.to(next(self.parameters()).device)
                outputs = self(inputs)
                
                if criterion is not None:
                    loss = criterion(outputs, targets)
                    total_loss += loss.item() * inputs.size(0)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = correct / total
        avg_loss = total_loss / total if criterion is not None else None
        
        self.train()
        return accuracy, avg_loss
    
    def save_checkpoint(self, path, optimizer=None, epoch=None, best_acc=None):
        """
        Save model checkpoint
        
        Args:
            path: Path to save the checkpoint
            optimizer: Optimizer state (optional)
            epoch: Current epoch (optional)
            best_acc: Best accuracy (optional)
        """
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if best_acc is not None:
            checkpoint['best_acc'] = best_acc
            
        torch.save(checkpoint, path)
    
    @classmethod
    def load_from_checkpoint(cls, path, map_location=None):
        """
        Load model from checkpoint
        
        Args:
            path: Path to the checkpoint
            map_location: Device mapping function (optional)
            
        Returns:
            model: Loaded model
            checkpoint: Full checkpoint dictionary
        """
        checkpoint = torch.load(path, map_location=map_location)
        config = checkpoint['config']
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint


# Example of how to use the model for MNIST
if __name__ == "__main__":
    from utils.data_utils import get_mnist_loaders
    import torch.optim as optim
    import os
    
    # Create configuration
    config = MambaConfig(
        input_size=28*28,  # MNIST image size
        dim=128,          # Hidden dimension
        n_layers=4,       # Number of layers
        d_state=16,       # State size
        d_conv=4,         # Convolution kernel size
        expand=2,         # Expansion factor
        num_classes=10    # 10 digits
    )
    
    # Create model
    model = MambaMNIST(config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Using device: {device}")
    
    # Get data loaders
    train_loader, test_loader = get_mnist_loaders(batch_size=256)
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    
    # Training loop with tqdm progress bar
    num_epochs = 10
    best_acc = 0.0
    
    for epoch in range(num_epochs):
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
        
        # Evaluate on test set with progress bar
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
                'checkpoints/best_model.pt',
                optimizer=optimizer,
                epoch=epoch,
                best_acc=best_acc
            )