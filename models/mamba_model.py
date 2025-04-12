import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import pickle


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
                 dim=16,          # Hidden dimension size
                 n_layers=1,       # Number of Mamba layers
                 d_state=8,       # State dimension
                 d_conv=5,         # Convolution kernel size
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
    """Mamba model for MNIST classification"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_size, config.dim)
        
        # Mamba layers with normalization
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'norm': RMSNorm(config.dim),
                'mamba': MambaLayer(config)
            }) for _ in range(config.n_layers)
        ])
        
        # Output head
        self.norm_f = RMSNorm(config.dim)
        self.output_head = nn.Linear(config.dim, config.num_classes)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, map_location=None):
        """Load model from checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file
            map_location (str or torch.device): Device to load the model onto
            
        Returns:
            tuple: (model, checkpoint_dict)
        """
        # Allow MambaConfig in torch.load
        import torch.serialization
        torch.serialization.add_safe_globals([MambaConfig])
        
        try:
            # Try loading with weights_only=False since we need the config
            checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            raise
        
        # Get model configuration - try both possible locations
        config = checkpoint.get('config', None)
        if config is None and 'model_args' in checkpoint:
            config = checkpoint['model_args'].get('config', None)
            
        if config is None:
            raise ValueError("Checkpoint does not contain model configuration")
            
        # Ensure config is a MambaConfig instance
        if not isinstance(config, MambaConfig):
            # If it's a dict, convert to MambaConfig
            if isinstance(config, dict):
                config = MambaConfig(**config)
            else:
                raise ValueError(f"Invalid config type: {type(config)}")
            
        # Create new model instance with loaded config
        model = cls(config)
        
        # Load state dictionary
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            raise ValueError("Checkpoint does not contain model state dict")
        
        return model, checkpoint
    
    def forward(self, x):
        # Reshape input: (batch, 1, 28, 28) -> (batch, 784)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Project to model dimension: (batch, 784) -> (batch, 1, dim)
        x = self.input_proj(x).unsqueeze(1)
        
        # Apply Mamba layers
        for layer in self.layers:
            # Pre-normalization
            x_norm = layer['norm'](x)
            # Mamba layer
            x = x + layer['mamba'](x_norm)
        
        # Final normalization and pool
        x = self.norm_f(x).squeeze(1)
        
        # Classification head
        logits = self.output_head(x)
        
        return logits
