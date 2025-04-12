import os
import struct
import argparse
import json
import torch
import torchvision
import numpy as np
from models.mamba_model import MambaMNIST, MambaConfig
from torch.serialization import add_safe_globals

# Add MambaConfig to safe globals for loading checkpoints
add_safe_globals([MambaConfig])

# -----------------------------------------------------------------------------
# common utilities

def serialize_fp32(file, tensor):
    """ writes one fp32 tensor to file that is open in wb mode """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

# -----------------------------------------------------------------------------
# model export functions

def write_weights(file, tensor, key):
    """ writes the layer weights to file """
    d = tensor.detach().cpu().view(-1).to(torch.float32).numpy()
    print(f"writing {key} {list(tensor.shape)[::-1]}")
    print(f"  first values: {d[:4]}")  # Print first 4 values
    b = struct.pack(f'{len(d)}f', *d)
    file.write(b)

def model_export(model, config, filepath):
    """Export the MNIST Mamba model weights in float32 .bin file"""
    version = 1
    
    with open(filepath, 'wb') as out_file:
        # Write header (256 bytes)
        out_file.write(struct.pack('I', 0x4d616d62))  # magic "Mamb"
        out_file.write(struct.pack('i', version))
        
        # Get model dimensions
        d_inner = config.d_inner
        dt_rank = config.dt_rank
        d_state = config.d_state
        d_conv = config.d_conv
        
        print(f"writing header:\n"
              f"  layers: {config.n_layers}\n"
              f"  input_size: {config.input_size}\n"
              f"  dim: {config.dim}\n"
              f"  d_inner: {d_inner}\n"
              f"  dt_rank: {dt_rank}\n"
              f"  d_state: {d_state}\n"
              f"  d_conv: {d_conv}")
        
        # Write config
        header = struct.pack('iiiiiiii', 
            config.n_layers, config.input_size, config.dim,
            d_inner, dt_rank, d_state, d_conv, config.num_classes)
        out_file.write(header)
        
        # Pad to 256 bytes
        pad = 256 - out_file.tell()
        out_file.write(b'\0' * pad)
        
        # Write model weights
        # Input projection
        write_weights(out_file, model.input_proj.weight, 'input_proj.weight')
        
        # Layer weights
        for i in range(config.n_layers):
            layer_dict = model.layers[i]
            mamba = layer_dict['mamba']
            norm = layer_dict['norm']
            
            write_weights(out_file, mamba.in_proj.weight, f'layer_{i}.in_proj.weight')
            write_weights(out_file, mamba.conv1d.weight.squeeze(1), f'layer_{i}.conv1d.weight')
            write_weights(out_file, mamba.conv1d.bias, f'layer_{i}.conv1d.bias')
            write_weights(out_file, mamba.x_proj.weight, f'layer_{i}.x_proj.weight')
            write_weights(out_file, mamba.dt_proj.weight, f'layer_{i}.dt_proj.weight')
            write_weights(out_file, mamba.dt_proj.bias, f'layer_{i}.dt_proj.bias')
            write_weights(out_file, -torch.exp(mamba.A_log), f'layer_{i}.A')
            write_weights(out_file, mamba.D, f'layer_{i}.D')
            write_weights(out_file, mamba.out_proj.weight, f'layer_{i}.out_proj.weight')
            write_weights(out_file, norm.weight, f'layer_{i}.norm.weight')
        
        # Final normalization
        write_weights(out_file, model.norm_f.weight, 'norm_f.weight')
        
        # Output head
        write_weights(out_file, model.output_head.weight, 'output_head.weight')


# -----------------------------------------------------------------------------
# Load / import functions

def load_model(path):
    print(f"loading model from {path}")

    # load the model
    if os.path.isdir(path):
        filepath = os.path.join(path, 'pytorch_model.bin')
    else:
        filepath = path
    model = torch.load(filepath, map_location='cpu')

    # remove the 'backbone.' prefix from the keys
    unwanted_prefix = 'backbone.'
    for k,v in list(model.items()):
        if k.startswith(unwanted_prefix):
            model[k[len(unwanted_prefix):]] = model.pop(k)

    # get the path to the config file
    if os.path.isdir(path):
        config_path = os.path.join(path, 'config.json')
    else:
        config_path = os.path.join(os.path.dirname(path), 'config.json')
    # load the config
    with open(config_path) as f:
        config = json.load(f)
    # rename config.n_layers to config.n_layers
    config['n_layers'] = config.pop('n_layer')
    config = argparse.Namespace(**config)    

    return model, config


def get_model_from_huggingface(model_name: str):
    """Download model from HuggingFace and get the path to the model file.
    The model name can be one of the following:
        'state-spaces/mamba-130m'
        'state-spaces/mamba-370m'
        'state-spaces/mamba-790m'
        'state-spaces/mamba-1.4b'
        'state-spaces/mamba-2.8b'
        'state-spaces/mamba-2.8b-slimpj'
    """
    from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
    from transformers.utils.hub import cached_file

    config_path = cached_file(model_name, CONFIG_NAME, _raise_exceptions_for_missing_entries=False)
    model_path = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)

    return model_path

# -----------------------------------------------------------------------------
# Export MNIST image

def export_mnist_image(index=0, output_file="input.bin"):
    """Export a single MNIST test image to binary format for C model inference"""
    # Load MNIST dataset
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    
    # Get image and label
    image, label = mnist[index]
    
    # Convert to float32 numpy array and normalize to [0,1]
    image_array = np.array(image, dtype=np.float32) / 255.0
    
    # Flatten the 28x28 image
    flat_image = image_array.reshape(784)
    
    # Save to binary file
    with open(output_file, 'wb') as f:
        f.write(flat_image.tobytes())
    
    print(f"Exported image with label {label} to {output_file}")
    print(f"Run the C model with: ./mamba model.bin {output_file}")

def export_best_model(checkpoint_path='checkpoints/best_model.pth', export_path='mnist_mamba.bin'):
    """Export the best trained model to binary format for C inference"""
    try:
        # First try with weights_only=True
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        print("Loaded checkpoint with weights_only=True")
    except Exception as e:
        # If that fails, try with weights_only=False
        print("Attempting to load with weights_only=False...")
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        print("Loaded checkpoint with weights_only=False")
    
    # Check if we have a state dict directly
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        config = checkpoint.get('config', MambaConfig())  # Use default config if not found
        accuracy = checkpoint.get('accuracy', 0.0)
    else:
        # Assume checkpoint is the state dict directly
        state_dict = checkpoint
        config = MambaConfig()
        accuracy = 0.0
    
    # Initialize model with config
    model = MambaMNIST(config)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Exporting model with accuracy: {accuracy:.2f}%")
    
    # Export model to binary format
    model_export(model, config, export_path)
    print(f"Model exported to {export_path}")

# -----------------------------------------------------------------------------
# CLI entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Export MNIST Mamba model and test images')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Model export command
    model_parser = subparsers.add_parser('model', help='Export trained model to binary format')
    model_parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', 
                            help='Path to model checkpoint')
    model_parser.add_argument('--output', type=str, default='mnist_mamba.bin',
                            help='Output binary file path')
    
    # Image export command
    image_parser = subparsers.add_parser('image', help='Export MNIST test image to binary format')
    image_parser.add_argument('--index', type=int, default=0,
                            help='Index of the MNIST test image to export')
    image_parser.add_argument('--output', type=str, default='input.bin',
                            help='Output binary file path')
    
    args = parser.parse_args()
    
    if args.command == 'model':
        export_best_model(args.checkpoint, args.output)
    elif args.command == 'image':
        export_mnist_image(args.index, args.output)
    else:
        parser.print_help()
