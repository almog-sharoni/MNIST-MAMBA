import torchvision
import numpy as np
import subprocess
import os
import time
from tqdm import tqdm

def evaluate_mnist(model_path, temp_input="temp_input.bin"):
    """Evaluate MNIST test dataset using the C Mamba model"""
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Check if mamba executable exists
    if not os.path.exists("./mamba"):
        raise FileNotFoundError("Mamba executable not found in current directory")
    
    # Load MNIST test dataset
    mnist = torchvision.datasets.MNIST(root='./data', train=False, download=True)
    print(f"Loaded {len(mnist)} test images")
    
    correct = 0
    total = len(mnist)
    
    # Create progress bar
    pbar = tqdm(total=total, desc="Evaluating")
    
    start_time = time.time()
    
    for idx in range(total):
        # Get image and label
        image, label = mnist[idx]
        
        # Convert to float32 and normalize
        image_array = np.array(image, dtype=np.float32) / 255.0
        flat_image = image_array.reshape(784)
        
        # Save to temporary file
        with open(temp_input, 'wb') as f:
            f.write(flat_image.tobytes())
        
        # Run mamba inference
        try:
            result = subprocess.run(
                ['./mamba', model_path, temp_input],  # Removed --quiet flag
                capture_output=True,
                text=True,
                check=False
            )
            
            if result.returncode != 0:
                print(f"Error on image {idx}:")
                print(f"  Command: ./mamba {model_path} {temp_input}")  # Added command debug info
                print(f"  Exit code: {result.returncode}")
                print(f"  stderr: {result.stderr.strip()}")
                print(f"  stdout: {result.stdout.strip()}")
                continue
                
            output = result.stdout.strip()
            try:
                # Extract just the predicted class number from the formatted output
                prediction = int(output.split(':')[1].split()[0])
            except (ValueError, IndexError):
                print(f"Error: Unable to parse output on image {idx}: '{output}'")
                continue
            
            # Check if correct
            if prediction == label:
                correct += 1
                
        except Exception as e:
            print(f"Unexpected error on image {idx}: {str(e)}")
            continue
            
        # Update progress bar
        accuracy = (correct * 100.0) / (idx + 1)
        pbar.set_postfix({'acc': f'{accuracy:.2f}%'})
        pbar.update(1)
    
    pbar.close()
    
    # Clean up temporary file
    if os.path.exists(temp_input):
        os.remove(temp_input)
    
    # Print final results
    duration = time.time() - start_time
    print(f"\nFinal Results:")
    print(f"Accuracy: {(correct * 100.0 / total):.2f}%")
    print(f"Correct: {correct}/{total}")
    print(f"Time: {duration:.2f}s ({duration/total*1000:.1f}ms per image)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate MNIST test set using C Mamba model')
    parser.add_argument('model', help='Path to the model.bin file')
    args = parser.parse_args()
    
    evaluate_mnist(args.model)
