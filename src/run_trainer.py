import argparse
import subprocess
import os

def validate_positive_float(value):
    try:
        float_value = float(value)
        if float_value <= 0:
            raise ValueError
        return float_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a positive float")

def validate_positive_int(value):
    try:
        int_value = int(value)
        if int_value <= 0:
            raise ValueError
        return int_value
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value} must be a positive integer")

def main():
    parser = argparse.ArgumentParser(description="Pet Breed Classifier Training Configuration")
    
    # Training parameters
    parser.add_argument("--epochs-frozen", type=validate_positive_int, default=5,
                        help="Number of epochs for training with frozen layers")
    parser.add_argument("--epochs-unfrozen", type=validate_positive_int, default=10,
                        help="Number of epochs for training with unfrozen layers")
    parser.add_argument("--lr-frozen", type=validate_positive_float, default=1e-3,
                        help="Learning rate for training with frozen layers")
    parser.add_argument("--lr-unfrozen", type=validate_positive_float, default=1e-5,
                        help="Learning rate for training with unfrozen layers")
    
    # Docker configuration
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to expose the Gradio interface")
    parser.add_argument("--shm-size", type=str, default="1g",
                        help="Shared memory size for Docker container")
    parser.add_argument("--model-path", type=str,
                        help="Path to a specific model file to use")
    parser.add_argument("--quick-train", action="store_true",
                        help="Enable quick training mode (1 epoch each phase)")
    parser.add_argument("--no-retrain", action="store_true",
                        help="Use existing model without retraining")
    
    args = parser.parse_args()
    
    # Build Docker command
    docker_cmd = [
        "docker", "run",
        "--gpus", "all",
        "--shm-size", args.shm_size,
        "-p", f"{args.port}:7860",
        "-v", f"{os.path.abspath('data')}:/app/data",
        "-v", f"{os.path.abspath('models')}:/app/models",
        "--name", "pet-classifier",
        "--rm",
        "pet-classifier"
    ]
    
    # Add Python command and arguments
    python_cmd = ["python", "src/pet_classifier.py"]
    
    if not args.no_retrain:
        python_cmd.append("--retrain")
    
    if args.quick_train:
        python_cmd.append("--quick-train")
    elif not args.no_retrain:
        python_cmd.extend([
            "--epochs-frozen", str(args.epochs_frozen),
            "--epochs-unfrozen", str(args.epochs_unfrozen),
            "--lr-frozen", str(args.lr_frozen),
            "--lr-unfrozen", str(args.lr_unfrozen)
        ])
    
    if args.model_path:
        python_cmd.extend(["--model-path", args.model_path])
    
    # Combine Docker and Python commands
    full_cmd = docker_cmd + python_cmd
    
    # Check if Docker image exists
    result = subprocess.run(["docker", "images", "-q", "pet-classifier"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("Building Docker image...")
        subprocess.run(["docker", "build", "-t", "pet-classifier", "-f", "docker/Dockerfile", "."], check=True)
    
    # Run the container
    print("Starting pet breed classifier container...")
    print(f"Command: {' '.join(full_cmd)}")
    print(f"\nTraining Configuration:")
    if not args.no_retrain:
        if args.quick_train:
            print("Quick training mode: 1 epoch each phase")
        else:
            print(f"Frozen layers: {args.epochs_frozen} epochs (lr={args.lr_frozen})")
            print(f"Unfrozen layers: {args.epochs_unfrozen} epochs (lr={args.lr_unfrozen})")
    else:
        print("Using existing model (no training)")
    
    subprocess.run(full_cmd)
    print(f"\nContainer started successfully. Access the interface at http://localhost:{args.port}")

if __name__ == "__main__":
    main()