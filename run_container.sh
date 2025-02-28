#!/bin/bash

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Docker runtime is available
if ! docker info | grep -i "nvidia" &> /dev/null; then
    echo "Error: NVIDIA Docker runtime not found. Please install nvidia-docker2."
    exit 1
fi

# Check if the image exists, if not build it
if ! docker images | grep "pet-classifier" &> /dev/null; then
    echo "Building Docker image..."
    docker build -t pet-classifier -f docker/Dockerfile .
    if [ $? -ne 0 ]; then
        echo "Error: Failed to build Docker image."
        exit 1
    fi
fi

# Function to display the menu
show_menu() {
    echo ""
    echo "Pet Breed Classifier - Operation Mode Selection"
    echo "----------------------------------------"
    echo "1) Use existing model (no training)"
    echo "2) Quick training mode (1 epoch each phase)"
    echo "3) Full training with custom configuration"
    echo "4) Use specific model file"
    echo "5) Exit"
    echo ""
    echo -n "Please select an option [1-5]: "
}

# Function to read model path
read_model_path() {
    echo -n "Enter the model path (relative to models directory, e.g., my_model.pkl): "
    read model_path
}

# Function to read custom training parameters
read_training_params() {
    # Default values
    default_epochs_frozen=5
    default_epochs_unfrozen=10
    default_lr_frozen=1e-3
    default_lr_unfrozen_min=1e-6
    default_lr_unfrozen_max=1e-4

    echo -n "Enter number of epochs for frozen layers [default: ${default_epochs_frozen}] (press Enter to use default): "
    read input
    epochs_frozen=${input:-$default_epochs_frozen}

    echo -n "Enter number of epochs for unfrozen layers [default: ${default_epochs_unfrozen}] (press Enter to use default): "
    read input
    epochs_unfrozen=${input:-$default_epochs_unfrozen}

    echo -n "Enter learning rate for frozen layers [default: ${default_lr_frozen}] (press Enter to use default): "
    read input
    lr_frozen=${input:-$default_lr_frozen}

    echo -n "Enter minimum learning rate for unfrozen layers [default: ${default_lr_unfrozen_min}] (press Enter to use default): "
    read input
    lr_unfrozen_min=${input:-$default_lr_unfrozen_min}

    echo -n "Enter maximum learning rate for unfrozen layers [default: ${default_lr_unfrozen_max}] (press Enter to use default): "
    read input
    lr_unfrozen_max=${input:-$default_lr_unfrozen_max}
}

# Main menu loop
while true; do
    show_menu
    read choice

    case $choice in
        1)  # Use existing model
            cmd="docker run --gpus all --shm-size=1g -p 7860:7860 \
                -v \"$(pwd)/data:/app/data\" \
                -v \"$(pwd)/models:/app/models\" \
                --name pet-classifier \
                --rm \
                pet-classifier \
                python src/pet_classifier.py --no-retrain"
            break
            ;;
        2)  # Quick training
            cmd="docker run --gpus all --shm-size=1g -p 7860:7860 \
                -v \"$(pwd)/data:/app/data\" \
                -v \"$(pwd)/models:/app/models\" \
                --name pet-classifier \
                --rm \
                pet-classifier \
                python src/pet_classifier.py --retrain --quick-train"
            break
            ;;
        3)  # Custom training
            read_training_params
            cmd="docker run --gpus all --shm-size=1g -p 7860:7860 \
                -v \"$(pwd)/data:/app/data\" \
                -v \"$(pwd)/models:/app/models\" \
                --name pet-classifier \
                --rm \
                pet-classifier \
                python src/pet_classifier.py --retrain \
                --epochs-frozen $epochs_frozen \
                --epochs-unfrozen $epochs_unfrozen \
                --lr-frozen $lr_frozen \
                --lr-unfrozen-min $lr_unfrozen_min \
                --lr-unfrozen-max $lr_unfrozen_max"
            break
            ;;
        4)  # Use specific model
            read_model_path
            cmd="docker run --gpus all --shm-size=1g -p 7860:7860 \
                -v \"$(pwd)/data:/app/data\" \
                -v \"$(pwd)/models:/app/models\" \
                --name pet-classifier \
                --rm \
                pet-classifier \
                python src/pet_classifier.py --model-path models/$model_path"
            break
            ;;
        5)  # Exit
            echo "Exiting..."
            exit 0
            ;;
        *)  # Invalid option
            echo "Invalid option. Please try again."
            ;;
    esac
done

# Run the container with selected configuration
echo "Starting pet breed classifier container..."
eval $cmd

echo "Container started successfully. Access the interface at http://localhost:7860"