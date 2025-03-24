#!/bin/bash

# Auto-training and publishing script for Pet Breed Classifier
# This script retrains the model with customizable parameters and publishes to Hugging Face

# Set default parameters
EPOCHS_FROZEN=5
EPOCHS_UNFROZEN=10
LR_FROZEN=1e-3
LR_UNFROZEN=1e-5
MODEL_PATH="models/pet_classifier.pkl"
HF_SPACE="JinApang/pet_classifier"  # Default Hugging Face space name
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
PUBLISH_TO_HF=true

# Parse command-line arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --epochs-frozen)
      EPOCHS_FROZEN="$2"
      shift 2
      ;;
    --epochs-unfrozen)
      EPOCHS_UNFROZEN="$2"
      shift 2
      ;;
    --lr-frozen)
      LR_FROZEN="$2"
      shift 2
      ;;
    --lr-unfrozen)
      LR_UNFROZEN="$2"
      shift 2
      ;;
    *)
      echo "Unknown parameter: $1"
      exit 1
      ;;
  esac
done

# Display banner
echo "=============================================="
echo "  Pet Breed Classifier - Auto Training Tool"
echo "=============================================="
echo ""

# Function to read user input with a default value
read_input_with_default() {
    local prompt=$1
    local default=$2
    local input
    
    echo -n "$prompt [$default]: "
    read input
    echo "${input:-$default}"
}

# Function to read yes/no input with a default value
read_yes_no() {
    local prompt=$1
    local default=$2
    local input
    
    while true; do
        echo -n "$prompt (y/n) [$default]: "
        read input
        input=${input:-$default}
        case $input in
            [Yy]* ) return 0;;
            [Nn]* ) return 1;;
            * ) echo "Please answer yes (y) or no (n).";;
        esac
    done
}

# Ask if user wants to use custom parameters
if read_yes_no "Do you want to customize training parameters?" "n"; then
    # Get custom parameters from user
    EPOCHS_FROZEN=$(read_input_with_default "Number of epochs for frozen layers" "$EPOCHS_FROZEN")
    EPOCHS_UNFROZEN=$(read_input_with_default "Number of epochs for unfrozen layers" "$EPOCHS_UNFROZEN")
    LR_FROZEN=$(read_input_with_default "Learning rate for frozen layers" "$LR_FROZEN")
    LR_UNFROZEN=$(read_input_with_default "Learning rate for unfrozen layers" "$LR_UNFROZEN")
    MODEL_PATH=$(read_input_with_default "Model save path" "$MODEL_PATH")
fi

# Ask if user wants to publish to Hugging Face
if read_yes_no "Do you want to publish the model to Hugging Face after training?" "y"; then
    PUBLISH_TO_HF=true
    HF_SPACE=$(read_input_with_default "Hugging Face space name" "$HF_SPACE")
else
    PUBLISH_TO_HF=false
fi

# Confirm settings
echo ""
echo "Training with the following parameters:"
echo "- Frozen epochs: $EPOCHS_FROZEN"
echo "- Unfrozen epochs: $EPOCHS_UNFROZEN"
echo "- Frozen learning rate: $LR_FROZEN"
echo "- Unfrozen learning rate: $LR_UNFROZEN"
echo "- Model save path: $MODEL_PATH"
if $PUBLISH_TO_HF; then
    echo "- Publishing to Hugging Face Space: $HF_SPACE"
else
    echo "- Not publishing to Hugging Face"
fi
echo ""
echo "Logging to: $LOG_FILE"
echo ""

# Confirm to proceed
if ! read_yes_no "Proceed with these settings?" "y"; then
    echo "Operation cancelled by user."
    exit 0
fi

# Function to handle errors
handle_error() {
    echo "Error: $1"
    exit 1
}

# Start training
echo "Starting model training..."
python src/pet_classifier.py --retrain \
    --epochs-frozen $EPOCHS_FROZEN \
    --epochs-unfrozen $EPOCHS_UNFROZEN \
    --lr-frozen $LR_FROZEN \
    --lr-unfrozen $LR_UNFROZEN \
    --model-path $MODEL_PATH 2>&1 | tee $LOG_FILE

if [ $? -ne 0 ]; then
    handle_error "Training failed. Check $LOG_FILE for details."
fi

echo "Training completed successfully!"

# Publish to Hugging Face if selected
if $PUBLISH_TO_HF; then
    echo "Preparing to publish to Hugging Face Spaces: $HF_SPACE..."
    
    # Check if running in GitHub Actions
    if [ -n "$GITHUB_ACTIONS" ] && [ -n "$HUGGINGFACE_TOKEN" ]; then
        echo "Running in GitHub Actions with HUGGINGFACE_TOKEN"
        # Use non-interactive authentication for CI environments
        huggingface-cli login --token "$HUGGINGFACE_TOKEN" --add-to-git-credential
    else
        # Check if huggingface_hub is installed
        if ! pip list | grep -q "huggingface-hub"; then
            echo "Installing huggingface_hub..."
            pip install huggingface-hub
        fi
        
        # Interactive login if not in CI environment
        if ! huggingface-cli whoami &>/dev/null; then
            echo "Please login to Hugging Face:"
            huggingface-cli login
        fi
    fi
    
    # Verify Hugging Face authentication
    if ! huggingface-cli whoami &>/dev/null; then
        handle_error "Failed to authenticate with Hugging Face. Please check your credentials."
    else
        echo "Successfully authenticated with Hugging Face as: $(huggingface-cli whoami)"
    fi
    
    # Check if git-lfs is installed
    if ! command -v git-lfs &> /dev/null; then
        echo "Error: git-lfs is required but not installed."
        echo "Please install git-lfs:"
        echo "  - On Ubuntu/Debian: sudo apt-get install git-lfs"
        echo "  - On macOS: brew install git-lfs"
        echo "  - On Windows: Download from https://git-lfs.github.com/"
        exit 1
    fi

    # Create a temporary directory for Hugging Face upload
    TEMP_DIR=$(mktemp -d)
    echo "Created temporary directory: $TEMP_DIR"
    
    # Initialize git-lfs
    echo "Initializing git-lfs..."
    git lfs install
    
    # Check if the space exists
    echo "Checking if Hugging Face Space exists: $HF_SPACE"
    if ! huggingface-cli repo info spaces/$HF_SPACE &>/dev/null; then
        echo "Warning: Hugging Face Space $HF_SPACE may not exist or you may not have access to it."
        echo "Available spaces for your account:"
        huggingface-cli repo list --spaces
        
        if read_yes_no "Do you want to continue anyway?" "n"; then
            echo "Proceeding with clone attempt..."
        else
            handle_error "Operation cancelled. Please check the space name and your permissions."
        fi
    fi
    
    # Clone with verbose output and error handling
    echo "Cloning Hugging Face space repository..."
    if ! git clone https://huggingface.co/spaces/$HF_SPACE $TEMP_DIR --verbose; then
        echo "Clone failed. Attempting alternative methods..."
        
        # Try with personal access token if available
        if [ -n "$HUGGINGFACE_TOKEN" ]; then
            echo "Trying to clone with personal access token..."
            if ! git clone https://$HUGGINGFACE_TOKEN@huggingface.co/spaces/$HF_SPACE $TEMP_DIR --verbose; then
                handle_error "Failed to clone repository with token. Please check the space name and your permissions."
            fi
        else
            # Try with SSH if available
            echo "Trying to clone with SSH..."
            if ! git clone git@hf.co:spaces/$HF_SPACE $TEMP_DIR --verbose; then
                handle_error "Failed to clone repository with SSH. Please check the space name, your permissions, and SSH setup."
            fi
        fi
    fi
    
    echo "Repository cloned successfully to $TEMP_DIR"
    
    # Copy the trained model and necessary files
    echo "Copying files to the repository..."
    mkdir -p $TEMP_DIR/models
    cp $MODEL_PATH $TEMP_DIR/models/
    cp -r src/* $TEMP_DIR/
    cp requirements.txt $TEMP_DIR/
    
    # Update README if it exists
    if [ -f "$TEMP_DIR/README.md" ]; then
        echo "Updating README.md with latest training information..."
        echo "## Latest Training" >> $TEMP_DIR/README.md
        echo "- Date: $(date)" >> $TEMP_DIR/README.md
        echo "- Frozen epochs: $EPOCHS_FROZEN" >> $TEMP_DIR/README.md
        echo "- Unfrozen epochs: $EPOCHS_UNFROZEN" >> $TEMP_DIR/README.md
        echo "- Frozen learning rate: $LR_FROZEN" >> $TEMP_DIR/README.md
        echo "- Unfrozen learning rate: $LR_UNFROZEN" >> $TEMP_DIR/README.md
    fi
    
    # Commit and push changes
    cd $TEMP_DIR
    git lfs track "*.pkl"
    git add .
    git commit -m "Update model: $(date) - Auto-trained with $EPOCHS_FROZEN frozen and $EPOCHS_UNFROZEN unfrozen epochs"
    git push
    
    if [ $? -ne 0 ]; then
        handle_error "Failed to push to Hugging Face repository. Check your credentials and permissions."
    fi
    
    echo "Successfully published to Hugging Face Spaces: $HF_SPACE"
    echo "Cleaning up temporary directory..."
    rm -rf $TEMP_DIR
    
    echo ""
    echo "=============================================="
    echo "  Process completed successfully!"
    echo "=============================================="
    echo "Model has been trained and published to Hugging Face."
    echo "Visit: https://huggingface.co/spaces/$HF_SPACE"
else
    echo ""
    echo "=============================================="
    echo "  Training completed successfully!"
    echo "=============================================="
    echo "Model has been trained and saved to: $MODEL_PATH"
fi
