#!/bin/bash

# Auto-training and publishing script for Pet Breed Classifier
# This script retrains the model with default parameters and publishes to Hugging Face

# Set default parameters
EPOCHS_FROZEN=5
EPOCHS_UNFROZEN=10
LR_FROZEN=1e-3
LR_UNFROZEN=1e-5
MODEL_PATH="models/pet_classifier.pkl"
HF_SPACE="JinApang/pet_classifier"  # Replace with your actual Hugging Face space name
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

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
echo "Starting automated training with parameters:"
echo "- Frozen epochs: $EPOCHS_FROZEN"
echo "- Unfrozen epochs: $EPOCHS_UNFROZEN"
echo "- Frozen learning rate: $LR_FROZEN"
echo "- Unfrozen learning rate: $LR_UNFROZEN"
echo "- Model save path: $MODEL_PATH"
echo ""
echo "Logging to: $LOG_FILE"
echo ""

# Check if huggingface_hub is installed
if ! pip list | grep -q "huggingface-hub"; then
    echo "Installing huggingface_hub..."
    pip install huggingface-hub
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

# Publish to Hugging Face
echo "Preparing to publish to Hugging Face Spaces: $HF_SPACE..."

# Create a temporary directory for Hugging Face upload
TEMP_DIR=$(mktemp -d)
echo "Created temporary directory: $TEMP_DIR"

# Clone the Hugging Face space repository
echo "Cloning Hugging Face space repository..."
git lfs install
if ! huggingface-cli login; then
    echo "Please login to Hugging Face:"
    huggingface-cli login
fi

git clone https://huggingface.co/spaces/$HF_SPACE $TEMP_DIR || handle_error "Failed to clone Hugging Face space repository"

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
