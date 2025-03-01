import gradio as gr
import psutil
import torch
import GPUtil
from fastai.vision.all import *
from pathlib import Path
from tqdm import tqdm
import datetime

def get_system_stats():
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        return f"CPU: {cpu_percent}% | RAM: {memory.percent}% | GPU Mem: {gpu.memoryUtil*100:.1f}% | GPU Load: {gpu.load*100:.1f}%"
    return f"CPU: {cpu_percent}% | RAM: {memory.percent}%"


def get_breed_from_filename(x):
    return x.name.split('_')[0]

class PetBreedClassifier:
    def __init__(self, model_path=None, force_retrain=False, epochs_frozen=1, epochs_unfrozen=2, lr_frozen=3e-3, lr_unfrozen_min=1e-6, lr_unfrozen_max=1e-4):
        if not force_retrain and model_path and Path(model_path).exists():
            self.learn = load_learner(model_path)
            # Load model metadata if available
            metadata_path = Path(model_path).with_suffix('.meta')
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.loads(f.read())
                print(f"\nModel Information:")
                print(f"Created: {self.metadata.get('created_at')}")
                print(f"Training Parameters:")
                print(f"- Epochs (frozen/unfrozen): {self.metadata.get('epochs_frozen')}/{self.metadata.get('epochs_unfrozen')}")
                print(f"- Learning Rates: {self.metadata.get('lr_frozen')} (frozen), {self.metadata.get('lr_unfrozen_min')}-{self.metadata.get('lr_unfrozen_max')} (unfrozen)")
        else:
            # Initialize DataBlock for training
            dblock = DataBlock(
                blocks=(ImageBlock, CategoryBlock),
                get_items=get_image_files,
                splitter=RandomSplitter(valid_pct=0.2),
                get_y=get_breed_from_filename,
                item_tfms=[Resize(224)],
                batch_tfms=Normalize.from_stats(*imagenet_stats)
            )
            
            # Load data and create model
            path = Path('data')
            if not path.exists():
                raise FileNotFoundError(f"Data directory {path} not found!")
                
            dls = dblock.dataloaders(path)
            # Check for GPU availability
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {device}')
            
            self.learn = vision_learner(dls, resnet34, metrics=error_rate)
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.learn.model = self.learn.model.to(device)
            
            # Train the model with monitoring
            print("\nStarting model training...")
            print("Phase 1: Training with frozen layers")
            self.learn.freeze()
            self.learn.fit_one_cycle(epochs_frozen, lr_frozen)
            print(f"System Stats: {get_system_stats()}")
            
            print("\nPhase 2: Fine-tuning all layers")
            self.learn.unfreeze()
            self.learn.fit_one_cycle(epochs_unfrozen, slice(lr_unfrozen_min, lr_unfrozen_max))
            print(f"System Stats: {get_system_stats()}")
            
            # Save the trained model
            model_path = Path('models')
            model_path.mkdir(exist_ok=True)
            self.learn.export(model_path/'pet_classifier.pkl')
            
            # Save model metadata
            metadata = {
                'created_at': datetime.datetime.now().isoformat(),
                'epochs_frozen': epochs_frozen,
                'epochs_unfrozen': epochs_unfrozen,
                'lr_frozen': lr_frozen,
                'lr_unfrozen_min': lr_unfrozen_min,
                'lr_unfrozen_max': lr_unfrozen_max
            }
            with open(model_path/'pet_classifier.meta', 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\nModel saved to {model_path/'pet_classifier.pkl'}")
            print(f"Model metadata saved to {model_path/'pet_classifier.meta'}")

    
    def predict(self, img):
        # Ensure image is in the correct format
        if isinstance(img, str):
            img = PILImage.create(img)
        
        print(f"\nPrediction System Stats: {get_system_stats()}")
        # Get prediction
        pred, pred_idx, probs = self.learn.predict(img)
        confidence = float(probs[pred_idx])
        
        return {
            'breed': str(pred),
            'confidence': f"{confidence:.2%}",
            'probabilities': {str(c): float(p) for c, p in zip(self.learn.dls.vocab, map(float, probs))}
        }

def create_gradio_interface(classifier):
    def classify_image(image):
        result = classifier.predict(image)
        return f"Breed: {result['breed']}\nConfidence: {result['confidence']}"
    
    # Get the list of breeds from classifier vocabulary
    breeds = classifier.learn.dls.vocab
    breeds_list = ", ".join(breeds)
    
    interface = gr.Interface(
        fn=classify_image,
        inputs=gr.Image(),
        outputs=gr.Textbox(),
        title="Pet Breed Classifier",
        description=f"Upload a pet image to identify its breed.\n\nThis model can classify 37 pet breeds: {breeds_list}",
        theme=gr.themes.Soft()
    )
    
    return interface

def main():
    import argparse
    
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Pet Breed Classifier')
    parser.add_argument('--retrain', action='store_true', help='Force retrain the model even if a saved model exists')
    parser.add_argument('--model-path', type=str, default='models/pet_classifier.pkl', help='Path to the model file')
    parser.add_argument('--epochs-frozen', type=int, default=1, help='Number of epochs for training with frozen layers')
    parser.add_argument('--epochs-unfrozen', type=int, default=2, help='Number of epochs for fine-tuning all layers')
    parser.add_argument('--lr-frozen', type=float, default=3e-3, help='Learning rate for training with frozen layers')
    parser.add_argument('--lr-unfrozen-min', type=float, default=1e-6, help='Minimum learning rate for fine-tuning')
    parser.add_argument('--lr-unfrozen-max', type=float, default=1e-4, help='Maximum learning rate for fine-tuning')
    args = parser.parse_args()
    
    # Initialize classifier with command line arguments
    classifier = PetBreedClassifier(
        model_path=args.model_path,
        force_retrain=args.retrain,
        epochs_frozen=args.epochs_frozen,
        epochs_unfrozen=args.epochs_unfrozen,
        lr_frozen=args.lr_frozen,
        lr_unfrozen_min=args.lr_unfrozen_min,
        lr_unfrozen_max=args.lr_unfrozen_max
    )
    
    # Create and launch Gradio interface
    interface = create_gradio_interface(classifier)
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)

if __name__ == "__main__":
    main()