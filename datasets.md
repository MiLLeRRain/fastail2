# Recommended Datasets for Image Classification

## General Purpose Datasets

### 1. CIFAR-10
- **Description**: 60,000 32x32 color images in 10 classes (6,000 images per class)
- **Size**: ~170 MB
- **Download**: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
- **Structure**: FastAI compatible format after extraction
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### 2. ImageNette
- **Description**: A subset of 10 easily classified classes from ImageNet
- **Size**: 1.3 GB (full) or 400 MB (160px)
- **Download**: https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tar.gz
- **Structure**: Already organized in FastAI-compatible format
- **Classes**: tench, English springer, cassette player, chain saw, church, French horn, garbage truck, gas pump, golf ball, parachute

### 3. Food-101
- **Description**: 101 food categories with 101,000 images
- **Size**: 5 GB
- **Download**: http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz
- **Structure**: Organized by category, FastAI compatible
- **Ideal for**: Restaurant menu item classification, food recognition systems

## Domain-Specific Datasets

### 1. Stanford Cars Dataset
- **Description**: 196 car classes with ~8,144 training images
- **Size**: 1.9 GB
- **Download**: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- **Ideal for**: Vehicle classification systems

### 2. Oxford Flowers-102
- **Description**: 102 flower categories with ~8,189 images
- **Size**: 330 MB
- **Download**: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- **Ideal for**: Plant and flower classification

### 3. Oxford-IIIT Pet Dataset
- **Description**: 37 categories of pet breeds (25 dog breeds and 12 cat breeds)
- **Size**: ~750 MB
- **Download**: https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
- **Structure**: Requires reorganization for FastAI format
- **Classes**: Various dog and cat breeds including Persian, Maine Coon, British Shorthair, Chihuahua, Pug, etc.
- **Ideal for**: Pet breed classification, beginner-friendly projects

## Dataset Structure for FastAI

To use these datasets with FastAI, organize the images in this structure:

```
/data
  /train
    /class1
      image1.jpg
      image2.jpg
    /class2
      image3.jpg
      image4.jpg
  /valid
    /class1
      image5.jpg
    /class2
      image6.jpg
```

## Loading Data in FastAI

```python
from fastai.vision.all import *

# Set your data path
path = Path('data')

# Create DataBlock
dblock = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(),
    get_y=parent_label
)

# Create DataLoaders
dls = dblock.dataloaders(path)
```

## Memory Considerations

With your 12GB VRAM GPU:
- For larger datasets, consider using smaller image sizes (e.g., 224x224)
- Adjust batch size based on memory usage (start with 32-64)
- Use mixed precision training to reduce memory footprint

## Download and Preparation Tips

1. Use `wget` or `curl` to download datasets
2. Extract using `tar -xzf` for .tar.gz files
3. Verify folder structure matches FastAI requirements
4. Consider using `fastai.vision.download_images()` for custom datasets