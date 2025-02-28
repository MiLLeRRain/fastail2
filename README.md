# FastAI Pet Breed Classifier
# FastAI 宠物品种分类器

A deep learning pet breed classification system built with FastAI, PyTorch, and Gradio.
基于 FastAI、PyTorch 和 Gradio 构建的深度学习宠物品种分类系统。

## Project Overview
## 项目概述

This project implements a pet breed classification system with the following features:
本项目实现了一个具有以下特性的宠物品种分类系统：

- Real-time pet breed classification using ResNet34 pre-trained model
- 使用 ResNet34 预训练模型进行实时宠物品种分类
- Web interface for image upload and prediction
- 用于图片上传和预测的 Web 界面
- System resource monitoring (CPU, RAM, GPU)
- 系统资源监控（CPU、内存、GPU）
- Docker deployment support
- Docker 容器化部署支持

## Project Structure
## 项目结构

```
/pet-classifier
├── notebooks/          # Jupyter notebooks for development
├── src/               # Source code
├── docker/            # Docker configuration
├── data/              # Training data directory
├── models/            # Saved model directory
├── requirements.txt   # Project dependencies
└── README.md         # Project documentation
```

## Dataset Preparation
## 数据集准备

Use the provided script to download and extract the Oxford-IIIT Pet Dataset:
使用提供的脚本下载并解压 Oxford-IIIT Pet Dataset：

```bash
# Download and extract the dataset
# 下载并解压数据集
./download_dataset.sh
```

This will automatically download and prepare the dataset in the correct format for training.
这将自动下载数据集并准备好用于训练的正确格式。

## Run Trainer Script Usage
## 训练脚本使用说明

The `run_trainer.sh` script provides a convenient way to manage model training with different configurations.
`run_trainer.sh` 脚本提供了一种便捷的方式来管理不同配置的模型训练。

### Basic Usage
### 基本用法

```bash
# Start training with default settings
# 使用默认设置开始训练
./run_trainer.sh

# Force retrain the model
# 强制重新训练模型
./run_trainer.sh --retrain

# Quick training mode (1 epoch each phase)
# 快速训练模式（每个阶段1轮）
./run_trainer.sh --quick-train
```

### Advanced Usage
### 高级用法

```bash
# Custom epochs and learning rates
# 自定义训练轮次和学习率
./run_trainer.sh --retrain \
    --epochs-frozen 5 \
    --epochs-unfrozen 10 \
    --lr-frozen 1e-3 \
    --lr-unfrozen 1e-5

# Train with larger image size
# 使用更大的图片尺寸训练
./run_trainer.sh --retrain \
    --image-size 448 \
    --batch-size 16

# Save model to custom path
# 将模型保存到自定义路径
./run_trainer.sh --retrain \
    --model-path models/custom_model.pkl
```

### Environment Variables
### 环境变量

You can configure the following environment variables before running the script:
在运行脚本之前，你可以配置以下环境变量：

```bash
# Set CUDA device
# 设置 CUDA 设备
export CUDA_VISIBLE_DEVICES=0

# Set number of workers for data loading
# 设置数据加载的工作进程数
export NUM_WORKERS=4
```

## Docker Deployment
## Docker 部署

1. Build Docker image:
1. 构建 Docker 镜像：
```bash
docker build -t pet-classifier .
```

2. Run container with different configurations:
2. 使用不同配置运行容器：

### Basic run (use existing model if available, train if not):
### 基本运行（如果有现有模型则使用，否则训练）：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier
```

### Force retrain (ignore existing model):
### 强制重新训练（忽略现有模型）：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier \
    python src/pet_classifier.py --retrain
```

### Use specific model without training:
### 使用指定模型（不进行训练）：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier \
    python src/pet_classifier.py --model-path models/my_custom_model.pkl
```

### Quick training mode (1 epoch each phase):
### 快速训练模式（每个阶段 1 轮）：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier \
    python src/pet_classifier.py --retrain --quick-train
```

### Custom Training Epochs:
### 自定义训练轮次：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier \
    python src/pet_classifier.py --retrain --epochs-frozen 5 --epochs-unfrozen 10
```

### Full training with custom configuration:
### 使用自定义配置的完整训练：
```bash
docker run --gpus all --shm-size=1g -p 7860:7860 \
    -v ./data:/app/data \
    -v ./models:/app/models \
    --name pet-classifier \
    --rm pet-classifier \
    python src/pet_classifier.py --retrain \
        --epochs-frozen 5 \
        --epochs-unfrozen 10 \
        --lr-frozen 1e-3 \
        --lr-unfrozen 1e-5 \
        --model-path models/custom_model.pkl
```

## Development Environment
## 开发环境

- Windows + WSL2
- Python virtual environment（Python 虚拟环境）
- VSCode + JupyterLab
- GPU: 12GB VRAM
- RAM: 32GB

## Tech Stack
## 技术栈

- FastAI + PyTorch (ResNet34 pre-trained model)
- JupyterLab (development environment)
- Gradio (Web UI)
- Docker (deployment)

## Setup Instructions
## 安装说明

1. Create and activate Python virtual environment:
1. 创建并激活 Python 虚拟环境：
```bash
python -m venv venv
source venv/bin/activate  # On Linux/WSL2
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies:
2. 安装依赖：
```bash
pip install -r requirements.txt
```

3. Prepare your training data in the `data` directory with the following structure:
3. 在 `data` 目录中准备训练数据，结构如下：
```
data/
  ├── breed1_image1.jpg
  ├── breed1_image2.jpg
  ├── breed2_image1.jpg
  └── ...
```
Note: Image filenames should start with the breed name followed by an underscore.
注意：图片文件名应以品种名称开头，后跟下划线。

## Training and Running the Model
## 模型训练与运行

### Basic Usage python
### 基本用法 python

Start the classifier with default settings:
使用默认设置启动分类器：
```bash
python src/pet_classifier.py
```
This will:
这将会：
- Load the pre-trained model if available at `models/pet_classifier.pkl`
- 如果存在，加载位于 `models/pet_classifier.pkl` 的预训练模型
- Train a new model if no pre-trained model exists
- 如果没有预训练模型，训练一个新模型
- Launch the Gradio web interface at http://localhost:7860
- 在 http://localhost:7860 启动 Gradio Web 界面

### Training Options
### 训练选项

1. Force retrain the model (ignore existing model):
1. 强制重新训练模型（忽略现有模型）：
```bash
python src/pet_classifier.py --retrain
```

2. Use a specific model file:
2. 使用指定的模型文件：
```bash
python src/pet_classifier.py --model-path models/my_custom_model.pkl
```

3. Force retrain with a custom save path:
3. 强制重新训练并使用自定义保存路径：
```bash
python src/pet_classifier.py --retrain --model-path models/new_model.pkl
```

### Advanced Training Configurations
### 高级训练配置

1. Train with more epochs (5 epochs for frozen layers, 10 for fine-tuning):
1. 使用更多训练轮次（冻结层 5 轮，微调 10 轮）：
```bash
python src/pet_classifier.py --retrain --epochs-frozen 5 --epochs-unfrozen 10
```

2. Train with custom learning rates:
2. 使用自定义学习率：
```bash
python src/pet_classifier.py --retrain --lr-frozen 1e-3 --lr-unfrozen 1e-5
```

3. Combined training configuration:
3. 组合训练配置：
```bash
python src/pet_classifier.py --retrain \
    --epochs-frozen 5 \
    --epochs-unfrozen 10 \
    --lr-frozen 1e-3 \
    --lr-unfrozen 1e-5 \
    --model-path models/custom_model.pkl
```

4. Quick training mode (1 epoch each phase):
4. 快速训练模式（每个阶段 1 轮）：
```bash
python src/pet_classifier.py --retrain --quick-train
```

5. Full training mode with larger image size:
5. 使用更大图片尺寸的完整训练模式：
```bash
python src/pet_classifier.py --retrain --image-size 448 --batch-size 16
```

## Usage
## 使用说明

1. Access the web interface at http://localhost:7860
1. 访问 Web 界面：http://localhost:7860
2. Upload a pet image
2. 上传宠物图片
3. Get real-time breed classification results with confidence scores
3. 获取实时品种分类结果和置信度分数
4. Monitor system resource usage during inference
4. 监控推理过程中的系统资源使用情况

## Development Workflow
## 开发工作流程

1. Local development (Python virtual environment + VSCode)
1. 本地开发（Python 虚拟环境 + VSCode）
2. Model training and testing (Jupyter Notebook)
2. 模型训练和测试（Jupyter Notebook）
3. Packaging and deployment (Docker)
3. 打包和部署（Docker）

## License
## 许可证

MIT License
MIT 许可证
