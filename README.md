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

## Live Demo
## 在线演示

Try out the live demo on Hugging Face Spaces:
在 Hugging Face Spaces 上试用在线演示：

🔗 [Pet Breed Classifier Demo](https://huggingface.co/spaces/JinApang/pet_classifier)

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

## Container Script Usage
## 容器脚本使用说明

The `run_container.sh` script provides a convenient way to manage model training with different configurations.
`run_container.sh` 脚本提供了一种便捷的方式来管理不同配置的模型训练。

### Basic Usage
### 基本用法

```bash
# Start the container script and follow the interactive menu
# 启动容器脚本并按照交互式菜单操作
./run_container.sh
```

The script will present a menu with the following options:
脚本将显示以下选项的菜单：

1. Use existing model (no training)
   使用现有模型（不进行训练）
2. Quick training mode (1 epoch each phase)
   快速训练模式（每个阶段1轮）
3. Full training with custom configuration
   使用自定义配置进行完整训练
4. Use specific model file
   使用特定的模型文件
5. Exit
   退出

### Advanced Usage
### 高级用法

When selecting option 3 (Full training with custom configuration), you can customize the following parameters:
选择选项 3（使用自定义配置进行完整训练）时，您可以自定义以下参数：

- Number of epochs for frozen layers (默认：5)
- Number of epochs for unfrozen layers (默认：10)
- Learning rate for frozen layers (默认：1e-3)
- Minimum learning rate for unfrozen layers (默认：1e-6)
- Maximum learning rate for unfrozen layers (默认：1e-4)

When selecting option 4 (Use specific model file), you can specify the model path relative to the models directory.
选择选项 4（使用特定的模型文件）时，您可以指定相对于 models 目录的模型路径。

### Docker Environment
### Docker 环境

The `run_container.sh` script manages Docker containers with the following features:
`run_container.sh` 脚本管理具有以下特性的 Docker 容器：

- Automatically checks if Docker and NVIDIA Docker runtime are installed
  自动检查是否安装了 Docker 和 NVIDIA Docker 运行时
- Builds the Docker image if it doesn't exist
  如果 Docker 镜像不存在，则构建它
- Mounts data and models directories for persistence
  挂载数据和模型目录以实现持久化
- Configures GPU access for training
  配置用于训练的 GPU 访问权限

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

## Auto Training and Deployment
## 自动训练和部署

The `auto_train.sh` script provides a convenient way to retrain the model with default parameters and publish it to Hugging Face Spaces.
`auto_train.sh` 脚本提供了一种便捷的方式来使用默认参数重新训练模型并将其发布到 Hugging Face Spaces。

### Usage
### 使用方法

```bash
# Make the script executable
# 使脚本可执行
chmod +x auto_train.sh

# Run the auto training and deployment script
# 运行自动训练和部署脚本
./auto_train.sh
```

The script will:
该脚本将：

1. Retrain the model with the following default parameters:
   使用以下默认参数重新训练模型：
   - Frozen epochs: 5
   - Unfrozen epochs: 10
   - Frozen learning rate: 1e-3
   - Unfrozen learning rate: 1e-5

2. Save the model to `models/pet_classifier.pkl`
   将模型保存到 `models/pet_classifier.pkl`

3. Publish the model to your Hugging Face Space
   将模型发布到您的 Hugging Face Space

4. Update the README on Hugging Face with training information
   使用训练信息更新 Hugging Face 上的 README

### GitHub Actions Integration
### GitHub Actions 集成

This project includes a GitHub Actions workflow that can automatically train and publish your model to Hugging Face Spaces.
本项目包含一个 GitHub Actions 工作流，可以自动训练模型并发布到 Hugging Face Spaces。

#### Setting up Hugging Face Token in GitHub
#### 在 GitHub 中设置 Hugging Face 令牌

To allow GitHub Actions to publish to your Hugging Face Space, you need to add your Hugging Face token as a GitHub secret:
要允许 GitHub Actions 发布到您的 Hugging Face Space，您需要将 Hugging Face 令牌添加为 GitHub 密钥：

1. Generate a Hugging Face token:
1. 生成 Hugging Face 令牌：
   - Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
   - Click "New token"
   - Select "Write" access
   - Give it a name (e.g., "GitHub Actions")
   - Click "Generate token"
   - Copy the generated token

2. Add the token to GitHub repository secrets:
2. 将令牌添加到 GitHub 仓库密钥：
   - Go to your GitHub repository
   - Click on "Settings" > "Secrets and variables" > "Actions"
   - Click "New repository secret"
   - Name: `HUGGINGFACE_TOKEN`
   - Value: Paste your Hugging Face token
   - Click "Add secret"

Once configured, the GitHub Actions workflow can be triggered manually from the "Actions" tab in your repository.
配置完成后，可以从仓库的 "Actions" 标签页手动触发 GitHub Actions 工作流。

#### Troubleshooting GitHub Actions
#### GitHub Actions 故障排除

If you encounter authentication issues with Hugging Face in GitHub Actions:
如果您在 GitHub Actions 中遇到 Hugging Face 的身份验证问题：

1. Make sure your `HUGGINGFACE_TOKEN` secret is correctly set in the repository settings
1. 确保您的 `HUGGINGFACE_TOKEN` 密钥在仓库设置中正确设置
2. Check that your token has write permissions for the Hugging Face space
2. 检查您的令牌对 Hugging Face space 具有写入权限
3. Verify the Hugging Face space name in the `auto_train.sh` script is correct
3. 验证 `auto_train.sh` 脚本中的 Hugging Face space 名称是否正确

The workflow is configured to use non-interactive authentication for CI environments, avoiding the terminal interaction issues.
工作流程配置为在 CI 环境中使用非交互式身份验证，避免终端交互问题。

### Requirements
### 要求

- Hugging Face CLI (`pip install huggingface-hub`)
- Git LFS (for handling large model files)
- Valid Hugging Face credentials with write access to your space

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

## Docker Resource Optimization
## Docker 资源优化

This project includes tools and configurations for optimizing Docker resource usage.
本项目包含用于优化 Docker 资源使用的工具和配置。

### Docker Cleanup Script
### Docker 清理脚本

The `docker-cleanup.sh` script helps manage Docker resources and optimize disk space:
`docker-cleanup.sh` 脚本帮助管理 Docker 资源并优化磁盘空间：

```bash
# Make the script executable
# 使脚本可执行
chmod +x docker-cleanup.sh

# Run the Docker cleanup utility
# 运行 Docker 清理工具
./docker-cleanup.sh
```

The script provides options for:
脚本提供以下选项：

- Removing unused containers
  移除未使用的容器
- Removing unused images
  移除未使用的镜像
- Removing unused volumes
  移除未使用的卷
- Removing unused networks
  移除未使用的网络
- Complete system cleanup
  完整的系统清理
- Advanced cleanup (including builder cache)
  高级清理（包括构建缓存）

### Docker Optimization Tips
### Docker 优化技巧

1. **Layer Caching**: Structure your Dockerfile with frequently changing content at the bottom to maximize cache usage.
   **层缓存**: 将频繁变化的内容放在 Dockerfile 底部，以最大化缓存使用。

2. **Regular Cleanup**: Run the cleanup script regularly to prevent excess resource usage:
   **定期清理**: 定期运行清理脚本以防止过度资源使用：
   ```bash
   # Remove unused containers
   docker container prune

   # Remove unused images
   docker image prune

   # Remove all unused Docker objects
   docker system prune
   ```

3. **Multi-stage Builds**: Use multi-stage builds in your Dockerfile to reduce final image size.
   **多阶段构建**: 在 Dockerfile 中使用多阶段构建以减小最终镜像大小。

4. **Volume Mounting**: Use volume mounting for development to avoid rebuilding images.
   **卷挂载**: 在开发中使用卷挂载，避免重新构建镜像。

5. **Resource Limits**: Configure resource limits in your Docker Compose or run commands.
   **资源限制**: 在 Docker Compose 或运行命令中配置资源限制。

## License
## 许可证

MIT License
MIT 许可证
