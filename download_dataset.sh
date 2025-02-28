#!/bin/bash

# 创建必要的目录
mkdir -p data/images

# 下载数据集
echo "Downloading Oxford-IIIT Pet Dataset..."
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P data/

# 解压数据集
echo "Extracting dataset..."
tar -xzf data/images.tar.gz -C data/images/

# 清理下载文件
rm data/images.tar.gz

echo "Dataset downloaded and extracted successfully!"