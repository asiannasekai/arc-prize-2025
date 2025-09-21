#!/bin/bash
# GPU Environment Setup for ARC Federated System

echo "🚀 Setting up GPU environment for ARC optimization..."

# Update system
sudo apt update && sudo apt upgrade -y

# Install CUDA dependencies (if not already installed)
sudo apt install -y nvidia-driver-525 nvidia-cuda-toolkit

# Install Python GPU packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers[torch] accelerate
pip install nvidia-ml-py3

# Verify GPU setup
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"

# Clone repository (if needed)
# git clone https://github.com/asiannasekai/arc-prize-2025.git
# cd arc-prize-2025

# Install project dependencies
pip install -r requirements.txt

# Start Neo4j
./neo4j-community-5.15.0/bin/neo4j start

echo "✅ GPU environment ready for ARC optimization!"
