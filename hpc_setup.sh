#!/bin/bash
# HPC Cluster Setup Script for LeapGNN
# Run this script on the HPC cluster after transferring the code

set -e

# Configuration
PYTHON_VERSION=3.9
ENV_NAME=repgnn
CUDA_VERSION=11.7  # Adjust based on your HPC's CUDA version

echo "=========================================="
echo "LeapGNN HPC Cluster Setup Script"
echo "=========================================="

# Step 1: Create conda environment
echo "[Step 1] Creating conda environment..."
conda create -n $ENV_NAME python==$PYTHON_VERSION -y
source activate $ENV_NAME

# Step 2: Install Python dependencies
echo "[Step 2] Installing Python dependencies..."
pip install --upgrade pip
pip install -r hpc_requirements.txt

# Step 3: Install PyTorch (adjust CUDA version as needed for your HPC)
echo "[Step 3] Installing PyTorch with CUDA support..."
pip install torch==1.10.1 torchvision==0.11.2 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Step 4: Install torch_scatter, torch_cluster, torch_sparse (if .whl files exist)
echo "[Step 4] Installing torch geometric packages..."
if [ -f "torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl" ]; then
    pip install torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
fi
if [ -f "torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl" ]; then
    pip install torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
fi
if [ -f "torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl" ]; then
    pip install torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl
fi

# Step 5: Install Go and gRPC (if not already available as module)
echo "[Step 5] Setting up Go and gRPC..."
mkdir -p ~/tools

# Check if Go is already available
if ! command -v go &> /dev/null; then
    echo "Installing Go..."
    cd ~/tools
    wget https://go.dev/dl/go1.19.3.linux-amd64.tar.gz
    tar -xzf go1.19.3.linux-amd64.tar.gz
    export PATH=$PATH:$HOME/tools/go/bin
    echo "export PATH=$PATH:$HOME/tools/go/bin" >> ~/.bashrc
fi

# Setup protoc if needed
if ! command -v protoc &> /dev/null; then
    echo "Installing protoc..."
    mkdir -p ~/.local/bin
    cd ~/tools
    PB_REL="https://github.com/protocolbuffers/protobuf/releases"
    curl -LO $PB_REL/download/v3.12.1/protoc-3.12.1-linux-x86_64.zip
    unzip -o protoc-3.12.1-linux-x86_64.zip -d $HOME/.local
    export PATH="$PATH:$HOME/.local/bin"
    echo "export PATH=$PATH:$HOME/.local/bin" >> ~/.bashrc
fi

# Step 6: Build DGL from source
echo "[Step 6] Building DGL from source..."
cd 3rdparties/dgl
git submodule init
git submodule update

# Check if CUDA module is available on HPC
if command -v module &> /dev/null; then
    echo "Loading CUDA module (adjust module name as needed for your HPC)..."
    module load cuda  # or cuda/11.7 depending on your HPC setup
fi

rm -rf build && mkdir build && cd build
cmake -DUSE_CUDA=ON ..
make -j$(nproc)
cd ../python && python setup.py install

# Step 7: Verify installation
echo "[Step 7] Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import dgl; print(f'DGL version: {dgl.__version__}')"
python -c "import torch_scatter; print('torch_scatter: OK')"
python -c "import torch_geometric; print(f'PyG version: {torch_geometric.__version__}')"

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "IMPORTANT: Before running jobs, ensure:"
echo "  1. Dataset is transferred to HPC cluster"
echo "  2. Update data paths in your scripts"
echo "  3. Modify SBATCH scripts with HPC-specific settings (partition, QoS, time, etc.)"
