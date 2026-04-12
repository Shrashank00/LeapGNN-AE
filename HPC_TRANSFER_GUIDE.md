# LeapGNN: HPC Cluster Transfer Guide

## Overview
This guide explains how to transfer LeapGNN code and dependencies to an HPC cluster.

---

## Step 1: Transfer Code to HPC

### Option A: Using Git (Recommended)

```bash
# On HPC cluster, clone the repository
ssh user@hpc-cluster
cd /path/to/workspace

git clone https://github.com/ISCS-ZJU/LeapGNN-AE.git
cd LeapGNN-AE
git checkout distributed_version
git submodule init
git submodule update
```

### Option B: Using SCP (if Git is not available or for private repo)

```bash
# From your local machine
scp -r /data/shrashank/leapgnn/LeapGNN-AE user@hpc-cluster:/path/to/workspace/

# SSH into HPC and verify
ssh user@hpc-cluster
cd /path/to/workspace/LeapGNN-AE
```

### Option C: Using Rsync (fastest for large transfers with resume capability)

```bash
# From your local machine (best for resumable transfers)
rsync -avz --progress --delete \
  /data/shrashank/leapgnn/LeapGNN-AE/ \
  user@hpc-cluster:/path/to/workspace/LeapGNN-AE/

# Resume if interrupted
rsync -avz --progress --delete \
  /data/shrashank/leapgnn/LeapGNN-AE/ \
  user@hpc-cluster:/path/to/workspace/LeapGNN-AE/
```

---

## Step 2: Transfer Wheels (PyTorch Geometric packages)

Your project has pre-downloaded `.whl` files. Transfer them:

```bash
# From your local machine
scp torch_*.whl user@hpc-cluster:/path/to/workspace/LeapGNN-AE/
```

Or use rsync:
```bash
rsync -avz --progress torch_*.whl user@hpc-cluster:/path/to/workspace/LeapGNN-AE/
```

---

## Step 3: Prepare Dataset on HPC

### Option A: Transfer Dataset (if small)
```bash
rsync -avz --progress /data/shrashank/leapgnn/dataset/ \
  user@hpc-cluster:/path/to/workspace/dataset/
```

### Option B: Download Directly on HPC (Recommended for large datasets)

```bash
ssh user@hpc-cluster
cd /path/to/workspace/LeapGNN-AE

# Download preprocessed datasets from Zenodo
wget https://zenodo.org/records/14557307/files/datasets.tar.gz
tar -xzf datasets.tar.gz -C dataset/
```

### Option C: Use Shared Storage (Best for HPC)

Most HPC clusters have shared storage. Check with your HPC admin:
```bash
# Copy to shared storage that's accessible from compute nodes
cp -r dataset/ /shared/storage/path/datasets_leapgnn/
```

---

## Step 4: Run Setup Script on HPC

```bash
ssh user@hpc-cluster
cd /path/to/workspace/LeapGNN-AE

# Make script executable
chmod +x hpc_setup.sh

# Run setup (this may take 20-30 minutes)
./hpc_setup.sh

# Or run step-by-step if there are issues:
conda create -n repgnn python=3.9 -y
conda activate repgnn
pip install -r hpc_requirements.txt
# ... etc
```

---

## Step 5: Configure for HPC Environment

### Update SBATCH Script Example

Create `hpc_submit.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=leapgnn_test
#SBATCH --partition=gpu              # Adjust to your HPC queue name
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=1          # 1 task per node (for distributed training)
#SBATCH --gpus-per-node=1            # 1 GPU per node (adjust as needed)
#SBATCH --time=04:00:00              # Time limit HH:MM:SS
#SBATCH --output=logs/slurm_%j.out   # Output log file
#SBATCH --error=logs/slurm_%j.err    # Error log file
#SBATCH --mem=128G                   # Memory per node (adjust as needed)

# Load necessary modules (adjust based on your HPC cluster)
module load cuda/11.7                # Or your installed CUDA version
module load gcc/9.3                  # Or compatible GCC

# Activate conda environment
source activate repgnn

# Get compute node IPs for distributed training
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=29500
RANK=$SLURM_PROCID
WORLD_SIZE=$(($SLURM_NTASKS))

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export RANK=$RANK
export WORLD_SIZE=$WORLD_SIZE

# Change to workspace directory
cd /path/to/workspace/LeapGNN-AE

# Update data path if necessary
DATA_PATH="/path/to/datasets/ogbn_arxiv"  # Update this path

# Run your training script
python dgl_jpgnn_trans.py \
    --data_path $DATA_PATH \
    --num_epochs 10 \
    --batch_size 1024 \
    --num_workers 4

# If using PyTorch DDP:
# python -m torch.distributed.launch \
#     --nproc_per_node=$SLURM_GPUS_PER_NODE \
#     --nnodes=$SLURM_NNODES \
#     --node_rank=$SLURM_NODEID \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     dgl_jpgnn_trans.py --args
```

### Submit Job

```bash
sbatch hpc_submit.sbatch
```

---

## Step 6: Verify Installation on HPC

After setup, verify everything works:

```bash
conda activate repgnn

# Test PyTorch and CUDA
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}')"

# Test DGL
python -c "import dgl; print(f'DGL version: {dgl.__version__}')"

# Test torch_geometric
python -c "import torch_geometric; print('PyG OK')"

# Test a simple script with distributed setup
python -c "import torch.distributed as dist; print('Distributed training available')"
```

---

## Important HPC Considerations

### 1. **Module System**
Many HPC clusters use module systems. Check available modules:
```bash
module avail
module load cuda/11.7          # Load required modules
module load gcc/9.3
```

### 2. **Conda in HPC**
Some HPC clusters limit conda usage. If needed:
```bash
# Install miniconda locally
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
export PATH=~/miniconda3/bin:$PATH
```

### 3. **Network Configuration**
For distributed training, ensure:
- All compute nodes can communicate
- Update `MASTER_ADDR` and `MASTER_PORT` in your scripts
- Test connectivity: `srun -N4 hostname` (on 4 nodes)

### 4. **GPU Access**
Verify GPU visibility:
```bash
srun -N4 nvidia-smi
```

### 5. **Storage Paths**
- **Home directory**: Usually has limited space (`~/`)
- **Scratch/Temp**: Fast local storage on compute nodes (`/tmp`, `/scratch`)
- **Shared storage**: Slower but persistent (`/shared`, `/data`)

Update paths in your scripts accordingly!

### 6. **Troubleshooting**

**Issue**: CUDA not found
```bash
# Solution: Load CUDA module
module load cuda/11.7
```

**Issue**: Cannot import DGL
```bash
# Solution: Rebuild DGL from source (as per hpc_setup.sh)
cd 3rdparties/dgl && bash rebuild.sh
```

**Issue**: Out of memory (OOM)
```bash
# Solution: Increase memory allocation in SBATCH
#SBATCH --mem=256G
```

---

## Quick Reference Commands

| Task | Command |
|------|---------|
| Check job status | `squeue -u $USER` |
| Cancel job | `scancel JOB_ID` |
| Check node availability | `sinfo --Node` |
| Check GPU availability | `sinfo --gres` |
| Run interactive job | `srun -N2 --gpus-per-node=1 --pty bash` |
| Monitor running job | `sstat -j JOB_ID --format=AveCPU,AveVMSize` |

---

## Contact & Support

- For HPC-specific issues, contact your HPC cluster administrator
- For LeapGNN setup issues, refer to the main [README.md](README.md)
- Original authors: See README.md for contact info
