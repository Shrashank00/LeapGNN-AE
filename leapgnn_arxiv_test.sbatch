#!/bin/bash
#SBATCH --job-name=leapgnn_arxiv_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=leapgnn_arxiv_%j.log
#SBATCH --error=leapgnn_arxiv_%j.err

# ====================== CENTRAL CONDA FIX ======================
source /data/anaconda3/etc/profile.d/conda.sh
conda activate /data/shrashank/conda_envs/leapgnn
# ============================================================

module load gcc-9.3.0
module load cuda-11.8

echo "=== LeapGNN OGBn-Arxiv Test Started (2-Node Distributed) ==="
echo "Hostname      : $(hostname)"
echo "Node ID       : $SLURM_NODEID"
echo "GPU           : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Python        : $(which python)"
echo "Python version: $(python --version)"
echo "================================================="

# Setup master address for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
echo "Master Address: $MASTER_ADDR"
echo "Master Port   : $MASTER_PORT"
echo "================================================="

cd /data/shrashank/leapgnn/LeapGNN-AE

echo "=== Starting Training on Both Nodes at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node List: $SLURM_JOB_NODELIST"  
echo "Master: $MASTER_ADDR:29500"
echo "Model: graphsage"
echo "================================================="

# Use srun to launch on both nodes with proper rank assignment
srun -N 2 --ntasks=2 --ntasks-per-node=1 --unbuffered \
    bash -c 'python dgl_single.py \
        --world-size 2 \
        --rank $SLURM_PROCID \
        --dist-url "tcp://'$MASTER_ADDR':29500" \
        --dataset ogbn_arxiv0 \
        --batch-size 1024 \
        --sampling 25-25 \
        --epoch 20 \
        --hidden-size 256 \
        --model graphsage \
        --lr 0.0001 \
        --log \
        --eval'

TRAIN_EXIT=$?
echo "=== Training Finished at $(date) with exit code $TRAIN_EXIT ==="

