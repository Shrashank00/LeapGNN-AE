#!/bin/bash
#SBATCH --job-name=leapgnn_cpu_2nodes
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=leapgnn_cpu_%j.log
#SBATCH --error=leapgnn_cpu_%j.err

# ====================== CONDA SETUP ======================
source /data/anaconda3/etc/profile.d/conda.sh
conda activate /data/shrashank/conda_envs/leapgnn
# ============================================================

module load gcc-9.3.0

echo "=== LeapGNN OGBn-Arxiv CPU 2-Node Training ==="
echo "Hostname      : $(hostname)"
echo "Node ID       : $SLURM_NODEID"
echo "CPUs/Task     : $SLURM_CPUS_PER_TASK"
echo "================================================="

# Setup master address for distributed training
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
echo "Master Address: $MASTER_ADDR"
echo "Master Port   : $MASTER_PORT"
echo "================================================="

cd /data/shrashank/leapgnn/LeapGNN-AE

echo "=== Starting Training at $(date) ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Model: graphsage (CPU)"
echo "================================================="

# Run training on both nodes (removed --eval to prevent hanging)
srun -N 2 --ntasks=2 --ntasks-per-node=1 --unbuffered \
    python dgl_single.py \
        --world-size 2 \
        --rank $SLURM_PROCID \
        --dist-url "tcp://$MASTER_ADDR:29500" \
        --dataset ogbn_arxiv0 \
        --batch-size 128 \
        --sampling 5-5 \
        --epoch 5 \
        --hidden-size 128 \
        --model graphsage \
        --lr 0.001 \
        --log

TRAIN_EXIT=$?
echo "=== Training Finished at $(date) with exit code $TRAIN_EXIT ==="

if [ $TRAIN_EXIT -eq 0 ]; then
    echo "✓ Training completed successfully"
    ls -lt logs/test_graphsage_ogbn_arxiv0_trainer2* 2>/dev/null | head -1
else
    echo "✗ Training failed with exit code $TRAIN_EXIT"
fi
