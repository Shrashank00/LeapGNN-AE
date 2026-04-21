#!/bin/bash
#SBATCH --job-name=leapgnn_cpu_2nodes
#SBATCH --partition=shared
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=leapgnn_cpu_%j.log
#SBATCH --error=leapgnn_cpu_%j.err

set -euo pipefail

module purge
module load compiler/gcc/9.5.0

# repo path
export WORKSPACE="/scratch/vishweshj_iitbhilai/shrashank/framework/LeapGNN-AE"

# Use env directly (no conda activate needed)
export CONDA_ENV="/scratch/vishweshj_iitbhilai/shrashank/conda/leapgnn"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1

cd "$WORKSPACE"

echo "=== CPU 2-node LeapGNN run ==="
echo "Submit dir: $SLURM_SUBMIT_DIR"
echo "Workspace : $WORKSPACE"
echo "Python    : $(which python)"
echo "Version   : $(python --version)"
echo "Nodes     : $SLURM_JOB_NODELIST"
echo "===================================="

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"
echo "===================================="

# FAST sanity parameters for CPU (adjust later)
EPOCHS=2
BATCH=64
SAMPLING="2-2"
HIDDEN=64
WORKERS=1

srun -N 2 --ntasks=2 --ntasks-per-node=1 --unbuffered \
  bash -c "python dgl_single.py \
    --world-size 2 \
    --rank \$SLURM_PROCID \
    --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
    --dataset ogbn_arxiv0 \
    --model graphsage \
    --epoch $EPOCHS \
    --batch-size $BATCH \
    --sampling $SAMPLING \
    --hidden-size $HIDDEN \
    --num-worker $WORKERS \
    --lr 0.001 \
    --log"
