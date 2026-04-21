#!/bin/bash
#SBATCH --job-name=leapgnn_lessjp_cpu
#SBATCH --partition=shared
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=leapgnn_lessjp_cpu_%j.log
#SBATCH --error=leapgnn_lessjp_cpu_%j.err

set -euo pipefail
set -x

module purge
module load compiler/gcc/9.5.0

export WORKSPACE="/scratch/vishweshj_iitbhilai/shrashank/framework/LeapGNN-AE"
export CONDA_ENV="/scratch/vishweshj_iitbhilai/shrashank/conda/leapgnn"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

cd "$WORKSPACE"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

EPOCHS=1
BATCH=16
SAMPLING="1-1"
HIDDEN=32
WORKERS=0

srun -N 2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --unbuffered --label --kill-on-bad-exit=1 \
  bash -lc "hostname; echo RANK=\$SLURM_PROCID; \
    python -u dgl_jpgnn_trans_lessjp_cpu.py \
      --world-size 2 \
      --rank \$SLURM_PROCID \
      --dist-url tcp://$MASTER_ADDR:$MASTER_PORT \
      --dataset ogbn_arxiv0 \
      --model-name graphsage \
      --epoch $EPOCHS \
      --batch-size $BATCH \
      --sampling $SAMPLING \
      --hidden-size $HIDDEN \
      --num-worker $WORKERS \
      --lr 0.001 \
      --log"
