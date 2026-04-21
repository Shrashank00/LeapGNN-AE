#!/bin/bash
#SBATCH --job-name=leapgnn_lessjp_cpu_nc
#SBATCH --partition=shared
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=leapgnn_lessjp_cpu_nc_%j.log
#SBATCH --error=leapgnn_lessjp_cpu_nc_%j.err

set -euo pipefail
set -x

export WORKSPACE="/scratch/vishweshj_iitbhilai/shrashank/framework/LeapGNN-AE"
PY="/scratch/vishweshj_iitbhilai/shrashank/conda/leapgnn/bin/python"

cd "$WORKSPACE"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

EPOCHS=5
BATCH=128
SAMPLING="10-10"
HIDDEN=256
WORKERS=0

# 1) sanity check: confirm DGL import on BOTH nodes
srun -N 2 --ntasks=2 --ntasks-per-node=1 --unbuffered --label --kill-on-bad-exit=1 \
  bash -lc "hostname; echo RANK=\$SLURM_PROCID; $PY -V; $PY -c 'import dgl; print(\"DGL_OK\", dgl.__version__)'"

# 2) run
srun -N 2 --ntasks=2 --ntasks-per-node=1 --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --unbuffered --label --kill-on-bad-exit=1 \
  bash -lc "hostname; echo RANK=\$SLURM_PROCID; \
    $PY -u dgl_lessjp_cpu_nocache.py \
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
      --lr 0.001"
