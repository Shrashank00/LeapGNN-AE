#!/bin/bash
#SBATCH --job-name=metis_2parts
#SBATCH --partition=shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=make_metis_%j.log
#SBATCH --error=make_metis_%j.err

set -euo pipefail

export WORKSPACE="/scratch/vishweshj_iitbhilai/shrashank/framework/LeapGNN-AE"
export CONDA_ENV="/scratch/vishweshj_iitbhilai/shrashank/conda/leapgnn"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

cd "$WORKSPACE"

python -u make_metis_partitions.py --dataset ogbn_arxiv0 --world-size 2

ls -l ogbn_arxiv0/dist_True/2_metis
