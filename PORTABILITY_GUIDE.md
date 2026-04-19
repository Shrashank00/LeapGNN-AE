# LeapGNN CPU 2-Node Training - Portability Guide

This guide explains how to adapt the working setup to another HPC cluster.

## Prerequisites Check

Before migrating, verify the target cluster has:
- [ ] SLURM job scheduler
- [ ] 2+ compute nodes available
- [ ] Python 3.9+
- [ ] PyTorch + DGL compatible with system (CPU or GPU)

## Step 1: Adapt the SLURM Directives

Edit `leapgnn_cpu_simple.sbatch` for your cluster's resource constraints:

```bash
#SBATCH --nodes=2              # Keep as 2 for 2-node training
#SBATCH --ntasks-per-node=1   # 1 process per node
#SBATCH --cpus-per-task=16    # Adjust to your cluster's node capacity
#SBATCH --time=04:00:00       # Adjust if CPU training is slower (up to 8 hours)
#SBATCH --partition=gpu        # CHANGE: Use your cluster's partition name
```

**Common partition names:**
- `gpu`, `gpuq`, `compute-gpu` (for GPU nodes)
- `cpu`, `cpuq`, `compute`, `standard` (for CPU nodes)

Check available partitions:
```bash
sinfo -o "%P %a %D" | head -10
```

## Step 2: Update Environment Setup

Replace paths in the sbatch script:

```bash
# BEFORE (LeapGNN cluster):
source /data/anaconda3/etc/profile.d/conda.sh
conda activate /data/shrashank/conda_envs/leapgnn
module load gcc-9.3.0

# AFTER (your cluster):
source /path/to/your/anaconda/etc/profile.d/conda.sh
conda activate /path/to/your/leapgnn_env

# Check available modules:
module avail gcc      # or module avail compiler
# Load appropriate compiler
module load your_gcc_version
```

## Step 3: Prepare the Dataset

The setup expects `ogbn_arxiv0/` in the working directory.

**Option A: Copy from current cluster**
```bash
scp -r /data/shrashank/leapgnn/LeapGNN-AE/ogbn_arxiv0/ \
    your_user@new_cluster:/path/to/leapgnn/
```

**Option B: Re-download on new cluster**
```bash
cd /path/to/leapgnn/LeapGNN-AE
python data/get_data.py --dataset ogbn_arxiv0
```

## Step 4: Update Sbatch Script Paths

In `leapgnn_cpu_simple.sbatch`:

```bash
# BEFORE:
cd /data/shrashank/leapgnn/LeapGNN-AE

# AFTER:
cd /path/to/your/leapgnn/LeapGNN-AE
```

Also update conda activation if needed:
```bash
# BEFORE:
conda activate /data/shrashank/conda_envs/leapgnn

# AFTER:
conda activate /your/new/conda/env/path or just use: conda activate leapgnn
```

## Step 5: Verify Communication Setup

Ensure nodes can communicate:

```bash
# Test with simple 2-node srun
srun -N 2 -n 2 hostname

# Output should show:
# node1
# node2
```

If this fails, contact your cluster admins about inter-node communication setup.

## Step 6: Adjust Training Parameters (Optional)

CPU training is slower. Modify if needed:

```bash
# In leapgnn_cpu_simple.sbatch:

--batch-size 128        # Increase to 256 for faster training (if memory allows)
--sampling 5-5          # Increase to 10-10 for better accuracy (trades speed)
--epoch 5               # Increase to 10 for better convergence
--hidden-size 128       # Increase to 256 for capacity (trades speed)
--time=04:00:00         # Increase time limit if training is slower
```

Performance estimates (CPU 16 cores/node):
- **Current:** ~50 seconds/epoch → 5 epochs ≈ 25 minutes
- **Doubled params:** ~100 seconds/epoch → 10 epochs ≈ 17 minutes + overhead
- **Max batch:** ~150 seconds/epoch → 10 epochs ≈ 26 minutes + overhead

## Step 7: Test Run

Create a test script to verify setup:

```bash
#!/bin/bash
# test_connection.sbatch

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:05:00
#SBATCH --output=test_%j.log

echo "=== Node Connectivity Test ==="
srun -N 2 -n 2 hostname
srun -N 2 -n 2 pwd

echo "=== Python & PyTorch Test ==="
srun -N 2 -n 2 python -c "import torch; print(f'PyTorch: {torch.__version__}')"
srun -N 2 -n 2 python -c "import dgl; print(f'DGL: {dgl.__version__}')"

echo "=== Dataset Availability ==="
srun -N 2 -n 2 ls -lh ogbn_arxiv0/ | head -3
```

Submit and verify:
```bash
sbatch test_connection.sbatch
cat test_*.log
```

## Step 8: Final Sbatch File

After all edits, your final `leapgnn_cpu_simple.sbatch` should have:

1. ✅ Correct partition name
2. ✅ Correct conda environment path
3. ✅ Correct working directory (`cd` path)
4. ✅ Correct module loads for compiler
5. ✅ Updated time limit (if needed)
6. ✅ Verified dataset location

## Submission

```bash
# Navigate to your LeapGNN directory
cd /path/to/your/leapgnn/LeapGNN-AE

# Submit job
sbatch leapgnn_cpu_simple.sbatch

# Monitor
squeue -u $USER

# View logs
tail -f leapgnn_cpu_*.log
tail -f logs/test_graphsage_*.log
```

## Troubleshooting

### Job stays in PENDING
```bash
sacct -j YOUR_JOB_ID --format=State,Reason
# Common reasons:
# - Partition doesn't exist: sinfo -o "%P" (check partition names)
# - Not enough resources: sinfo -o "%P %a %D %c" (check CPU cores)
```

### Import errors (PyTorch/DGL)
```bash
# Verify environment
srun -N 1 -n 1 python -c "import torch; import dgl; print('OK')"

# If fails, install in conda environment
conda install pytorch torchvision torchaudio -c pytorch
pip install dgl
```

### Nodes can't communicate
```bash
# Test inter-node TCP
srun -N 2 -n 2 python -c "
import socket
rank = int(os.environ['SLURM_PROCID'])
print(f'Rank {rank}: {socket.gethostname()}')
"

# If fails, contact cluster admin - may need different interconnect setup
```

### Training takes too long
Reduce parameters:
- Decrease `--batch-size` (128 → 64)
- Decrease `--sampling` (5-5 → 3-3)
- Decrease `--epoch` (5 → 3)
- Decrease `--hidden-size` (128 → 64)

### GPU Available - Use It!

If your cluster has GPUs, use the GPU variant instead:

```bash
# Use leapgnn_arxiv_test.sbatch as template
# Add: #SBATCH --gpus-per-node=1
# Module: module load cuda-11.8  (or your version)
# Remove: --cpus-per-task adjustments
```

## Quick Checklist

Copy this template and fill in:
```bash
# Your cluster configuration:
CLUSTER_NAME: ___________
PARTITION: ___________
CONDA_PATH: ___________
DATASET_PATH: ___________
WORK_DIR: ___________
CORES_PER_NODE: ___________
ESTIMATED_EPOCH_TIME: _____ seconds

# After filling, create modified sbatch and test with test_connection.sbatch
```

## Support

If issues persist:
1. Check cluster documentation: `cat /etc/motd` or `man slurm-quickstart`
2. Contact cluster support with:
   - Cluster name and version: `sinfo --Version`
   - Your modified sbatch file
   - Error output: `cat leapgnn_cpu_*.err`
