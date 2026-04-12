# HPC Transfer Checklist for LeapGNN

Use this checklist to ensure a smooth transfer of LeapGNN to your HPC cluster.

## Before Transfer

- [ ] Identify target HPC cluster details:
  - [ ] Cluster name: _________________
  - [ ] Username: _________________
  - [ ] Login node: _________________
  - [ ] Workspace path: _________________
  - [ ] Available CUDA version(s): _________________
  - [ ] Available partition names: _________________
  - [ ] Max nodes available: _________________
  - [ ] GPUs per node: _________________
  - [ ] Time limit per job: _________________

- [ ] Contact HPC admin for:
  - [ ] Maximum file size allowed
  - [ ] Shared storage vs. home directory
  - [ ] Network bandwidth (for large dataset transfer)
  - [ ] Job submission guidelines
  - [ ] Environment modules available

- [ ] Prepare local machine:
  - [ ] Ensure all code is committed to git
  - [ ] Test all scripts locally (if possible)
  - [ ] Generate SSH key for passwordless login: `ssh-keygen`
  - [ ] Add SSH key to HPC cluster

## Code Transfer

Choose ONE of these methods:

### Git Transfer (Recommended)
- [ ] Ensure code is in Git repo
- [ ] SSH into HPC cluster
- [ ] Clone repo: `git clone https://github.com/ISCS-ZJU/LeapGNN-AE.git`
- [ ] Checkout correct branch: `git checkout distributed_version`
- [ ] Update submodules: `git submodule init && git submodule update`

### Rsync Transfer
- [ ] From local machine, run rsync command (see HPC_TRANSFER_GUIDE.md)
- [ ] Verify transfer completed: `rsync ... --dry-run` to verify
- [ ] Actual transfer: `rsync ... /path/to/LeapGNN-AE/`

### SCP Transfer (if Git not available)
- [ ] Compress code: `tar -czf LeapGNN-AE.tar.gz LeapGNN-AE/`
- [ ] SCP to HPC: `scp LeapGNN-AE.tar.gz user@hpc-cluster:/path/`
- [ ] Extract on HPC: `tar -xzf LeapGNN-AE.tar.gz`

## Wheel Files Transfer

- [ ] Transfer PyTorch wheels to HPC:
  - [ ] torch_scatter-2.0.9-cp39-cp39-linux_x86_64.whl
  - [ ] torch_cluster-1.6.0-cp39-cp39-linux_x86_64.whl
  - [ ] torch_sparse-0.6.13-cp39-cp39-linux_x86_64.whl

## Dataset Transfer

Choose ONE of these methods:

### Direct Transfer (Small datasets)
- [ ] Transfer dataset using rsync/scp
- [ ] Verify file integrity: `diff -r local_dataset/ remote_dataset/`

### Download from Source (Recommended)
- [ ] SSH into HPC cluster
- [ ] Download from Zenodo link provided in README
- [ ] Extract dataset to correct location

### Use HPC Shared Storage
- [ ] Confirm with HPC admin if shared storage is available
- [ ] Copy dataset to shared path: `/shared/storage/leapgnn_data/`
- [ ] Update paths in scripts to use shared storage

## Environment Setup on HPC

- [ ] SSH into HPC cluster
- [ ] Navigate to LeapGNN directory: `cd LeapGNN-AE`
- [ ] Run setup script: `chmod +x hpc_setup.sh && ./hpc_setup.sh`
- [ ] Wait for setup to complete (20-30 minutes)
- [ ] Run verification: `chmod +x verify_hpc_setup.sh && ./verify_hpc_setup.sh`

### Manual Setup (if script fails)

- [ ] Create conda environment: `conda create -n repgnn python=3.9 -y`
- [ ] Activate environment: `conda activate repgnn`
- [ ] Install basic packages: `pip install -r hpc_requirements.txt`
- [ ] Load modules: `module load cuda/[version]`
- [ ] Build DGL from source: See hpc_setup.sh for commands
- [ ] Install wheels: `pip install torch_scatter*.whl torch_cluster*.whl torch_sparse*.whl`

## Configuration for HPC

- [ ] Review SBATCH script template: `hpc_submit.sbatch`
- [ ] Update HPC-specific parameters:
  - [ ] Job name: `--job-name=leapgnn_test`
  - [ ] Partition: `--partition=[your_queue]`
  - [ ] Number of nodes: `--nodes=4`
  - [ ] GPUs per node: `--gpus-per-node=1`
  - [ ] Time limit: `--time=04:00:00`
  - [ ] Memory: `--mem=128G`
  - [ ] Account/QoS if required

- [ ] Update script paths:
  - [ ] Data path: Update `DATA_PATH` variable
  - [ ] Output directory: Update log paths if needed
  - [ ] Model save path: Update where models are saved

- [ ] Test job submission:
  - [ ] Submit with dry-run first (if available)
  - [ ] Check job queued: `squeue -u $USER`

## Testing & Verification

In HPC environment:

- [ ] Run verification script: `./verify_hpc_setup.sh`
- [ ] Check all outputs are ✓
- [ ] Test simple Python import:
  ```bash
  python -c "import torch; print(torch.cuda.is_available())"
  ```
- [ ] Test with single GPU:
  ```bash
  srun --gpus=1 python -c "import torch; print(torch.cuda.get_device_name(0))"
  ```
- [ ] Test with multiple nodes (mini-test):
  - [ ] Create small test job
  - [ ] Verify distributed training setup works
  - [ ] Check all nodes communicate correctly

## Running Jobs

- [ ] Submit test job: `sbatch hpc_submit.sbatch`
- [ ] Monitor job: `squeue -u $USER`
- [ ] Check output: `tail -f logs/slurm_[jobid].out`
- [ ] If errors occur:
  - [ ] Check error log: `cat logs/slurm_[jobid].err`
  - [ ] Consult HPC_TRANSFER_GUIDE.md troubleshooting section

## Performance Tuning

- [ ] Monitor GPU usage: `nvidia-smi` on compute node
- [ ] Check network bandwidth for distributed training
- [ ] Adjust batch size if OOM occurs
- [ ] Profile code to identify bottlenecks
- [ ] Optimize for multi-GPU/multi-node setup

## Documentation

- [ ] Document any HPC-specific changes
- [ ] Note which commands work on your cluster
- [ ] Keep record of optimal SBATCH parameters
- [ ] Document any issues encountered and solutions

## Final Notes

- Project location: _________________________
- Conda environment: `repgnn`
- Main script: `dgl_jpgnn_trans.py`
- Log files: `logs/` directory
- Dataset location: _________________________

---

### Troubleshooting Quick Links

- CUDA/GPU issues → See "Troubleshooting" in HPC_TRANSFER_GUIDE.md
- Build failures → Check `hpc_setup.sh` step-by-step
- Memory issues → Increase `--mem` in SBATCH
- Network issues → Contact HPC admin
- Dataset issues → Download from Zenodo or check paths
