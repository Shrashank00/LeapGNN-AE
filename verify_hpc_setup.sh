#!/bin/bash
# HPC Transfer Verification Script
# Run this after transferring code and setting up the environment

echo "=========================================="
echo "LeapGNN HPC Setup Verification"
echo "=========================================="
echo ""

FAILED=0

# Check conda environment
echo "[1/10] Checking conda environment..."
if conda info --envs | grep -q repgnn; then
    echo "✓ Conda environment 'repgnn' found"
else
    echo "✗ Conda environment 'repgnn' NOT found"
    FAILED=1
fi
echo ""

# Check Python version
echo "[2/10] Checking Python version..."
conda activate repgnn 2>/dev/null
PY_VERSION=$(python --version 2>&1 | awk '{print $2}')
if [[ $PY_VERSION == 3.9.* ]]; then
    echo "✓ Python version: $PY_VERSION (OK)"
else
    echo "⚠ Python version: $PY_VERSION (expected 3.9.x)"
fi
echo ""

# Check PyTorch
echo "[3/10] Checking PyTorch..."
if python -c "import torch; print(f'PyTorch {torch.__version__}`)" 2>/dev/null; then
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "✓ PyTorch found"
    echo "  CUDA available: $CUDA_AVAILABLE"
else
    echo "✗ PyTorch NOT found"
    FAILED=1
fi
echo ""

# Check DGL
echo "[4/10] Checking DGL..."
if python -c "import dgl; print(f'DGL {dgl.__version__}')" 2>/dev/null; then
    echo "✓ DGL found"
else
    echo "✗ DGL NOT found"
    FAILED=1
fi
echo ""

# Check torch_scatter
echo "[5/10] Checking torch_scatter..."
if python -c "import torch_scatter" 2>/dev/null; then
    echo "✓ torch_scatter found"
else
    echo "✗ torch_scatter NOT found"
    FAILED=1
fi
echo ""

# Check torch_cluster
echo "[6/10] Checking torch_cluster..."
if python -c "import torch_cluster" 2>/dev/null; then
    echo "✓ torch_cluster found"
else
    echo "✗ torch_cluster NOT found"
    FAILED=1
fi
echo ""

# Check torch_geometric
echo "[7/10] Checking torch_geometric..."
if python -c "import torch_geometric; print(f'PyG {torch_geometric.__version__}')" 2>/dev/null; then
    echo "✓ torch_geometric found"
else
    echo "✗ torch_geometric NOT found"
    FAILED=1
fi
echo ""

# Check code files
echo "[8/10] Checking code structure..."
if [ -f "dgl_jpgnn_trans.py" ] && [ -d "data" ] && [ -d "model" ]; then
    echo "✓ Main code files present"
else
    echo "✗ Some code files missing"
    FAILED=1
fi
echo ""

# Check dataset
echo "[9/10] Checking dataset..."
if [ -d "dataset" ] && [ ! -z "$(ls -A dataset 2>/dev/null)" ]; then
    echo "✓ Dataset directory found"
    echo "  Dataset files:"
    du -sh dataset/* 2>/dev/null | head -5
else
    echo "⚠ Dataset directory empty or missing (you may need to download it)"
fi
echo ""

# Check distributed training setup
echo "[10/10] Checking distributed training setup..."
if python -c "import torch.distributed as dist" 2>/dev/null; then
    echo "✓ Distributed training available"
else
    echo "✗ Distributed training NOT available"
    FAILED=1
fi
echo ""

echo "=========================================="
if [ $FAILED -eq 0 ]; then
    echo "✓ All checks passed! Ready to run on HPC."
else
    echo "✗ Some checks failed. Rerun hpc_setup.sh to fix issues."
fi
echo "=========================================="

# Print next steps
echo ""
echo "Next steps:"
echo "1. Update data paths in your SBATCH scripts"
echo "2. Customize hpc_submit.sbatch for your HPC cluster"
echo "3. Run: sbatch hpc_submit.sbatch"
echo ""
echo "To debug further, run individual Python imports manually:"
echo "  python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
