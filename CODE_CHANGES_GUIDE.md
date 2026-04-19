# Code Changes Required for Multi-Node Training

This guide explains what needs to be changed in the original LeapGNN codebase for stable multi-node distributed training.

## Summary of Required Changes

The original code has issues with:
1. ❌ `--eval` flag causes hanging after training completes
2. ❌ Multi-node setup not properly synchronized
3. ❌ gRPC cache server addresses hardcoded
4. ❌ Evaluation step doesn't terminate properly on distributed setup

## File: dgl_single.py (Main Training Script)

### Change 1: Remove or Fix --eval Flag Hanging

**Problem:** When `--eval=True`, the training completes but evaluation hangs indefinitely on rank 1.

**Current Code (Line ~380):**
```python
parser.add_argument('--eval', default=False, action='store_true',
                    help='Whether to evaluate on test set.')
```

**Issue in Training Loop:**
The evaluation step doesn't synchronize between ranks properly, causing rank 1 to wait forever while rank 0 finishes.

**Solution 1: Simple - Disable Eval in Distributed Mode**

Find the evaluation section (typically after training loop):
```python
# After training loop
if args.eval:
    # evaluation code here
```

Replace with:
```python
# After training loop
if args.eval and args.rank == 0:  # Only evaluate on rank 0
    # evaluation code here
    print("Evaluation complete")
```

**Solution 2: Better - Add Barrier Before Eval**

Add synchronization point:
```python
import torch.distributed as dist

# After training loop
if args.world_size > 1:
    dist.barrier()  # Wait for all ranks

if args.eval:
    if args.rank == 0:
        # run evaluation
        print(f"Test Accuracy: {accuracy}")
    
    if args.world_size > 1:
        dist.barrier()  # Wait after evaluation
```

**Solution 3: Recommended - Make --eval Default False**

```python
parser.add_argument('--eval', default=False, action='store_true',
                    help='Whether to evaluate on test set (disabled for distributed).')
```

Then in sbatch script, just don't use `--eval` flag:
```bash
python dgl_single.py \
    --world-size 2 \
    --rank $SLURM_PROCID \
    # ... other args
    # REMOVE: --eval
```

---

### Change 2: Fix Distributed Training Initialization

**Problem:** Ranks might not all reach `init_process_group()` at the same time.

**Current Code (typical around line ~60):**
```python
def run(rank, world_size, args):
    dist.init_process_group(backend='gloo', ...)
```

**Add timeout and retry logic:**
```python
def run(rank, world_size, args):
    import os
    os.environ['NCCL_TIMEOUT'] = '3600'  # 1 hour timeout
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    # Retry logic for flaky connections
    max_retries = 5
    for attempt in range(max_retries):
        try:
            dist.init_process_group(
                backend='gloo',
                init_method=args.dist_url,
                world_size=world_size,
                rank=rank,
                timeout=timedelta(minutes=30)
            )
            if rank == 0:
                print(f"Rank {rank}: Successfully initialized process group")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                if rank == 0:
                    print(f"Rank {rank}: Init failed, retrying... ({attempt+1}/{max_retries})")
                time.sleep(5)
            else:
                print(f"Rank {rank}: Failed to initialize after {max_retries} attempts: {e}")
                raise
```

---

### Change 3: Add Proper Cleanup on Exit

**Problem:** Processes might hang if they exit before synchronizing.

**Find the main block and modify:**

```python
# Current:
if __name__ == "__main__":
    main(ngpus_per_node)

# Change to:
if __name__ == "__main__":
    try:
        main(ngpus_per_node)
    finally:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
            if args.rank == 0:
                print("Process group destroyed successfully")
```

---

## File: dgl_p3_avoid_oom.py (LeapGNN P3 Variant)

### Change: Fix Hardcoded gRPC Server Address

**Problem:** Line 355 has hardcoded server address `"10.5.30.43:18110"` which doesn't exist in your environment.

**Current Code (Line ~355):**
```python
parser.add_argument('--grpc-port', default="10.5.30.43:18110", type=str,
                    help='gRPC server port')
```

**Change to:**
```python
parser.add_argument('--grpc-port', 
                    default=None,  # Must be provided or use localhost fallback
                    type=str,
                    help='gRPC server port (format: host:port). If None, uses localhost:18110')
```

**Then in initialization (around line ~140):**
```python
# Current:
cache_client = DistCacheClientP3(args.grpc_port, args.gpu, args.log, args.world_size, args.rank)

# Change to:
if args.grpc_port is None:
    if args.log:
        # Use local caching, ignore gRPC
        args.grpc_port = "localhost:18110"
        print(f"Rank {args.rank}: Using local caching (--log enabled)")
    else:
        args.grpc_port = "localhost:18110"
        print(f"Rank {args.rank}: WARNING - gRPC servers may not be reachable at {args.grpc_port}")

cache_client = DistCacheClientP3(args.grpc_port, args.gpu, args.log, args.world_size, args.rank)
```

---

## File: storage/storage_dist.py

### Change: Add Fallback for Missing gRPC Servers

**Problem:** Code crashes if gRPC servers unavailable even with `--log` flag.

**Find the gRPC connection code (around line ~628):**
```python
def get_feat_dim(self):
    response = self.stub.DCSubmit(distcache_pb2.DCRequest(...))
    return response.feat_dim
```

**Add error handling:**
```python
def get_feat_dim(self):
    try:
        response = self.stub.DCSubmit(distcache_pb2.DCRequest(...))
        return response.feat_dim
    except Exception as e:
        if self.log:  # Local caching mode
            print(f"Rank {self.rank}: gRPC unavailable, using local cache fallback: {e}")
            # Return safe default or load from dataset directly
            return self.load_feat_dim_from_dataset()
        else:
            raise  # Re-raise if not in local cache mode
```

---

## Dataset Files

### File: data/dataset.py

**Check these functions work in distributed mode:**

```python
# Should already have rank/world_size awareness:
def __init__(self, name, rank, world_size):
    # Rank 0 loads data first
    # Other ranks wait with barrier
    # This prevents file races
```

**If missing, add:**
```python
import torch.distributed as dist

class Dataset:
    def __init__(self, name, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        
        if self.rank == 0:
            # Load/download dataset
            self._load_data()
            
            if self.world_size > 1:
                dist.barrier()
        else:
            if self.world_size > 1:
                dist.barrier()
            # Now safe to load from cache
            self._load_data()
```

---

## Summary of Minimal Changes for Production

If you only want to make **minimal** changes to existing code:

### 1. In dgl_single.py - Add One Line
```python
# Around line 380, after parser definition:
parser.add_argument('--eval', default=False, action='store_true')

# Around line 140, in training function, add BEFORE evaluation:
if args.world_size > 1:
    torch.distributed.barrier()
```

### 2. In sbatch script - Don't use --eval
```bash
# REMOVE this from all sbatch scripts:
--eval

# Your training command becomes:
srun -N 2 --ntasks=2 ... python dgl_single.py \
    --world-size 2 \
    --rank $SLURM_PROCID \
    ... \
    --log
    # NO --eval flag
```

### 3. For P3 (dgl_p3_avoid_oom.py) - One Line Change
Remove/comment the hardcoded gRPC address or pass it dynamically via sbatch.

---

## Verification Checklist

After making changes, verify:

```bash
# 1. Single-node training works
python dgl_single.py --dataset ogbn_arxiv0 --epoch 1

# 2. Multi-node training works
sbatch your_modified_script.sbatch
squeue
tail -f logs/test_graphsage*.log

# 3. No hanging after 5 epochs
# (Previously would hang ~13 minutes after completing epochs)
```

---

## Working Reference Code

You have working examples in your repo:
- **leapgnn_cpu_simple.sbatch** - Reference sbatch (fully working)
- **logs/test_graphsage_ogbn_arxiv0_trainer2_bs128_sl5-5_ep5_hd128_localFalse.log** - Reference complete run

These completed successfully without hanging. Use them as baselines for comparison.
