import os, time, random, argparse, threading
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import dgl
from dgl import DGLGraph

import data
from model import gcn, graphsage
from storage.storage_dist import DistCacheClient

# optional GPUtil
try:
    import GPUtil
except Exception:
    GPUtil = None

sem = threading.Semaphore(1000)


def reverse_columns(arr, k):
    num_cols = arr.shape[1]
    return np.concatenate((arr[:, k:num_cols], arr[:, 0:k]), axis=1)


def substract_columns(arr):
    return arr[:, 1:] - arr[:, :-1]


def get_model_trace(jp_times, world_size, sub_batch_offsets):
    sub_batches = [[None for _ in range(len(sub_batch_offsets))] for _ in range(world_size)]
    dist.all_gather_object(sub_batches, sub_batch_offsets)

    sub_batches = np.array(sub_batches)
    sub_batches_size = substract_columns(sub_batches)

    assert sub_batches_size.shape[1] % world_size == 0, f"bad sub_batches_size shape: {sub_batches_size.shape}"
    sub_batches_num = sub_batches_size.shape[1] // world_size

    model_trace = np.array(
        [[(j - i + world_size) % world_size for i in range(world_size)] * sub_batches_num for j in range(world_size)]
    )
    # no pruning for now (keep jp_times=world_size)
    return model_trace


def get_sub_batchs(epoch, fg_train_nid, world_size, ntrain_per_rank,
                   cache_client, nid2pid, rank, split_fn, jp_times):
    np.random.seed(epoch)
    np.random.shuffle(fg_train_nid)

    useful = fg_train_nid[:world_size * ntrain_per_rank].reshape(world_size, ntrain_per_rank)
    useful = np.apply_along_axis(split_fn, 1, useful).T

    cache_partidx = cache_client.get_cache_partid()
    assert cache_partidx == rank, "rank must equal partid"

    sub_batch_offsets = [0]
    cur_offset = 0
    sub_batch_nid = []

    reversed_useful = reverse_columns(useful, rank)
    for row in reversed_useful:
        for batch in row:
            mask = (nid2pid[batch] == cache_partidx)
            sub = batch[mask]
            sub_batch_nid.extend(sub.tolist())
            cur_offset += len(sub)
            sub_batch_offsets.append(cur_offset)

    model_trace = get_model_trace(jp_times, world_size, sub_batch_offsets)
    return sub_batch_nid, sub_batch_offsets, model_trace


def send_recv_model_trace_trace(model_trace, model, rank, jp_cnt, machine2model, world_size):
    cur_model_id = machine2model[rank]
    dst_machine_id = model_trace[cur_model_id][jp_cnt]

    # avoid deadlock: choose send_first ranks
    chains = []
    checked = [False] * world_size
    send_ranks = []
    for r in range(world_size):
        dst = model_trace[machine2model[r]][jp_cnt]
        chains.append((r, dst))
    chains.sort()
    for s, d in chains:
        if not checked[s]:
            send_ranks.append(s)
            checked[s] = True
            checked[d] = True
        else:
            if not checked[d]:
                send_ranks.append(d)
                checked[d] = True

    if dst_machine_id != rank:
        send_first = (rank in send_ranks)
        for val in model.parameters():
            val_cpu = val.detach().to("cpu")
            new_val = torch.zeros_like(val_cpu)
            if send_first:
                dist.send(val_cpu, dst=dst_machine_id)
                dist.recv(new_val)
            else:
                dist.recv(new_val)
                dist.send(val_cpu, dst=dst_machine_id)
            with torch.no_grad():
                val[:] = new_val.to(val.device)

    for rowid in range(world_size):
        machineid = model_trace[rowid][jp_cnt]
        machine2model[machineid] = rowid


def run(args):
    print(f"[rank {args.rank}] host={os.uname().nodename} starting", flush=True)
    print(f"[rank {args.rank}] dist_url={args.dist_url} world_size={args.world_size}", flush=True)

    device = torch.device("cpu")
    dist.init_process_group(backend="gloo", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print(f"[rank {args.rank}] init_process_group done", flush=True)

    sampling = args.sampling.split("-")
    assert len(set(sampling)) == 1

    fg_adj = data.get_struct(args.dataset)
    fg_labels_np = data.get_labels(args.dataset)
    fg_train_mask, _, fg_test_mask = data.get_masks(args.dataset)

    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    fg_labels = torch.from_numpy(fg_labels_np).long()
    fg = DGLGraph(fg_adj, readonly=True)

    print(f"[rank {args.rank}] graph loaded -> barrier", flush=True)
    dist.barrier()
    print(f"[rank {args.rank}] barrier done", flush=True)

    # cache client (must be reachable)
    cache_client = DistCacheClient(args.grpc_port, 0, args.log)
    cache_client.Reset()
    cache_client.ConstructNid2Pid(args.dataset, args.world_size, "metis", len(fg_train_mask))
    featdim = cache_client.feat_dim
    if args.rank == 0:
        print(f"[rank 0] featdim={featdim}", flush=True)

    if "ogbn_arxiv" in args.dataset:
        n_classes = 40
    else:
        raise Exception("Set dataset to ogbn_arxiv0 (or name containing ogbn_arxiv)")

    if args.model_name == "graphsage":
        model = graphsage.GraphSageSampling(featdim, args.hidden_size, n_classes, len(sampling), F.relu, args.dropout)
    elif args.model_name == "gcn":
        model = gcn.GCNSampling(featdim, args.hidden_size, n_classes, len(sampling), F.relu, args.dropout)
    else:
        raise Exception("Use --model-name graphsage or gcn")

    model = model.to(device)
    model = torch.nn.parallel.DistributedDataParallel(model)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)

    world_size = args.world_size
    ntrain_per_rank = int(fg_train_nid.shape[0] / world_size)

    max_train_nid = np.max(fg_train_nid) + 1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64) - 1
    for pid in range(world_size):
        part = data.get_partition_results(os.path.join(args.dataset, "dist_True"), "metis", world_size, pid)
        necessary = part[part < max_train_nid]
        nid2pid[necessary] = pid

    def split_fn(a):
        if len(a) <= args.batch_size:
            return np.split(a, np.array([len(a)]))
        return np.split(a, np.arange(args.batch_size, len(a), args.batch_size))

    fetch_func = cache_client.fetch_multiple_nfs_elimredun if args.deduplicate else cache_client.fetch_multiple_nfs_v2

    jp_times = world_size
    machine2model = [i for i in range(world_size)]

    for epoch in range(args.epoch):
        print(f"[rank {args.rank}] epoch {epoch} starting jp_times={jp_times}", flush=True)

        sub_batch_nid, sub_batch_offsets, model_trace = get_sub_batchs(
            epoch, fg_train_nid, world_size, ntrain_per_rank,
            cache_client, nid2pid, args.rank, split_fn, jp_times
        )

        with sem:
            sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(
                fg, sub_batch_offsets,
                expand_factor=int(sampling[0]),
                num_hops=len(sampling) + 1,
                neighbor_type="in",
                shuffle=False,
                num_workers=args.num_worker,
                seed_nodes=sub_batch_nid,
                prefetch=True,
                add_self_loop=True
            )

        sampler_iterator = iter(sampler)
        optimizer.zero_grad()
        jp_cnt = 0
        sub_iter = 0

        for sub_nf_id in range(len(sub_batch_offsets) - 1):
            send_recv_model_trace_trace(model_trace, model, args.rank, jp_cnt, machine2model, world_size)
            jp_cnt += 1

            if sub_nf_id % jp_times == 0:
                sub_nfs_lst = []
                for _ in range(jp_times):
                    try:
                        sub_nfs_lst.append(next(sampler_iterator))
                    except StopIteration:
                        break
                fetch_func(sub_nfs_lst)

            sub_nf = sub_nfs_lst[sub_nf_id % jp_times]

            if sub_nf._node_mapping.tousertensor().shape[0] > 0:
                batch_nid = sub_nf.layer_parent_nid(-1)
                labels = fg_labels[batch_nid].view(-1).long().to(device)

                pred = model(sub_nf)
                loss = loss_fn(pred, labels)
                loss.backward()

            dist.barrier()

            if (sub_iter + 1) % jp_times == 0:
                optimizer.step()
                optimizer.zero_grad()

            if sub_iter % 50 == 0:
                print(f"[rank {args.rank}] epoch {epoch} sub_iter {sub_iter}", flush=True)
            sub_iter += 1

        print(f"=> cur_epoch {epoch} finished on rank {args.rank}", flush=True)


def parse_args_func(argv):
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--dataset", default="ogbn_arxiv0", type=str)
    p.add_argument("-s", "--sampling", default="1-1", type=str)
    p.add_argument("-hd", "--hidden-size", default=32, type=int)
    p.add_argument("-bs", "--batch-size", default=16, type=int)
    p.add_argument("-dr", "--dropout", default=0.2, type=float)
    p.add_argument("-lr", "--lr", default=1e-3, type=float)
    p.add_argument("-wdy", "--weight-decay", default=0.0, type=float)
    p.add_argument("-mn", "--model-name", default="graphsage", type=str)
    p.add_argument("-ep", "--epoch", default=1, type=int)
    p.add_argument("-wkr", "--num-worker", default=0, type=int)
    p.add_argument("--log", dest="log", action="store_true")
    p.add_argument("--dist-url", required=True, type=str)
    p.add_argument("--world-size", required=True, type=int)
    p.add_argument("--rank", required=True, type=int)
    p.add_argument("--grpc-port", default="10.5.30.43:18110", type=str)
    p.add_argument("--nodedup", dest="deduplicate", action="store_false", default=True)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args_func(None)
    run(args)
