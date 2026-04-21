import os
import argparse
import threading
import numpy as np

import torch
import torch.distributed as dist
import torch.nn.functional as F

import dgl
from dgl import DGLGraph
from dgl.frame import Frame, FrameRef

import data
from model import gcn, graphsage

sem = threading.Semaphore(1000)


def reverse_columns(arr, k):
    num_cols = arr.shape[1]
    return np.concatenate((arr[:, k:num_cols], arr[:, 0:k]), axis=1)


def substract_columns(arr):
    return arr[:, 1:] - arr[:, :-1]


def fetch_feats_into_nf(feat_np, nf, device):
    nf_nids = nf._node_mapping.tousertensor()
    offsets = nf._layer_offsets
    for i in range(nf.num_layers):
        tnid = nf_nids[offsets[i]:offsets[i + 1]]
        tnid_list = tnid.tolist()
        x = torch.as_tensor(feat_np[tnid_list], dtype=torch.float32, device=device)
        nf._node_frames[i] = FrameRef(Frame({"features": x}))


def get_model_trace(jp_times, world_size, sub_batch_offsets):
    sub_batches = [[None for _ in range(len(sub_batch_offsets))] for _ in range(world_size)]
    dist.all_gather_object(sub_batches, sub_batch_offsets)

    sub_batches = np.array(sub_batches)
    sub_batches_size = substract_columns(sub_batches)
    assert sub_batches_size.shape[1] % world_size == 0
    sub_batches_num = sub_batches_size.shape[1] // world_size

    model_trace = np.array(
        [[(j - i + world_size) % world_size for i in range(world_size)] * sub_batches_num for j in range(world_size)]
    )
    return model_trace


def get_sub_batches(epoch, fg_train_nid, world_size, ntrain_per_rank, nid2pid, rank, batch_size, jp_times):
    np.random.seed(epoch)
    np.random.shuffle(fg_train_nid)

    useful = fg_train_nid[:world_size * ntrain_per_rank].reshape(world_size, ntrain_per_rank)

    def split_fn(a):
        if len(a) <= batch_size:
            return np.split(a, np.array([len(a)]))
        return np.split(a, np.arange(batch_size, len(a), batch_size))

    useful = np.apply_along_axis(split_fn, 1, useful).astype(object).T

    sub_batch_nid = []
    sub_batch_offsets = [0]
    cur = 0

    reversed_useful = reverse_columns(useful, rank)
    for row in reversed_useful:
        for batch in row:
            mask = (nid2pid[batch] == rank)
            sub = batch[mask]
            sub_batch_nid.extend(sub.tolist())
            cur += len(sub)
            sub_batch_offsets.append(cur)

    model_trace = get_model_trace(jp_times, world_size, sub_batch_offsets)
    return sub_batch_nid, sub_batch_offsets, model_trace


def send_recv_model_trace_trace(model_trace, model, rank, jp_cnt, machine2model, world_size):
    cur_model_id = machine2model[rank]
    dst_machine_id = model_trace[cur_model_id][jp_cnt]

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


def build_nid2pid(args, fg_train_nid, world_size):
    max_train_nid = int(np.max(fg_train_nid)) + 1
    nid2pid = np.zeros(max_train_nid, dtype=np.int64) - 1

    part_dir = os.path.join(args.dataset, "dist_True", f"{world_size}_metis")
    ok = os.path.isdir(part_dir) and all(os.path.exists(os.path.join(part_dir, f"{r}.npy")) for r in range(world_size))

    if ok:
        for pid in range(world_size):
            part = data.get_partition_results(os.path.join(args.dataset, "dist_True"), "metis", world_size, pid)
            necessary = part[part < max_train_nid]
            nid2pid[necessary] = pid

        missing = (nid2pid < 0)
        if np.any(missing):
            idx = np.nonzero(missing)[0]
            nid2pid[idx] = idx % world_size

        if args.rank == 0:
            print(f"[rank 0] Using METIS partitions from {part_dir}", flush=True)
        return nid2pid

    if args.rank == 0:
        print(f"[rank 0] METIS partitions not found at {part_dir}. Falling back to nid % world_size.", flush=True)
    idx = np.arange(max_train_nid, dtype=np.int64)
    return idx % world_size


def run(args):
    print(f"[rank {args.rank}] host={os.uname().nodename} starting", flush=True)
    dist.init_process_group(backend="gloo", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print(f"[rank {args.rank}] init_process_group done", flush=True)

    device = torch.device("cpu")

    fg_adj = data.get_struct(args.dataset)
    fg_labels_np = data.get_labels(args.dataset)
    fg_train_mask, _, fg_test_mask = data.get_masks(args.dataset)
    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)

    feat_np = np.load(os.path.join(args.dataset, "feat.npy"))
    featdim = feat_np.shape[1]

    fg_labels = torch.from_numpy(fg_labels_np).long()
    fg = DGLGraph(fg_adj, readonly=True)

    if "ogbn_arxiv" in args.dataset:
        n_classes = 40
    else:
        raise Exception("Unsupported dataset name for n_classes mapping")

    sampling = args.sampling.split("-")
    assert len(set(sampling)) == 1

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
    nid2pid = build_nid2pid(args, fg_train_nid, world_size)

    jp_times = world_size
    machine2model = [i for i in range(world_size)]

    def evaluate_test_accuracy():
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            with sem:
                sampler = dgl.contrib.sampling.NeighborSampler(
                    fg, len(test_nid),
                    expand_factor=int(sampling[0]),
                    num_hops=len(sampling) + 1,
                    neighbor_type="in",
                    shuffle=False,
                    num_workers=args.num_worker,
                    seed_nodes=test_nid,
                    prefetch=True,
                    add_self_loop=True
                )
            for nf in sampler:
                fetch_feats_into_nf(feat_np, nf, device)
                batch_nid = nf.layer_parent_nid(-1)
                if len(batch_nid) == 0:
                    continue
                labels = fg_labels[batch_nid].view(-1).long().to(device)
                pred = model(nf)
                correct += (pred.argmax(dim=1) == labels).sum().item()
                total += labels.numel()
        model.train()
        return correct / max(total, 1)

    for epoch in range(args.epoch):
        print(f"[rank {args.rank}] epoch {epoch} starting jp_times={jp_times}", flush=True)

        sub_batch_nid, sub_batch_offsets, model_trace = get_sub_batches(
            epoch, fg_train_nid, world_size, ntrain_per_rank, nid2pid, args.rank, args.batch_size, jp_times
        )

        # If this rank got no nodes this epoch, say it explicitly
        if args.rank == 0:
            print(f"[rank 0] epoch {epoch}: local sub_batch_nid={len(sub_batch_nid)}", flush=True)

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

        epoch_loss_sum = 0.0
        epoch_loss_steps = 0

        sub_nfs_lst = []

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
                if len(sub_nfs_lst) == 0:
                    break
                for nf in sub_nfs_lst:
                    fetch_feats_into_nf(feat_np, nf, device)

            sub_nf = sub_nfs_lst[sub_nf_id % len(sub_nfs_lst)]

            # FIX: decide “empty” by batch seed nodes, not _node_mapping
            batch_nid = sub_nf.layer_parent_nid(-1)
            if len(batch_nid) == 0:
                continue

            labels = fg_labels[batch_nid].view(-1).long().to(device)

            pred = model(sub_nf)
            loss = loss_fn(pred, labels)
            loss.backward()

            epoch_loss_sum += float(loss.item())
            epoch_loss_steps += 1

            if sub_iter % args.log_every == 0:
                print(f"[rank {args.rank}] epoch {epoch} sub_iter {sub_iter} loss {loss.item():.4f}", flush=True)

            dist.barrier()

            if (sub_iter + 1) % jp_times == 0:
                optimizer.step()
                optimizer.zero_grad()

            sub_iter += 1

        avg_loss = epoch_loss_sum / max(epoch_loss_steps, 1)

        dist.barrier()
        if args.rank == 0:
            print(f"[rank 0] epoch {epoch} finished avg_loss={avg_loss:.4f} steps={epoch_loss_steps}", flush=True)
        dist.barrier()

        print(f"=> cur_epoch {epoch} finished on rank {args.rank}", flush=True)

        dist.barrier()
        if args.eval:
            acc = evaluate_test_accuracy()
            if args.rank == 0:
                print(f"[rank 0] epoch {epoch} test_acc={acc:.4f}", flush=True)
        dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dist-url", required=True, type=str)
    p.add_argument("--world-size", required=True, type=int)
    p.add_argument("--rank", required=True, type=int)

    p.add_argument("-d", "--dataset", default="ogbn_arxiv0", type=str)
    p.add_argument("-mn", "--model-name", default="graphsage", type=str)
    p.add_argument("-ep", "--epoch", default=1, type=int)
    p.add_argument("-bs", "--batch-size", default=16, type=int)
    p.add_argument("-s", "--sampling", default="1-1", type=str)
    p.add_argument("-hd", "--hidden-size", default=32, type=int)
    p.add_argument("-dr", "--dropout", default=0.2, type=float)
    p.add_argument("-lr", "--lr", default=1e-3, type=float)
    p.add_argument("-wdy", "--weight-decay", default=0.0, type=float)
    p.add_argument("-wkr", "--num-worker", default=0, type=int)

    p.add_argument("--log-every", default=50, type=int)
    p.add_argument("--eval", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
