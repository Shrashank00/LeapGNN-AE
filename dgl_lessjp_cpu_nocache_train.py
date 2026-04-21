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


def fetch_feats_into_nf(feat_np, nf, device):
    nf_nids = nf._node_mapping.tousertensor()
    offsets = nf._layer_offsets
    for i in range(nf.num_layers):
        tnid = nf_nids[offsets[i]:offsets[i + 1]]
        x = torch.as_tensor(feat_np[tnid.tolist()], dtype=torch.float32, device=device)
        nf._node_frames[i] = FrameRef(Frame({"features": x}))


def send_recv_model_round_robin(model, rank, world_size):
    dst = (rank + 1) % world_size
    src = (rank - 1 + world_size) % world_size
    send_first = (rank % 2 == 0)

    for val in model.parameters():
        val_cpu = val.detach().to("cpu")
        new_val = torch.zeros_like(val_cpu)
        if send_first:
            dist.send(val_cpu, dst=dst)
            dist.recv(new_val, src=src)
        else:
            dist.recv(new_val, src=src)
            dist.send(val_cpu, dst=dst)
        with torch.no_grad():
            val[:] = new_val.to(val.device)


def make_offsets(n, batch_size):
    if n <= 0:
        return [0]
    offsets = list(range(0, n, batch_size))
    if offsets[-1] != n:
        offsets.append(n)
    return offsets


def get_batch_output_nodes(nf):
    """
    DGL 0.4.1 NodeFlow convention:
      layer 0 often corresponds to the seed/output/batch nodes.
    Using layer_parent_nid(-1) can be empty even when training is valid.
    """
    try:
        return nf.layer_parent_nid(0)
    except Exception:
        # fallback to old behavior
        return nf.layer_parent_nid(-1)


def run(args):
    print(f"[rank {args.rank}] host={os.uname().nodename} starting", flush=True)
    dist.init_process_group(backend="gloo", init_method=args.dist_url, world_size=args.world_size, rank=args.rank)
    print(f"[rank {args.rank}] init_process_group done", flush=True)

    device = torch.device("cpu")

    fg_adj = data.get_struct(args.dataset)
    fg_labels_np = data.get_labels(args.dataset)
    fg_train_mask, _, fg_test_mask = data.get_masks(args.dataset)

    train_nid_all = np.nonzero(fg_train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)

    feat_np = np.load(os.path.join(args.dataset, "feat.npy"))
    featdim = feat_np.shape[1]

    fg_labels = torch.from_numpy(fg_labels_np).long()
    fg = DGLGraph(fg_adj, readonly=True)

    if args.rank == 0:
        print(f"[rank 0] graph stats: nodes={fg.number_of_nodes()} edges={fg.number_of_edges()}", flush=True)

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
    ntrain_per_rank = int(len(train_nid_all) / world_size)
    start = args.rank * ntrain_per_rank
    end = (args.rank + 1) * ntrain_per_rank
    train_lnid = train_nid_all[start:end]

    # keep neighbor_type configurable; default "in"
    neighbor_type = args.neighbor_type
    if args.rank == 0:
        print(f"[rank 0] using neighbor_type={neighbor_type}", flush=True)
        print(f"[rank 0] using batch output nodes from layer 0", flush=True)

    for epoch in range(args.epoch):
        np.random.seed(epoch)
        np.random.shuffle(train_lnid)

        offsets = make_offsets(len(train_lnid), args.batch_size)
        if args.rank == 0:
            print(f"[rank 0] epoch {epoch} starting; train_lnid={len(train_lnid)} num_batches={len(offsets)-1}", flush=True)

        seed_nodes = train_lnid.tolist()
        sampler = dgl.contrib.sampling.NeighborSamplerWithDiffBatchSz(
            fg, offsets,
            expand_factor=int(sampling[0]),
            num_hops=len(sampling) + 1,
            neighbor_type=neighbor_type,
            shuffle=False,
            num_workers=args.num_worker,
            seed_nodes=seed_nodes,
            prefetch=True,
            add_self_loop=True
        )

        optimizer.zero_grad()
        epoch_loss_sum = 0.0
        epoch_steps = 0

        for step, nf in enumerate(sampler):
            if args.jump_model:
                send_recv_model_round_robin(model, args.rank, world_size)

            fetch_feats_into_nf(feat_np, nf, device)

            batch_nid = get_batch_output_nodes(nf)
            if len(batch_nid) == 0:
                if step < 5:
                    print(f"[rank {args.rank}] empty batch_nid at step={step} (layers={nf.num_layers})", flush=True)
                continue

            labels = fg_labels[batch_nid].view(-1).long().to(device)
            pred = model(nf)
            loss = loss_fn(pred, labels)
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            epoch_loss_sum += float(loss.item())
            epoch_steps += 1

            if step % args.log_every == 0:
                print(f"[rank {args.rank}] epoch {epoch} step {step} loss {loss.item():.4f} batch={len(batch_nid)}", flush=True)

        avg_loss = epoch_loss_sum / max(epoch_steps, 1)
        dist.barrier()
        if args.rank == 0:
            print(f"[rank 0] epoch {epoch} finished avg_loss={avg_loss:.4f} steps={epoch_steps}", flush=True)
        dist.barrier()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dist-url", required=True, type=str)
    p.add_argument("--world-size", required=True, type=int)
    p.add_argument("--rank", required=True, type=int)

    p.add_argument("-d", "--dataset", default="ogbn_arxiv0", type=str)
    p.add_argument("-mn", "--model-name", default="graphsage", type=str)

    p.add_argument("-ep", "--epoch", default=2, type=int)
    p.add_argument("-bs", "--batch-size", default=64, type=int)

    p.add_argument("-s", "--sampling", default="2-2", type=str)
    p.add_argument("-hd", "--hidden-size", default=64, type=int)
    p.add_argument("-dr", "--dropout", default=0.2, type=float)
    p.add_argument("-lr", "--lr", default=1e-3, type=float)
    p.add_argument("-wdy", "--weight-decay", default=0.0, type=float)
    p.add_argument("-wkr", "--num-worker", default=0, type=int)

    p.add_argument("--neighbor-type", default="in", choices=["in", "out"], type=str)

    p.add_argument("--log-every", default=10, type=int)
    p.add_argument("--jump-model", action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)
