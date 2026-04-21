from rpc_client import distcache_pb2_grpc
from rpc_client import distcache_pb2
import grpc
import random
import torch.backends.cudnn as cudnn
import argparse
import os
import sys
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import dgl
import numpy as np
import data

from dgl import DGLGraph
import logging
import time

from common.log import setup_primary_logging, setup_worker_logging
from dgl.frame import Frame, FrameRef

from model import gcn, graphsage, deep
from model import gat


def fetch_data(gpuid, feat, nodeflow, device):
    """Attach node features to nodeflow frames. Works on CPU and CUDA."""
    feat_dim = feat.shape[1]
    dims = {'features': feat_dim}

    nf_nids = nodeflow._node_mapping.tousertensor()
    offsets = nodeflow._layer_offsets

    for i in range(nodeflow.num_layers):
        tnid = nf_nids[offsets[i]:offsets[i + 1]]

        # allocate frame on CPU first
        frame = {name: torch.empty(tnid.size(0), dims[name]) for name in dims}
        tnid_list = tnid.tolist()

        # numpy gather
        features = feat[tnid_list]

        # bytes/ndarray -> torch float32
        for name in dims:
            frame[name].data = torch.frombuffer(features, dtype=torch.float32).reshape(len(tnid_list), feat_dim)

        # move to target device (CPU or CUDA)
        for name in dims:
            frame[name].data = frame[name].data.to(device)

        nodeflow._node_frames[i] = FrameRef(Frame(frame))


def main(local_procs_per_node):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    cudnn.benchmark = False

    args.distributed = args.world_size > 1

    if args.distributed:
        # world_size is number of nodes from CLI; multiply by local procs per node
        args.world_size = local_procs_per_node * args.world_size
        mp.spawn(run, nprocs=local_procs_per_node, args=(local_procs_per_node, args, log_queue))
    else:
        run(0, local_procs_per_node, args, log_queue)


def _pick_device(args, gpu):
    # CPU-only: when no CUDA available, always CPU.
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        torch.cuda.set_device(gpu)
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def run(gpu, local_procs_per_node, args, log_queue):
    # rank mapping: node-rank -> global rank
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * local_procs_per_node + gpu

    setup_worker_logging(args.rank, log_queue)

    device = _pick_device(args, gpu)
    use_cuda = (device.type == "cuda")

    sampling = args.sampling.split('-')
    assert len(set(sampling)) == 1, "Only support same #neighbors for each layer"

    if gpu == 0:
        logging.info(f"Args: {args}")
    logging.info(f"[rank {args.rank}] device={device}")

    # ---- Load graph data ----
    fg_adj = data.get_struct(args.dataset)
    fg_labels_np = data.get_labels(args.dataset)
    fg_train_mask, fg_val_mask, fg_test_mask = data.get_masks(args.dataset)

    fg_train_nid = np.nonzero(fg_train_mask)[0].astype(np.int64)
    test_nid = np.nonzero(fg_test_mask)[0].astype(np.int64)

    # labels tensor on CPU; move per batch
    fg_labels = torch.from_numpy(fg_labels_np).long()

    feat = np.load(os.path.join(args.dataset, "feat.npy"))
    featdim = feat.shape[1]

    # DGL graph for sampling
    fg = DGLGraph(fg_adj, readonly=True)

    # classes
    if "ogbn_arxiv" in args.dataset:
        args.n_classes = 40
    elif "ogbn_products" in args.dataset:
        args.n_classes = 47
    elif "citeseer" in args.dataset:
        args.n_classes = 6
    elif "pubmed" in args.dataset:
        args.n_classes = 3
    elif "reddit" in args.dataset:
        args.n_classes = 41
    elif "in" in args.dataset:
        args.n_classes = 60
    elif "uk" in args.dataset:
        args.n_classes = 60
    elif "test_dataset" in args.dataset:
        args.n_classes = 10
    else:
        raise Exception("Unsupported dataset")

    # ---- Model ----
    if args.model_name == "gcn":
        model = gcn.GCNSampling(featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    elif args.model_name == "graphsage":
        model = graphsage.GraphSageSampling(featdim, args.hidden_size, args.n_classes, len(sampling), F.relu, args.dropout)
    elif args.model_name == "gat":
        model = gat.GATSampling(
            featdim, args.hidden_size, args.n_classes, len(sampling), F.relu,
            [2 for _ in range(len(sampling) + 1)],
            args.dropout, args.dropout
        )
    elif args.model_name == "deepergcn":
        args.n_layers = len(sampling)
        args.in_feats = featdim
        model = deep.DeeperGCN(args)
    elif args.model_name == "film":
        model = deep.GNNFiLM(featdim, args.hidden_size, args.n_classes, len(sampling) + 1, args.dropout)
    else:
        raise Exception("Unsupported model_name")

    model = model.to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-5)

    if args.rank == 0:
        print("fg_train_nid:", fg_train_nid.shape[0], "world_size:", args.world_size, flush=True)
        print("Got feature dim:", featdim, flush=True)
        print("Total number of model params:", sum(p.numel() for p in model.parameters()), flush=True)

    # ---- Training ----
    for epoch in range(args.epoch):
        t_epoch = time.time()
        print(f"[rank {args.rank}] starting epoch {epoch}", flush=True)

        np.random.seed(epoch)
        np.random.shuffle(fg_train_nid)

        # Neighbor sampling (CPU-heavy)
        sampler = dgl.contrib.sampling.NeighborSampler(
            fg,
            args.batch_size,
            expand_factor=int(sampling[0]),
            num_hops=len(sampling) + 1,
            neighbor_type="in",
            shuffle=False,
            num_workers=args.num_worker,
            seed_nodes=fg_train_nid,
            prefetch=True,
            add_self_loop=True
        )

        print(f"[rank {args.rank}] sampler created for epoch {epoch}", flush=True)

        model.train()
        iters = 0
        t0 = time.time()

        for nf in sampler:
            # feature attach
            fetch_data(gpu, feat, nf, device)

            batch_nid = nf.layer_parent_nid(-1)

            # IMPORTANT: CrossEntropyLoss expects 1D target [N]
            labels = fg_labels[batch_nid].view(-1).long().to(device)

            pred = model(nf)
            loss = loss_fn(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iters % 20 == 0:
                print(f"[rank {args.rank}] epoch {epoch} iter {iters} loss {loss.item():.4f}", flush=True)

            iters += 1

        print(
            f"[rank {args.rank}] finished epoch {epoch} iters={iters} "
            f"epoch_time={time.time()-t_epoch:.1f}s total={time.time()-t0:.1f}s",
            flush=True
        )

        # Optional eval (disabled by default in sbatch below)
        if args.eval:
            model.eval()
            num_acc = 0
            with torch.no_grad():
                for nf in dgl.contrib.sampling.NeighborSampler(
                    fg,
                    len(test_nid),
                    expand_factor=int(sampling[0]),
                    neighbor_type="in",
                    num_workers=args.num_worker,
                    num_hops=len(sampling) + 1,
                    seed_nodes=test_nid,
                    prefetch=True,
                    add_self_loop=True
                ):
                    fetch_data(gpu, feat, nf, device)
                    pred = model(nf)
                    batch_nids = nf.layer_parent_nid(-1)
                    batch_labels = fg_labels[batch_nids].view(-1).long().to(device)
                    num_acc += (pred.argmax(dim=1) == batch_labels).sum().cpu().item()

            acc = num_acc / len(test_nid)
            print(f"[rank {args.rank}] epoch {epoch} test_acc={acc:.4f}", flush=True)


def parse_args_func(argv):
    p = argparse.ArgumentParser(description="GNN Training")
    p.add_argument("-d", "--dataset", default="/data/cwj/pagraph/gendemo", type=str)
    p.add_argument("-s", "--sampling", default="2-2-2", type=str)
    p.add_argument("-hd", "--hidden-size", default=256, type=int)
    p.add_argument("-ncls", "--n-classes", default=60, type=int)
    p.add_argument("-bs", "--batch-size", default=2, type=int)
    p.add_argument("-dr", "--dropout", default=0.2, type=float)
    p.add_argument("-lr", "--lr", default=3e-2, type=float)
    p.add_argument("-wdy", "--weight-decay", default=0.0, type=float)
    p.add_argument("-mn", "--model-name", default="graphsage", type=str,
                   choices=["deepergcn", "gat", "graphsage", "gcn", "film", "demo"])
    p.add_argument("-ep", "--epoch", default=3, type=int)
    p.add_argument("-wkr", "--num-worker", default=1, type=int)
    p.add_argument("-cs", "--cache-size", default=0, type=int)
    p.add_argument("--seed", default=None, type=int)
    p.add_argument("--log", dest="log", action="store_true")
    p.add_argument("--eval", action="store_true")

    # distributed
    p.add_argument("--dist-url", default="tcp://127.0.0.1:23456", type=str)
    p.add_argument("--world-size", default=1, type=int)  # number of nodes
    p.add_argument("--rank", default=0, type=int)        # node rank
    p.add_argument("--gpu", default=None, type=int)
    p.add_argument("--grpc-port", default="10.5.30.43:18110", type=str)

    # deepergcn extras
    p.add_argument("--mlp_layers", type=int, default=1)
    p.add_argument("--block", default="res+", type=str)
    p.add_argument("--conv", type=str, default="gen")
    p.add_argument("--gcn_aggr", type=str, default="max")
    p.add_argument("--norm", type=str, default="batch")
    p.add_argument("--t", type=float, default=1.0)
    p.add_argument("--p", type=float, default=1.0)
    p.add_argument("--y", type=float, default=0.0)
    p.add_argument("--learn_t", action="store_true")
    p.add_argument("--learn_p", action="store_true")
    p.add_argument("--learn_y", action="store_true")
    p.add_argument("--msg_norm", action="store_true")
    p.add_argument("--learn_msg_scale", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)

    args = parse_args_func(None)
    model_name = args.model_name
    datasetname = args.dataset.strip("/").split("/")[-1]

    # logs
    log_dir = os.path.dirname(os.path.abspath(__file__)) + "/logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir,
        f"test_{model_name}_{datasetname}_trainer{args.world_size}_bs{args.batch_size}_sl{args.sampling}"
        f"_ep{args.epoch}_hd{args.hidden_size}_cpu.log"
    )

    # IMPORTANT: only node-rank 0 deletes log; safe in multi-node
    if args.rank == 0 and os.path.exists(log_filename):
        try:
            os.remove(log_filename)
        except FileNotFoundError:
            pass

    # CPU-only: 1 process per node
    local_procs_per_node = 1

    log_queue, log_listener = setup_primary_logging(log_filename, "error.log")
    try:
        main(local_procs_per_node)
    finally:
        log_listener.stop()
