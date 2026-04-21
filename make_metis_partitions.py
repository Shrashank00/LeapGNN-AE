import os
import argparse
import numpy as np

import data
import scipy.sparse as sp

import pymetis


def csr_from_adj(adj):
    """
    Accepts a scipy sparse matrix (csr/csc/coo) and returns CSR.
    """
    if sp.isspmatrix_csr(adj):
        return adj
    return adj.tocsr()


def build_adjncy_xadj(csr: sp.csr_matrix):
    """
    Convert CSR graph to METIS adjacency format (xadj, adjncy).
    METIS expects an undirected graph; if the input is directed, we symmetrize.
    """
    # symmetrize (safe for already-undirected)
    csr = (csr + csr.T)
    csr.data[:] = 1
    csr.eliminate_zeros()

    indptr = csr.indptr.astype(np.int64)
    indices = csr.indices.astype(np.int64)

    # PyMetis wants python lists
    xadj = indptr.tolist()
    adjncy = indices.tolist()
    return xadj, adjncy, csr.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str, help="dataset dir, e.g. ogbn_arxiv0")
    ap.add_argument("--world-size", required=True, type=int, help="number of partitions, e.g. 2")
    args = ap.parse_args()

    dataset = args.dataset
    k = args.world_size

    print(f"Loading adjacency for dataset={dataset} ...", flush=True)
    adj = data.get_struct(dataset)

    # data.get_struct may return numpy/scipy; enforce CSR
    if not sp.issparse(adj):
        raise RuntimeError(f"Expected scipy sparse adjacency from data.get_struct, got {type(adj)}")

    csr = csr_from_adj(adj)
    xadj, adjncy, n = build_adjncy_xadj(csr)

    print(f"Running PyMetis partitioning: n={n}, nparts={k} ...", flush=True)
    # pymetis expects adjacency as list-of-lists OR xadj/adjncy style
    # Use xadj/adjncy for speed.
    _, parts = pymetis.part_graph(nparts=k, xadj=xadj, adjncy=adjncy)

    parts = np.asarray(parts, dtype=np.int64)
    assert parts.shape[0] == n

    out_dir = os.path.join(dataset, "dist_True", f"{k}_metis")
    os.makedirs(out_dir, exist_ok=True)

    # Save for each rank: list of node ids assigned to that partition
    for rank in range(k):
        nids = np.nonzero(parts == rank)[0].astype(np.int64)
        out_path = os.path.join(out_dir, f"{rank}.npy")
        np.save(out_path, nids)
        print(f"Wrote {out_path} (#nodes={nids.shape[0]})", flush=True)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
