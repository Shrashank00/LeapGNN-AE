#!/usr/bin/env python3
"""
Create a small synthetic test dataset for quick testing
"""
import numpy as np
import scipy.sparse as sp
import os

def create_test_dataset(output_dir, num_nodes=1000, num_edges=5000, num_features=128, num_classes=10):
    """Create a small synthetic graph dataset"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating test dataset in {output_dir}...")
    print(f"  Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}, Classes: {num_classes}")
    
    # Generate random adjacency matrix (sparse)
    rows = np.random.randint(0, num_nodes, num_edges)
    cols = np.random.randint(0, num_nodes, num_edges)
    data = np.ones(num_edges)
    
    adj = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes))
    print(f"  Created adjacency matrix with {adj.nnz} edges")
    
    # Save adjacency matrix
    sp.save_npz(os.path.join(output_dir, 'adj.npz'), adj)
    
    # Generate features
    features = np.random.randn(num_nodes, num_features).astype(np.float32)
    np.save(os.path.join(output_dir, 'feat.npy'), features)
    print(f"  Created features: {features.shape}")
    
    # Generate labels
    labels = np.random.randint(0, num_classes, num_nodes)
    np.save(os.path.join(output_dir, 'labels.npy'), labels)
    print(f"  Created labels: {labels.shape}")
    
    # Create train/val/test splits
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    train_mask = np.zeros(num_nodes, dtype=np.bool_)
    val_mask = np.zeros(num_nodes, dtype=np.bool_)
    test_mask = np.zeros(num_nodes, dtype=np.bool_)
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    
    np.save(os.path.join(output_dir, 'train.npy'), train_mask)
    np.save(os.path.join(output_dir, 'val.npy'), val_mask)
    np.save(os.path.join(output_dir, 'test.npy'), test_mask)
    
    print(f"  Train nodes: {train_mask.sum()}, Val nodes: {val_mask.sum()}, Test nodes: {test_mask.sum()}")
    print("Dataset created successfully!")

if __name__ == '__main__':
    # Create a small dataset (1000 nodes)
    create_test_dataset('./test_dataset_small', num_nodes=1000, num_edges=5000, 
                        num_features=128, num_classes=10)
