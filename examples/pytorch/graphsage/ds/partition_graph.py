import dgl
import numpy as np
import torch as th
import argparse
import time
import os
import time

from load_graph import load_reddit, load_ogb
from ogb.lsc import MAG240MDataset
from friendster import FriendSterDataset
from dgl.data.utils import generate_mask_tensor
from dgl.convert import to_homogeneous
import psutil
import scipy.sparse as sp
from scipy.sparse import coo_matrix
from dgl.convert import from_scipy

def load_mag240m(path='/data/dgl/mag240m'):
    '''
    We do not process the node feature here.
    '''

    print('loading graph')
    start = time.time()
    (g,), _ = dgl.load_graphs(os.path.join(path, 'graph.dgl'))
    end = time.time()
    print('Loading graph time', end - start)
    g = g.formats('coo')

    start = time.time()
    dataset = MAG240MDataset(root='/data/ogb/')
    end = time.time()
    print('Loading dataset time', end - start)

    print('Loading features')
    paper_offset = dataset.num_authors + dataset.num_institutions
    num_nodes = paper_offset + dataset.num_papers
    num_features = dataset.num_paper_features
    # feats = np.memmap(args.full_feature_path, mode='r', dtype='float16', shape=(num_nodes, num_features))

    train_idx = th.LongTensor(dataset.get_idx_split('train')) + paper_offset
    val_idx = th.LongTensor(dataset.get_idx_split('valid')) + paper_offset
    train_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
    train_mask[train_idx] = True
    val_mask = th.zeros((g.number_of_nodes(),), dtype=th.bool)
    val_mask[val_idx] = True
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    return g

def loadUK():
    dataset = WebUKDataset()
    graph = dataset[0]
    node_types = graph.ndata['node_type'].numpy()
    train_mask = (node_types == 0)
    val_mask = (node_types == 1)
    test_mask = (node_types == 2)
    graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    return graph

def loadFS():
    dataset = FriendSterDataset()
    graph = dataset[0]
    node_types = graph.ndata['node_type'].numpy()
    train_mask = (node_types == 0)
    val_mask = (node_types == 1)
    test_mask = (node_types == 2)
    graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    return graph

def loadTW():
    dataset = TwitterDataset()
    graph = dataset[0]
    node_types = graph.ndata['node_type'].numpy()
    train_mask = (node_types == 0)
    val_mask = (node_types == 1)
    test_mask = (node_types == 2)
    graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    return graph

def buildFakeGraph():
    n_nodes = 9
    row = np.array([0, 0, 1, 2, 2, 3, 3, 3, 4, 6, 7])
    col = np.array([5, 4, 3, 2, 8, 4, 5, 6, 5, 8, 8])
    node_types = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(n_nodes, n_nodes))
    graph = from_scipy(coo)
    graph = dgl.to_bidirected(graph)
    graph = dgl.to_simple(graph)
    graph.ndata['node_type'] = th.tensor(node_types, dtype=th.int32)
    node_types = graph.ndata['node_type'].numpy()
    train_mask = (node_types == 0)
    val_mask = (node_types == 1)
    test_mask = (node_types == 2)
    graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    return graph



if __name__ == '__main__':
    argparser = argparse.ArgumentParser("Partition builtin graphs")
    argparser.add_argument('--dataset', type=str, default='reddit',
                           help='datasets: reddit, ogb-product, ogb-paper100M, mag240m')
    argparser.add_argument('--num_parts', type=int, default=4,
                           help='number of partitions')
    argparser.add_argument('--part_method', type=str, default='metis',
                           help='the partition method')
    argparser.add_argument('--balance_train', action='store_true',
                           help='balance the training size in each partition.')
    argparser.add_argument('--undirected', action='store_true',
                           help='turn the graph into an undirected graph.')
    argparser.add_argument('--balance_edges', action='store_true',
                           help='balance the number of edges in each partition.')
    argparser.add_argument('--num_trainers_per_machine', type=int, default=1,
                           help='the number of trainers per machine. The trainer ids are stored\
                                in the node feature \'trainer_id\'')
    argparser.add_argument('--root', type=str, default='/data/ogb',
                           help='data root')
    argparser.add_argument('--output', type=str, default='data',
                           help='Output path of partitioned graph.')
    args = argparser.parse_args()

    start = time.time()
    if args.dataset == 'reddit':
        g, _ = load_reddit()
    elif args.dataset == 'ogb-product':
        g, _ = load_ogb('ogbn-products', root=args.root)
    elif args.dataset == 'ogbn-papers100M':
        g, _ = load_ogb('ogbn-papers100M', root=args.root)
    elif args.dataset == 'mag240m':
        g = load_mag240m()
    elif args.dataset == 'webuk':
        g = loadUK()
    elif args.dataset == 'friendster':
        g = loadFS()
    elif args.dataset == 'twitter':
        g = loadTW()
    elif args.dataset == 'fake':
        g = buildFakeGraph()

    print('load {} takes {:.3f} seconds'.format(args.dataset, time.time() - start))
    print('|V|={}, |E|={}'.format(g.number_of_nodes(), g.number_of_edges()))
    print('train: {}, valid: {}'.format(th.sum(g.ndata['train_mask']),
                                                  th.sum(g.ndata['val_mask'])))
    if args.balance_train:
        balance_ntypes = g.ndata['train_mask']
    else:
        balance_ntypes = None

    if args.undirected:
        print('calling to to_bidirected')
        sym_g = dgl.to_bidirected(g, readonly=True)
        for key in g.ndata:
            sym_g.ndata[key] = g.ndata[key]
        g = sym_g

    process = psutil.Process(os.getpid())
    print('Host memory usage before partition graph: {} GB'.format(process.memory_info().rss / 1e9))
    dgl.distributed.partition_graph(g, args.dataset, args.num_parts, args.output,
                                    part_method=args.part_method,
                                    balance_ntypes=balance_ntypes,
                                    balance_edges=args.balance_edges,
                                    num_trainers_per_machine=args.num_trainers_per_machine)
