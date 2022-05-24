import os

from dgl.random import seed
os.environ['DGLBACKEND']='pytorch'
from dgl.data import register_data_args, load_data
import dgl.ndarray as nd
import argparse
import torch as th
import dgl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import numpy as np
import time
import dgl.ds as ds
import dgl.backend as F
import time
import random
from model import SAGE
import threading
from threading import Thread
from queue import Queue
#from dgl.ds.graph_partition_book import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12499'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class NeighborSampler(object):
    def __init__(self, g, num_vertices, device_min_vids, device_min_eids, global_nid_map, 
                    fanouts, sample_neighbors, device, load_feat=True, weight=None, is_bias=False):
        self.g = g
        self.num_vertices = num_vertices
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.load_feat = load_feat
        self.device_min_vids = device_min_vids
        self.device_min_eids = device_min_eids
        self.global_nid_map = global_nid_map
        self.device = device
        self.weight = weight
        self.bias = is_bias

    '''
    suppose g, seed_nodes are all on gpu
    '''
    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        is_local = True 
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier, seeds = self.sample_neighbors(self.g, self.num_vertices,
                                             self.device_min_vids, self.device_min_eids,
                                             seeds, fanout, self.global_nid_map, is_local = is_local,
                                             is_bias = self.bias, weight = self.weight)
            
            is_local = False
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds, min_vids=self.device_min_vids)
            # block = ds.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

def run(rank, args):
    setup(rank, args.n_ranks)
    ds.init(rank, args.n_ranks)
    
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    g = dgl.add_self_loop(g)

    num_vertices = gpb._max_node_ids[-1]
    #exit(0)
    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    print('rank {}, # global: {}, # local: {}'.format(rank, num_vertices, n_local_nodes))
    print('edges: ', g.number_of_edges())
    # print('# in feats:', node_feats['_N/features'].shape[1])
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
    train_nid = dgl.ds.rebalance_train_nids(train_nid, args.batch_size, g.ndata[dgl.NID])

    # print('# batch: ', train_nid.size()[0] / args.batch_size)
    th.distributed.barrier()
    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    train_g = g.reverse().formats(['csr'])
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    index = train_g.adj_sparse('csr')[0]
    adj = train_g.adj_sparse('csr')[1]
    weights = th.ones(adj.shape, dtype=th.int32)
    num_vertices = index.shape[0] - 1
    print(num_vertices)
    print("rank", rank, index)
    for i in range(num_vertices):
        offset = index[i]
        degree = index[i + 1] - offset
        weights[offset: offset + degree] = th.arange(0, degree, dtype=th.int32)
    weights = F.tensor(weights, dtype=F.int32).to(device)
    th.cuda.synchronize()
    print("rank", rank, weights)
    print("rank", rank, weights.shape)
    dgl.ds.cache_graph(train_g, args.graph_cache_ratio)
    global_nid_map = train_g.ndata[dgl.NID].to(device)
    train_g = None
    #todo: transfer gpb to gpu
    min_vids = [0] + list(gpb._max_node_ids)
    min_vids = F.tensor(min_vids, dtype=F.int64).to(device)
    min_eids = [0] + list(gpb._max_edge_ids)
    min_eids = F.tensor(min_eids, dtype=F.int64).to(device)

    fanout = [int(fanout) for fanout in args.fan_out.split(',')]
    sampler = NeighborSampler(train_g, num_vertices,
                              min_vids,
                              min_eids,
                              global_nid_map,
                              fanout,
                              dgl.ds.sample_neighbors, device, weight=weights, is_bias=True)

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    th.distributed.barrier()
    stop_epoch = -1
    total = 0
    skip_epoch = 5

    s = th.cuda.Stream(device=device)
    dgl.ds.set_device_thread_local_stream(device, s)

    for epoch in range(args.num_epochs):
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            pass
        th.cuda.synchronize()
        th.distributed.barrier()
        toc = time.time()
        if rank == 0:
            print('Rank: ', rank, 'sampling time', toc - tic)
        if epoch >= skip_epoch:
            total += (toc - tic)

    if rank == 0:
        print("rank:", rank, "sampling time:", total/(args.num_epochs - skip_epoch))
    cleanup()
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./data-1/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=1, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--cache_ratio', default=100, type=int, help='Percentages of features on GPUs')
    parser.add_argument('--graph_cache_ratio', default=100, type=int, help='Ratio of edges cached in the GPU')
    parser.add_argument('--enable_profiler', action='store_true',
                           help='Profiler')
    parser.add_argument('--num_hidden', default=256,
                        type=int, help='Hidden size')
    parser.add_argument('--feat_mode', default='AllCache', type=str,
                        help='Feature cache mode. (AllCache, PartitionCache, ReplicateCache)')
    args = parser.parse_args()
    

    mp.spawn(run,
          args=(args,),
          nprocs=args.n_ranks,
          join=True)
  
