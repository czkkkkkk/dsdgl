
import os

from dgl.random import seed
os.environ['DGLBACKEND']='pytorch'
from dgl.data import register_data_args, load_data
import argparse
import torch as th
import dgl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import dgl.ds as ds
import dgl.backend as F
#from dgl.ds.graph_partition_book import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class NeighborSampler(object):
    def __init__(self, g, device_min_vids, fanouts, sample_neighbors, load_feat=True):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.load_feat = load_feat
        self.device_min_vids = device_min_vids

    '''
    suppose g, seed_nodes are all on gpu
    '''
    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        for fanout in self.fanouts:
            print("graph:", self.g)
            print("seeds:", seeds)
            print("fanout:", [fanout])
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, self.device_min_vids, seeds, 
                                             fanout)
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]

            blocks.insert(0, block)

        return blocks

def run(rank, args):
    print('Start rank', rank, 'with args:', args)

    setup(rank, args.n_ranks)
    ds.init(rank, args.n_ranks)
    
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    # print(g)

    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])

    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    train_g = g.formats(['csc'])
    train_g = train_g.to(device)
    #todo: transfer gpb to gpu

    print(F.tensor(gpb._max_node_ids, dtype=F.int64).to(device))
    print(train_nid.dtype)
    print(train_g)
    #pb = DsDevicePartitionBook(gpb, device)
    #print(pb._max_node_ids_gpu)
    #print(train_g.ndata)
    min_ids = [0] + list(gpb._max_node_ids)
    sampler = NeighborSampler(g, F.tensor(min_ids, dtype=F.int64).to(device), 
                              [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.ds.sample_neighbors)

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    for epoch in range(args.num_epochs):

        for step, blocks in enumerate(dataloader):
          exit()

    cleanup()
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./data/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=2, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=1, type=int, help='Epochs')
    args = parser.parse_args()

    mp.spawn(run,
          args=(args,),
          nprocs=args.n_ranks,
          join=True)
  