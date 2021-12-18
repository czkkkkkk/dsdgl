
import os
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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class NeighborSampler(object):
    def __init__(self, g, fanouts, sample_neighbors, load_feat=True):
        self.g = g
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.load_feat=load_feat

    def sample_blocks(self, seeds):
        seeds = th.LongTensor(np.asarray(seeds))
        blocks = []
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, seeds, fanout, replace=True)
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
    exit(0)

    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    print(g)

    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])

    sampler = NeighborSampler(g, [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.distributed.sample_neighbors)

    dataloader = dgl.dataloading.NodeDataLoader(
        g,
        train_nid,
        sampler,
        device='cuda:{}'.format(rank),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False)

    for epoch in range(args.num_epochs):

        for step, blocks in enumerate(dataloader):
          pass

    cleanup()
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./test_graph/test.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=2, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    args = parser.parse_args()

    mp.spawn(run,
          args=(args,),
          nprocs=args.n_ranks,
          join=True)
  