
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
import time
import torch
from load_graph import load_reddit, inductive_split

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12475'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class NeighborSampler(object):
    def __init__(self, row_idx, g, num_vertices, fanouts, sample_neighbors, device):
        self.row_idx = row_idx
        self.g = g
        self.num_vertices = num_vertices
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.device = device

    '''
    suppose g uva, seed_nodes are on gpu
    '''
    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        # seeds = seeds.to(self.device)
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.row_idx, self.g, seeds, self.num_vertices, fanout)
            block = dgl.to_block(frontier, seeds)
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

def run(rank, args, data):
    th.cuda.set_device(rank)
    setup(rank, args.n_ranks)
    #ds.init(rank, args.n_ranks)
    
    n_classes, train_g = data
    train_nfeat = train_g.ndata.pop('features')
    train_labels = train_g.ndata.pop('labels')

    train_mask = train_g.ndata['train_mask']
    train_nid = train_mask.nonzero().squeeze()
    chunk_size = train_nid.shape[0] / args.n_ranks
    start = int(rank * chunk_size)
    end = int((rank + 1) * chunk_size)
    if end > train_nid.shape[0]:
        end = train_nid.shape[0]
    train_nid = train_nid[start : end]
    num_vertices = train_nfeat.shape[0]
    train_nid = train_nid.to(rank)
    train_g = train_g.formats(['csr'])

    # row_idx on gpu, cols in train_g on uva
    row_idx = ds.pin_graph(train_g, rank)
    sampler = NeighborSampler(row_idx, train_g, num_vertices,
                              [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.ds.sample_neighbors_uva, rank)

    dataloader = dgl.dataloading.NodeDataLoader(
        train_g,
        train_nid,
        sampler,
        device=rank,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0)

    th.distributed.barrier()
    stop_epoch = -1
    if args.n_ranks == 2:
        stop_epoch = 74
    elif args.n_ranks == 4:
        stop_epoch = 33
    elif args.n_ranks == 8:
        stop_epoch = 17
    total = 0
    skip_epoch = 5
    for epoch in range(args.num_epochs):
        tic = time.time()
        cnt = 0
        for step, blocks in enumerate(dataloader):
            cnt += 1
            th.distributed.barrier()
            if cnt == stop_epoch:
                break
        toc = time.time()
        if epoch >= skip_epoch:
            total += (toc - tic)

        print("rank:", rank, toc - tic, "cnt:", cnt)
    print("rank:", rank, "sampling time:", total/(args.num_epochs - skip_epoch))
    cleanup()
  


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='GCN')
  register_data_args(parser)
  parser.add_argument('--n_ranks', default=2, type=int, help='Number of ranks')
  parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
  parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
  parser.add_argument('--num_epochs', default=20, type=int, help='Epochs')
  args = parser.parse_args()

  g, n_classes = load_reddit()
  g = dgl.as_heterograph(g)
  train_g = g
  train_g.create_formats_()
  # Pack data
  data = n_classes, train_g

  mp.spawn(run,
            args=(args, data),
            nprocs=args.n_ranks,
            join=True)
