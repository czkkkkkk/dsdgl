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
import random
#from dgl.ds.graph_partition_book import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class NeighborSampler(object):
    def __init__(self, g, num_vertices, device_min_vids, device_min_eids, global_nid_map, 
                    fanouts, sample_neighbors, device, load_feat=True):
        self.g = g
        self.num_vertices = num_vertices
        self.fanouts = fanouts
        self.sample_neighbors = sample_neighbors
        self.load_feat = load_feat
        self.device_min_vids = device_min_vids
        self.device_min_eids = device_min_eids
        self.global_nid_map = global_nid_map
        self.device = device

    '''
    suppose g, seed_nodes are all on gpu
    '''
    def sample_blocks(self, g, seeds, exclude_eids=None):
        blocks = []
        is_local = False
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, self.num_vertices,
                                             self.device_min_vids, self.device_min_eids,
                                             seeds, fanout, self.global_nid_map, is_local = is_local)
            is_local = False
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds)
            # block = ds.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

def run(rank, args, train_label):
    print('Start rank', rank, 'with args:', args)
    th.cuda.set_device(rank)
    setup(rank, args.n_ranks)
    ds.init(rank, args.n_ranks)
    
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    num_vertices = gpb._max_node_ids[-1]
    #test_sampling(num_vertices, g, rank)
    #time.sleep(2)
    #exit(0)
    g = dgl.add_self_loop(g)
    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    print('rank {}, # global: {}, # local: {}'.format(rank, num_vertices, n_local_nodes))
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
    train_nid = dgl.ds.rebalance_train_nids(train_nid, args.batch_size, g.ndata[dgl.NID])
    # print('# batch: ', train_nid.size()[0] / args.batch_size)
    th.distributed.barrier()
    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    train_g = g.formats(['csr'])
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    train_g = train_g.to(device)
    train_feature = node_feats['_N/features']
    train_feature = train_feature.to(device)
    train_label = train_label.to(device)
    global_nid_map = train_g.ndata[dgl.NID]
    #todo: transfer gpb to gpu
    min_vids = [0] + list(gpb._max_node_ids)
    min_vids = F.tensor(min_vids, dtype=F.int64).to(device)
    min_eids = [0] + list(gpb._max_edge_ids)
    min_eids = F.tensor(min_eids, dtype=F.int64).to(device)
    time.sleep(5)
    sampler = NeighborSampler(train_g, num_vertices,
                              min_vids,
                              min_eids,
                              global_nid_map,
                              [int(fanout) for fanout in args.fan_out.split(',')],
                              dgl.ds.sample_neighbors, device)

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
    print("start sampling")
    for epoch in range(args.num_epochs):
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            th.cuda.synchronize()
            print("start loading")
            batch_inputs, batch_labels = dgl.ds.load_subtensor(train_feature, train_label, input_nodes, min_vids)
            print(batch_inputs.shape, batch_labels.shape)
            th.distributed.barrier()
        toc = time.time()
        if epoch >= skip_epoch:
            total += (toc - tic)

        print("rank:", rank, toc - tic)
    print("rank:", rank, "sampling time:", total/(args.num_epochs - skip_epoch))
    cleanup()
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./data-8/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=8, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=10, type=int, help='Epochs')
    args = parser.parse_args()
    
    all_labels = th.tensor([])
    for rank in range(args.n_ranks):
        _, node_feats, _, _, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
        train_label = node_feats['_N/labels']
        all_labels = th.cat((all_labels, train_label), dim=0)

    mp.spawn(run,
          args=(args, all_labels),
          nprocs=args.n_ranks,
          join=True)
  
