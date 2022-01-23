
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
#from dgl.ds.graph_partition_book import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12375'

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

def test_sampling(num_vertices, g, rank):
    device = th.device('cuda:%d' % rank)
    if rank == 0:
        seeds = th.LongTensor([141625, 141734]).to(device)
    else:
        seeds = th.LongTensor([1, 2]).to(device)
    g = g.to(device)
    min_vids = th.LongTensor([0, 116366]).to(device)
    min_eids = th.LongTensor([0, 116366]).to(device)
    #print(seeds)
    frontier = ds.sample_neighbors(g, num_vertices, min_vids, min_eids, seeds, 2, g.ndata[dgl.NID], is_local=False)
    # try:
    block = dgl.to_block(frontier, seeds)
    seeds = block.srcdata[dgl.NID]
    # except:
    #     print(seeds)
    #     exit()
    #print(seeds)


def run(rank, args):
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
    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    train_g = g.formats(['csr'])
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    train_g = train_g.to(device)
    th.cuda.synchronize(device)
    train_feature = node_feats['_N/features']
    train_feature = train_feature.to(device)
    th.cuda.synchronize(device)
    train_label = node_feats['_N/labels']
    train_label = train_label.to(device)
    th.cuda.synchronize(device)
    global_nid_map = train_g.ndata[dgl.NID]
    #todo: transfer gpb to gpu
    min_vids = [0] + list(gpb._max_node_ids)
    min_eids = [0] + list(gpb._max_edge_ids)
    time.sleep(2)
    sampler = NeighborSampler(train_g, num_vertices,
                              F.tensor(min_vids, dtype=F.int64).to(device),
                              F.tensor(min_eids, dtype=F.int64).to(device),
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
    min_vids = F.tensor(min_vids, dtype=F.int64).to(device)
    for epoch in range(args.num_epochs):
        tic = time.time()
        for step, (input_nodes, seeds, blocks) in enumerate(dataloader):
            # batch_inputs = dgl.ds.load_subtensor(train_feature, input_nodes)
            test = th.tensor([2,3,1], dtype=F.int64).to(device)
            test_feature = th.tensor([[0,0], [1,1], [2,2], [3,3]], dtype=F.int64).to(device)
            ret = dgl.ds.load_subtensor(test_feature, test, min_vids)
            print(rank, ret)
            exit()
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
    parser.add_argument('--part_config', default='./data-2/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=2, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=1, type=int, help='Epochs')
    args = parser.parse_args()

    mp.spawn(run,
          args=(args,),
          nprocs=args.n_ranks,
          join=True)
  
