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
import os, psutil
#from dgl.ds.graph_partition_book import *

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12478'

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
        is_local = True 
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            # No need to pass graph now
            frontier = self.sample_neighbors(g, self.num_vertices,
                                             self.device_min_vids, self.device_min_eids,
                                             seeds, fanout, self.global_nid_map, is_local = is_local)
            
            is_local = False
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds, min_vids=self.device_min_vids)
            # block = ds.to_block(frontier, seeds)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

class SequentialQueue(object):
  def __init__(self):
    self.product = threading.Semaphore(1)
    self.capacity = threading.Semaphore(0)
    self.buffer = Queue()
  
  def get_lock(self):
    self.product.acquire()
  
  def put(self, item):
    self.buffer.put(item)
    self.capacity.release()
  
  def get(self):
    self.capacity.acquire()
    item = self.buffer.get()
    return item
  
  def release_lock(self):
    self.product.release()
  
  def stop_produce(self, num):
    for i in range(num):
      # self.get_lock()
      self.put(None)

class Sampler(Thread):
  def __init__(self, dataloader, labels, min_vids, pc_queue, feat_dim, rank, num_epochs):
    Thread.__init__(self)
    self.rank = rank
    self.dataloader = dataloader
    self.labels = labels
    self.min_vids = min_vids
    self.pc_queue = pc_queue
    self.num_epochs = num_epochs
    self.feat_dim = feat_dim
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    print('cuda stream', s)
    dgl.ds.set_thread_local_stream(s)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        self.pc_queue.get_lock()
        start_ts = time.time()
        sampling_time = 0
        loading_time = 0
        for step, (input_nodes, seeds, blocks) in enumerate(self.dataloader):
          s.synchronize()
          sample_ts = time.time()
          batch_inputs, batch_labels = dgl.ds.load_subtensor(self.labels, input_nodes, seeds, self.min_vids, self.feat_dim)
          s.synchronize()
          load_ts = time.time()
          self.pc_queue.put((batch_inputs, batch_labels, blocks))
          self.pc_queue.get_lock()
          sampling_time += sample_ts - start_ts
          loading_time += load_ts - sample_ts
          start_ts = time.time()
        print('[Epoch {}], rank {}, sample time {}, load time {}'.format(i, self.rank, sampling_time, loading_time))
        self.pc_queue.stop_produce(1)

def show_thread():
  t = threading.currentThread()
  print('python thread id: {}, thread name: {}'.format(t.ident, t.getName()))

def run(rank, args):
    print('Start rank', rank, 'with args:', args)
    process = psutil.Process(os.getpid())
    setup(rank, args.n_ranks)
    ds.init(rank, args.n_ranks)
    
    print('Rank {}, Host memory usage before load partition: {} GB'.format(rank, process.memory_info().rss / 1e9))
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
    print('Rank {}, Host memory usage after load partition: {} GB'.format(rank, process.memory_info().rss / 1e9))

    g = dgl.add_self_loop(g)


    num_vertices = gpb._max_node_ids[-1]
    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    if '_N/features' not in node_feats:
      print('Using fake features with feat dim: ', args.in_feats)
      node_feats['_N/features'] = th.ones([n_local_nodes, args.in_feats], dtype=th.float32)
      node_feats['_N/labels'] = th.zeros([n_local_nodes], dtype=th.float32)

    in_feats = node_feats['_N/features'].shape[1] 
    print('rank {}, # global: {}, # local: {}'.format(rank, num_vertices, n_local_nodes))
    print('# in feats:', node_feats['_N/features'].shape[1])
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
    train_nid = dgl.ds.rebalance_train_nids(train_nid, args.batch_size, g.ndata[dgl.NID])

    train_label = dgl.ds.allgather_train_labels(node_feats['_N/labels'])
    n_classes = len(th.unique(train_label[th.logical_not(th.isnan(train_label))]))
    print('#labels:', n_classes)

    # print('# batch: ', train_nid.size()[0] / args.batch_size)
    th.distributed.barrier()
    dgl.ds.cache_feats(args.feat_mode, g, node_feats['_N/features'], args.cache_ratio)
    del node_feats['_N/features']
    print('Rank {}, Host memory usage after cache feats: {} GB'.format(rank, process.memory_info().rss / 1e9))
    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    print('Rank {}, Host memory usage before create format: {} GB'.format(rank, process.memory_info().rss / 1e9))
    train_g = g.reverse().formats(['csr'])
    g = None
    print('Rank {}, Host memory usage after create format: {} GB'.format(rank, process.memory_info().rss / 1e9))
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    global_nid_map = train_g.ndata[dgl.NID].to(device)
    dgl.ds.cache_graph(train_g, args.graph_cache_ratio)
    train_g = None
    # train_g = train_g.to(device)
    train_label = train_label.to(device)
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

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, n_classes, len(fanout), th.relu, args.dropout)
    model = model.to(device)
    if args.n_ranks > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    th.distributed.barrier()
    stop_epoch = -1
    total = 0
    skip_epoch = 3
    data_buffer = SequentialQueue()
    sample_worker = Sampler(dataloader, train_label, min_vids, data_buffer, in_feats, rank, args.num_epochs)

    s = th.cuda.Stream(device=device)
    dgl.ds.set_device_thread_local_stream(device, s)

    sample_worker.start()

    for epoch in range(args.num_epochs):
        tic = time.time()

        training_time = 0
        while True:
          batch_data = data_buffer.get()
          # print('buffer size:', data_buffer.buffer.qsize())
          if batch_data is None:
            data_buffer.release_lock()
            break
          start_ts = time.time()
          with th.cuda.stream(s):
          
            batch_inputs = batch_data[0]
            batch_labels = batch_data[1]
            blocks = batch_data[2]

            batch_pred = model(blocks, batch_inputs)
            loss = loss_fcn(batch_pred, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            s.synchronize()
          data_buffer.release_lock()
          end_ts = time.time()
          training_time += end_ts - start_ts

        toc = time.time()
        if epoch >= skip_epoch:
            total += (toc - tic)
        print('rank', rank, 'training time:', training_time)
        print("rank:", rank, toc - tic)
    sample_worker.join()
    print("rank:", rank, "e2e time:", total/(args.num_epochs - skip_epoch))
    cleanup()
  


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./data-1/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=1, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--num_hidden', default=16, type=int, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=10, type=int, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--feat_mode', default='AllCache', type=str, help='Feature cache mode. (AllCache, PartitionCache, ReplicateCache)')
    parser.add_argument('--cache_ratio', default=100, type=int, help='Percentages of features on GPUs')
    parser.add_argument('--graph_cache_ratio', default=100, type=int, help='Ratio of edges cached in the GPU')
    parser.add_argument('--in_feats', default=256, type=int, help='In feature dimension used when the graph do not have feature')
    args = parser.parse_args()
    

    mp.spawn(run,
          args=(args, ),
          nprocs=args.n_ranks,
          join=True)
  
