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
    os.environ['MASTER_PORT'] = '12378'

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
  def __init__(self, dataloader, features, labels, min_vids, pc_queue, rank, num_epochs):
    Thread.__init__(self)
    self.rank = rank
    self.dataloader = dataloader
    self.features = features
    self.labels = labels
    self.min_vids = min_vids
    self.pc_queue = pc_queue
    self.num_epochs = num_epochs
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    print('cuda stream', s)
    dgl.ds.set_thread_local_stream(s)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        self.pc_queue.get_lock()
        for step, (input_nodes, seeds, blocks) in enumerate(self.dataloader):
          # batch_inputs = th.ones([input_nodes.shape[0], self.features.shape[1]], dtype=self.features.dtype, device=self.rank)
          # batch_labels = th.ones([seeds.shape[0]], dtype=self.labels.dtype, device=self.rank)
          batch_inputs, batch_labels = dgl.ds.load_subtensor(self.features, self.labels, input_nodes, seeds, self.min_vids)
          s.synchronize()
          self.pc_queue.put((batch_inputs, batch_labels, blocks))
          self.pc_queue.get_lock()
        self.pc_queue.stop_produce(1)

def show_thread():
  t = threading.currentThread()
  print('python thread id: {}, thread name: {}'.format(t.ident, t.getName()))

def run(rank, args, train_label):
    setup(rank, args.n_ranks)
    ds.init(rank, args.n_ranks)
    
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)

    g = dgl.add_self_loop(g)

    num_vertices = gpb._max_node_ids[-1]
    #test_sampling(num_vertices, g, rank)
    #time.sleep(2)
    #exit(0)
    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    print('rank {}, # global: {}, # local: {}'.format(rank, num_vertices, n_local_nodes))
    print('# in feats:', node_feats['_N/features'].shape[1])
    train_nid = th.masked_select(g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
    train_nid = dgl.ds.rebalance_train_nids(train_nid, args.batch_size, g.ndata[dgl.NID])

    n_classes = len(th.unique(train_label[th.logical_not(th.isnan(train_label))]))
    print('#labels:', n_classes)

    # print('# batch: ', train_nid.size()[0] / args.batch_size)
    th.distributed.barrier()
    #tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    train_g = g.formats(['csr'])
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    train_g = train_g.to(device)
    train_feature = node_feats['_N/features']
    in_feats = train_feature.shape[1]
    train_feature = train_feature.to(device)
    train_label = train_label.to(device)
    global_nid_map = train_g.ndata[dgl.NID]
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
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    th.distributed.barrier()
    stop_epoch = -1
    total = 0
    skip_epoch = 5
    data_buffer = SequentialQueue()
    sample_worker = Sampler(dataloader, train_feature, train_label, min_vids, data_buffer, rank, args.num_epochs)

    s = th.cuda.Stream(device=device)
    print('cuda stream', s)
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
        print('training time:', training_time)
        print("rank:", rank, toc - tic)
    sample_worker.join()
    print("rank:", rank, "sampling time:", total/(args.num_epochs - skip_epoch))
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
    args = parser.parse_args()
    
    all_labels = th.tensor([])
    for rank in range(args.n_ranks):
        _, node_feats, _, _, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)
        train_label = node_feats['_N/labels']
        all_labels = th.cat((all_labels, train_label), dim=0).long()

    mp.spawn(run,
          args=(args, all_labels),
          nprocs=args.n_ranks,
          join=True)
  
