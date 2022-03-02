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
        is_local = True
        for fanout in self.fanouts:
            # For each seed node, sample ``fanout`` neighbors.
            frontier = self.sample_neighbors(self.g, self.num_vertices,
                                             self.device_min_vids, self.device_min_eids,
                                             seeds, fanout, self.global_nid_map, 
                                             is_local = is_local)
            
            is_local = False
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(frontier, seeds, min_vids = self.device_min_vids)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks

class ParallelNodeDataLoader(object):
  def __init__(self, dataloader):
    self.dataloader = dataloader
    self.iterator = dataloader.__iter__()
    self.lock = threading.Lock()
    self.epoch = 0
  
  def next_epoch(self):
    self.iterator = self.dataloader.__iter__()
    self.epoch += 1

  def current_epoch(self):
    return self.epoch

  def next(self):
    return next(self.iterator)

class PCQueue(object):
  def __init__(self, capacity):
    self.capacity = threading.Semaphore(capacity)
    self.product = threading.Semaphore(0)
    self.buffer = Queue()
  
  def get(self):
    self.product.acquire()
    item = self.buffer.get()
    self.buffer.task_done()
    self.capacity.release()
    return item
  
  def put(self, item):
    self.capacity.acquire()
    self.buffer.put(item)
    self.product.release()
  
  def stop_produce(self, num):
    for i in range(num):
      self.put(None)

class Sampler(Thread):
  finish_sampler_ = 0
  lock_ = threading.Lock()
  sem_ = threading.Semaphore(0)

  def __init__(self, dataloader, pc_queue, rank, num_epochs, thread_id, sampler_number, batches):
    Thread.__init__(self)
    self.rank = rank
    self.dataloader = dataloader
    self.pc_queue = pc_queue
    self.num_epochs = num_epochs
    self.thread_id = thread_id
    self.sampler_number = sampler_number
    self.batches = batches
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    print('cuda stream', s)
    dgl.ds.set_thread_local_stream(s, self.thread_id)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        for j in range(self.batches):
          batch = self.dataloader.next()
          s.synchronize()
          (input_nodes, seeds, blocks) = batch
          self.pc_queue.put((input_nodes, seeds, blocks))
        Sampler.lock_.acquire()
        Sampler.finish_sampler_ += 1
        if Sampler.finish_sampler_ == self.sampler_number:
          Sampler.finish_sampler_ = 0
          self.dataloader.next_epoch()
          for i in range(self.sampler_number):
            Sampler.sem_.release()
        Sampler.lock_.release()
        Sampler.sem_.acquire()

class SubtensorLoader(Thread):
  finish_loader_ = 0
  lock_ = threading.Lock()
  sem_ = threading.Semaphore(0)

  def __init__(self, features, labels, min_vids, in_pc_queue, out_pc_queue, rank, num_epochs, thread_id, loader_number, batches):
    Thread.__init__(self)
    self.rank = rank
    self.features = features
    self.labels = labels
    self.min_vids = min_vids
    self.in_pc_queue = in_pc_queue
    self.out_pc_queue = out_pc_queue
    self.num_epochs = num_epochs
    self.thread_id = thread_id
    self.loader_number = loader_number
    self.batches = batches
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    print('cuda stream', s)
    dgl.ds.set_thread_local_stream(s, self.thread_id)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        for j in range(self.batches):
          sample_result = self.in_pc_queue.get()
          input_nodes = sample_result[0]
          seeds = sample_result[1]
          blocks = sample_result[2]
          batch_inputs, batch_labels = dgl.ds.load_subtensor(self.features, self.labels, input_nodes, seeds, self.min_vids)
          s.synchronize()
          self.out_pc_queue.put((batch_inputs, batch_labels, blocks))
        SubtensorLoader.lock_.acquire()
        SubtensorLoader.finish_loader_ += 1
        if SubtensorLoader.finish_loader_ == self.loader_number:
          SubtensorLoader.finish_loader_ = 0
          self.out_pc_queue.stop_produce(1)
          for i in range(self.loader_number):
            SubtensorLoader.sem_.release()
        SubtensorLoader.lock_.release()
        SubtensorLoader.sem_.acquire()

def show_thread():
  t = threading.currentThread()
  print('python thread id: {}, thread name: {}'.format(t.ident, t.getName()))
    
def run(rank, args, train_label):
    print('num threads: {}, iterop threads: {}'.format(th.get_num_threads(), th.get_num_interop_threads()))
    print('Start rank', rank, 'with args:', args)
    th.cuda.set_device(rank)
    th.set_num_threads(1)
    th.set_num_interop_threads(1)
    print('num threads: {}, iterop threads: {}'.format(th.get_num_threads(), th.get_num_interop_threads()))
    setup(rank, args.n_ranks)
    sampler_number = 4
    loader_number = 4
    ds.init(rank, args.n_ranks, thread_num=sampler_number + loader_number)
    
    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(args.part_config, rank)

    g = dgl.add_self_loop(g)

    num_vertices = gpb._max_node_ids[-1]
   
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
    skip_epoch = 6
    sample_data_buffer = PCQueue(10)
    subtensor_data_buffer = PCQueue(10)

    s = th.cuda.Stream(device=device)
    print('cuda stream', s)
    dgl.ds.set_device_thread_local_stream(device, s)

    my_dataloader = ParallelNodeDataLoader(dataloader)
    total_batches = dataloader.__len__()
    print("rank", rank, "batch number", total_batches)

    sample_workers = []
    acc_batches = 0
    for i in range(sampler_number):
      if i == sampler_number - 1:
        cur_batches = total_batches - acc_batches
      else:
        cur_batches = total_batches // sampler_number
      print("rank", rank, "sampler", i, "batches", cur_batches)
      acc_batches += cur_batches
      thread_id = i
      sample_workers.append(Sampler(my_dataloader, sample_data_buffer, rank, args.num_epochs, thread_id, sampler_number, cur_batches))
      sample_workers[i].start()

    load_workers = []
    acc_batches = 0
    for i in range(loader_number):
      if i == loader_number - 1:
        cur_batches = total_batches - acc_batches
      else:
        cur_batches = total_batches // loader_number
      print("rank", rank, "loader", i, "batches", cur_batches)
      acc_batches += cur_batches
      thread_id = i + sampler_number
      load_workers.append(SubtensorLoader(train_feature, train_label, min_vids, sample_data_buffer, subtensor_data_buffer, rank, args.num_epochs, thread_id, loader_number, cur_batches))
      load_workers[i].start()
  
    train_time = 0
    for epoch in range(args.num_epochs):
        tic = time.time()
        step = 0
        while True:
          batch_data = subtensor_data_buffer.get()
          if batch_data is None:
            break
          begin = time.time()
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
            th.distributed.barrier()
          step += time.time() - begin

        toc = time.time()
        if epoch >= skip_epoch:
            total += (toc - tic)
            train_time += step
        print("rank:", rank, toc - tic)
    for i in range(sampler_number):
      sample_workers[i].join()
    for i in range(loader_number):
      load_workers[i].join()
    print("rank:", rank, "end2end time:", total/(args.num_epochs - skip_epoch))
    print("train:", rank, train_time/(args.num_epochs - skip_epoch))
    cleanup()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test', type=str, help='graph name')
    parser.add_argument('--part_config', default='./reddit-data-2/reddit.json', type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=2, type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024, type=int, help='Batch size')
    parser.add_argument('--num_hidden', default=16, type=int, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fan_out', default="25,10", type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=20, type=int, help='Epochs')
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
  
