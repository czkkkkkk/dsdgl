import psutil
from utils import GPUMonitor
from pynvml import *
import sys
from torch.nn.parallel import DistributedDataParallel
from pc_queue import *
from queue import Queue
from threading import Thread
import threading
from model import SAGE
import random
import dgl.backend as F
import dgl.ds as ds
import time
import numpy as np
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import dgl
import torch as th
import argparse
import torchmetrics.functional as MF
import dgl.ndarray as nd
from dgl.data import register_data_args, load_data
import os

from dgl.random import seed

os.environ['DGLBACKEND'] = 'pytorch'


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12377'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)

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
            frontier, seeds = self.sample_neighbors(self.g, self.num_vertices,
                                             self.device_min_vids, self.device_min_eids,
                                             seeds, fanout, self.global_nid_map,
                                             is_local=is_local)

            is_local = False
            # Then we compact the frontier into a bipartite graph for message passing.
            block = dgl.to_block(
                frontier, seeds, min_vids=self.device_min_vids)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[dgl.NID]
            blocks.insert(0, block)
        return blocks


class ParallelNodeDataLoader(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = dataloader.__iter__()

    def next_epoch(self):
        self.iterator = self.dataloader.__iter__()

    def next(self):
        return next(self.iterator)


class Sampler(Thread):
    finish_sampler_ = 0
    lock_ = threading.Lock()
    sem_ = threading.Semaphore(0)

    def __init__(self, dataloader, mpmc_queue, rank, num_epochs, thread_id, sampler_number, batches_per_sampler):
        Thread.__init__(self)
        self.rank = rank
        self.dataloader = dataloader
        self.mpmc_queue = mpmc_queue
        self.num_epochs = num_epochs
        self.thread_id = thread_id
        self.sampler_number = sampler_number
        self.batches_per_sampler = batches_per_sampler
        self.batches = batches_per_sampler[thread_id]
        self.total_batches = sum(batches_per_sampler)

    def next_epoch(self):
        self.thread_id = (
            self.thread_id + self.total_batches) % self.sampler_number

    def run(self):
        th.cuda.set_device(self.rank)
        s = th.cuda.Stream(device=self.rank)
        dgl.ds.set_thread_local_stream(s, self.thread_id, 0)
        with th.cuda.stream(s):
            for i in range(self.num_epochs):
                for j in range(self.batches):
                    ds.set_queue_size(self.mpmc_queue.size())
                    batch = self.dataloader.next()
                    # s.synchronize()
                    (input_nodes, seeds, blocks) = batch
                    self.mpmc_queue.put(
                        (input_nodes, seeds, blocks), self.thread_id)
                Sampler.lock_.acquire()
                Sampler.finish_sampler_ += 1
                if Sampler.finish_sampler_ == self.sampler_number:
                    Sampler.finish_sampler_ = 0
                    self.dataloader.next_epoch()
                    for i in range(self.sampler_number):
                        Sampler.sem_.release()
                Sampler.lock_.release()
                Sampler.sem_.acquire()
                self.next_epoch()


class SubtensorLoader(Thread):
    finish_loader_ = 0
    lock_ = threading.Lock()
    sem_ = threading.Semaphore(0)

    def __init__(self, labels, min_vids, in_mpmc_queue, out_pc_queue, rank, num_epochs, feat_dim, thread_id, sampler_number, loader_number, batches_per_loader):
        Thread.__init__(self)
        self.rank = rank
        self.labels = labels
        self.min_vids = min_vids
        self.in_mpmc_queue = in_mpmc_queue
        self.out_pc_queue = out_pc_queue
        self.num_epochs = num_epochs
        self.feat_dim = feat_dim
        self.thread_id = thread_id
        self.sampler_number = sampler_number
        self.loader_number = loader_number
        self.batches_per_loader = batches_per_loader
        self.batches = batches_per_loader[thread_id]
        self.total_batches = sum(batches_per_loader)

    def next_epoch(self):
        self.thread_id = (
            self.thread_id + self.total_batches) % self.loader_number

    def run(self):
        th.cuda.set_device(self.rank)
        s = th.cuda.Stream(device=self.rank)
        dgl.ds.set_thread_local_stream(
            s, self.thread_id + self.sampler_number, 1)
        with th.cuda.stream(s):
            for i in range(self.num_epochs):
                for i in range(self.batches):
                    sample_result = self.in_mpmc_queue.get(self.thread_id)
                    ds.set_queue_size(self.out_pc_queue.size())
                    input_nodes = sample_result[0]
                    seeds = sample_result[1]
                    blocks = sample_result[2]
                    batch_inputs, batch_labels = dgl.ds.load_subtensor(
                        self.labels, input_nodes, seeds, self.min_vids, self.feat_dim)
                    # s.synchronize()
                    self.out_pc_queue.put(
                        (batch_inputs, batch_labels, blocks), self.thread_id)
                SubtensorLoader.lock_.acquire()
                SubtensorLoader.finish_loader_ += 1
                if SubtensorLoader.finish_loader_ == self.loader_number:
                    SubtensorLoader.finish_loader_ = 0
                    for i in range(self.loader_number):
                        SubtensorLoader.sem_.release()
                SubtensorLoader.lock_.release()
                SubtensorLoader.sem_.acquire()
                self.next_epoch()


def print_nvidia_mem(prefix):
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    print('{}, NVIDIA free {} G, used {} G'.format(
        prefix, info.free / 1e9, info.used / 1e9))


def calculate_ratio(gb, size, entry_size_in_byte):
    n_cached = gb * 1024 * 1024 * 1024 / entry_size_in_byte
    ret = n_cached / size * 100
    return min(100., ret)


def run(rank, args):
    print('Start rank', rank, 'with args:', args)
    th.cuda.set_device(rank)
    setup(rank, args.n_ranks)
    sampler_number = 1
    loader_number = 1
    nvmlInit()
    ds.init(rank, args.n_ranks, thread_num=sampler_number +
            loader_number, enable_kernel_control=False)

    # load partitioned graph
    g, node_feats, edge_feats, gpb, _, _, _ = dgl.distributed.load_partition(
        args.part_config, rank)
    process = psutil.Process(os.getpid())
    print('Host memory usage after load partition {} GB'.format(
        process.memory_info().rss / 1e9))
    g = dgl.add_self_loop(g)

    n_local_nodes = node_feats['_N/train_mask'].shape[0]
    if '_N/features' not in node_feats:
        print('Using fake features with feat dim: ', args.in_feats)
        node_feats['_N/features'] = th.ones(
            [n_local_nodes, args.in_feats], dtype=th.float32)
        node_feats['_N/labels'] = th.zeros([n_local_nodes], dtype=th.float32)

    if args.graph_cache_gb != -1:
        args.graph_cache_ratio = calculate_ratio(
            args.graph_cache_gb, g.number_of_edges(), 8)
    if args.feat_cache_gb != -1:
        args.cache_ratio = calculate_ratio(
            args.feat_cache_gb, n_local_nodes, 4 * node_feats['_N/features'].shape[1])

    in_feats = node_feats['_N/features'].shape[1]

    print('Rank {}, Host memory usage before cache feats: {} GB'.format(
        rank, process.memory_info().rss / 1e9))
    dgl.ds.cache_feats(args.feat_mode, g,
                       node_feats['_N/features'], args.cache_ratio)
    del node_feats['_N/features']
    print('Rank {}, Host memory usage after cache feats: {} GB'.format(
        rank, process.memory_info().rss / 1e9))

    num_vertices = int(gpb._max_node_ids[-1])

    print('Graph cache ratio {}, feature cache ratio {}'.format(
        args.graph_cache_ratio, args.cache_ratio))

    print('rank {}, # global: {}, # local: {}'.format(
        rank, num_vertices, n_local_nodes))
    print('# in feats:', in_feats)
    train_nid = th.masked_select(
        g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
    train_nid = dgl.ds.rebalance_train_nids(
        train_nid, args.batch_size, g.ndata[dgl.NID])
    train_label = dgl.ds.allgather_train_labels(node_feats['_N/labels'])
    n_classes = len(
        th.unique(train_label[th.logical_not(th.isnan(train_label))]))
    print('#labels:', n_classes)
    print('Rank {}, subgraph nodes {} B, subgraph edges {} B'.format(
        rank, g.number_of_nodes() / 1e9, g.number_of_edges() / 1e9))

    # print('# batch: ', train_nid.size()[0] / args.batch_size)
    th.distributed.barrier()
    # tansfer graph and train nodes to gpu
    device = th.device('cuda:%d' % rank)
    train_nid = train_nid.to(device)
    # train_g = g.reverse().formats(['csr'])
    train_g = g.formats(['csr'])
    g = None
    train_g = dgl.ds.csr_to_global_id(train_g, train_g.ndata[dgl.NID])
    global_nid_map = train_g.ndata[dgl.NID].to(device)
    dgl.ds.cache_graph(train_g, args.graph_cache_ratio)
    train_g = None
    print('Rank {}, pytorch memory usage after move train_g to device : {} GB'.format(
        rank, th.cuda.memory_allocated(rank) / 1e9))
    train_label = train_label.to(device)
    # todo: transfer gpb to gpu
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
        shuffle=True,
        drop_last=False,
        num_workers=0)

    s = th.cuda.Stream(device=device)
    dgl.ds.set_device_thread_local_stream(device, s)

    # Define model and optimizer
    with th.cuda.stream(s):
        model = SAGE(in_feats, args.num_hidden, n_classes,
                     len(fanout), th.relu, args.dropout)
        model = model.to(device)
        if args.n_ranks > 1:
            model = DistributedDataParallel(
                model, device_ids=[rank], output_device=rank)
        loss_fcn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('GPU memory usage before train: {} GB'.format(
        th.cuda.memory_allocated(rank) / 1e9))
    print_nvidia_mem('Before train')

    th.distributed.barrier()
    stop_epoch = -1
    total = 0
    skip_epoch = 5
    sample_data_buffer = MPMCQueue(1, sampler_number, loader_number)
    # sample_data_buffer = MPMCQueue_simple(10, sampler_number, loader_number)
    subtensor_data_buffer = MPMCQueue(1, loader_number, 1)
    # subtensor_data_buffer = PCQueue(10)
    my_dataloader = ParallelNodeDataLoader(dataloader)

    total_batches = dataloader.__len__()
    batch_per_sampler = divide(total_batches, sampler_number)
    print("batch per sampler", batch_per_sampler)
    sample_workers = []
    for i in range(sampler_number):
        thread_id = i
        sample_workers.append(Sampler(my_dataloader, sample_data_buffer, rank,
                              args.num_epochs, thread_id, sampler_number, batch_per_sampler))
        sample_workers[i].start()

    batch_per_loader = divide(total_batches, loader_number)
    print("batch per loader", batch_per_loader)
    load_workers = []
    for i in range(loader_number):
        thread_id = i
        load_workers.append(SubtensorLoader(train_label, min_vids, sample_data_buffer, subtensor_data_buffer,
                            rank, args.num_epochs, in_feats, thread_id, sampler_number, loader_number, batch_per_loader))
        load_workers[i].start()

    monitor = GPUMonitor(rank)
    time_array = []
    acc_array = []
    loss_array = []
    train_time = 0
    time_begin = time.time()
    for epoch in range(args.num_epochs):
        if epoch == skip_epoch:
            monitor.start()
        tic = time.time()
        step = 0
        for i in range(total_batches):
            batch_data = subtensor_data_buffer.get(0)
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

            # if i % args.log_every == 0:
            #     #acc = compute_acc(batch_pred, batch_labels)
            #     time_array.append(time.time() - time_begin)
            #     acc = MF.accuracy(batch_pred, batch_labels)
            #     acc = acc.reshape([-1]).cpu()
            #     loss = loss.reshape([-1]).cpu()
            #     dist.all_reduce(acc)
            #     acc /= args.n_ranks
            #     dist.all_reduce(loss)
            #     loss /= args.n_ranks

            #     acc_array.append(acc.item())
            #     loss_array.append(loss.item())
            #     if rank == 0:
            #         print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Train Acc {:.4f} | GPU {:.1f} MB'.format(
            #             epoch, i, loss.item(), acc.item(), th.cuda.max_memory_allocated() / 1000000))

        toc = time.time()
        if epoch >= skip_epoch:
            total += (toc - tic)
            train_time += step
        if rank == 0:
            print("Epoch {}, rank {}, train time {}, epoch time{}".format(
                epoch, rank, step, toc-tic))
    monitor.stop()
    print("rank:", rank, " end2end time:",
          total/(args.num_epochs - skip_epoch))
    print("train: ", rank, train_time/(args.num_epochs - skip_epoch))
    for i in range(sampler_number):
        sample_workers[i].join()
    for i in range(loader_number):
        load_workers[i].join()
    monitor.join()
    if rank == 0:
        print("array len:", len(time_array), len(acc_array), len(loss_array))
        with open("pipeline_paper_curve.txt", 'w') as file:
            for i in range(len(time_array)):
                file.write(str(time_array[i]) + ' ')
                file.write(str(acc_array[i]) + ' ')
                file.write(str(loss_array[i]) + '\n')
            file.close()
    cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN')
    register_data_args(parser)
    parser.add_argument('--graph_name', default='test',
                        type=str, help='graph name')
    parser.add_argument('--part_config', default='/data/ds/metis_ogbn-papers100M4/ogbn-papers100M.json',
                        type=str, help='The path to the partition config file')
    parser.add_argument('--n_ranks', default=4,
                        type=int, help='Number of ranks')
    parser.add_argument('--batch_size', default=1024,
                        type=int, help='Batch size')
    parser.add_argument('--num_hidden', default=256,
                        type=int, help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--fan_out', default="5, 10, 15",
                        type=str, help='Fanout')
    parser.add_argument('--num_epochs', default=20, type=int, help='Epochs')
    parser.add_argument('--lr', type=float, default=0.003)
    parser.add_argument('--feat_mode', default='PartitionCache', type=str,
                        help='Feature cache mode. (AllCache, PartitionCache, ReplicateCache)')
    parser.add_argument('--cache_ratio', default=10, type=int,
                        help='Percentages of features on GPUs')
    parser.add_argument('--graph_cache_ratio', default=100,
                        type=int, help='Ratio of edges cached in the GPU')
    parser.add_argument('--in_feats', default=256, type=int,
                        help='In feature dimension used when the graph do not have feature')
    parser.add_argument('--graph_cache_gb', default=-1, type=int,
                        help='Memory used to cache graph topology. Setting it not equal to -1 disables graph_cache_ratio')
    parser.add_argument('--feat_cache_gb', default=-1, type=int,
                        help='Memory used to cache features, Setting it not equal to -1 disables cache_ratio')
    parser.add_argument('--log_every', default=1, type=int)
    args = parser.parse_args()

    mp.spawn(run,
             args=(args, ),
             nprocs=args.n_ranks,
             join=True)
