
from threading import Thread
import threading
import torch as th
import psutil
import os

from ..transform import to_block
from ..base import NID
from ..distributed import load_partition as dgl_load_partition
from . import set_thread_local_stream, set_queue_size, load_subtensor
from . import cache_feats, rebalance_train_nids, allgather_train_labels
from . import csr_to_global_id
from . import cache_graph
from .. import add_self_loop
from .pc_queue import MPMCQueue


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
            block = to_block(
                frontier, seeds, min_vids=self.device_min_vids)
            # Obtain the seed nodes for next layer.
            seeds = block.srcdata[NID]
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

    def __init__(self, dataloader, rank, num_epochs, thread_id, sampler_number, batches_per_sampler, loader_number):
        Thread.__init__(self)
        self.rank = rank
        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.thread_id = thread_id
        self.sampler_number = sampler_number
        self.batches_per_sampler = batches_per_sampler
        self.batches = batches_per_sampler[thread_id]
        self.total_batches = sum(batches_per_sampler)
        self.mpmc_queue = MPMCQueue(1, sampler_number, loader_number)

    def next_epoch(self):
        self.thread_id = (
            self.thread_id + self.total_batches) % self.sampler_number

    def run(self):
        th.cuda.set_device(self.rank)
        s = th.cuda.Stream(device=self.rank)
        set_thread_local_stream(s, self.thread_id, 0)
        with th.cuda.stream(s):
            for i in range(self.num_epochs):
                for j in range(self.batches):
                    set_queue_size(self.mpmc_queue.size())
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

    def __init__(self, labels, min_vids, in_mpmc_queue, rank, num_epochs, feat_dim, thread_id, sampler_number, loader_number, batches_per_loader):
        Thread.__init__(self)
        self.rank = rank
        self.labels = labels
        self.min_vids = min_vids
        self.in_mpmc_queue = in_mpmc_queue
        self.num_epochs = num_epochs
        self.feat_dim = feat_dim
        self.thread_id = thread_id
        self.sampler_number = sampler_number
        self.loader_number = loader_number
        self.batches_per_loader = batches_per_loader
        self.batches = batches_per_loader[thread_id]
        self.total_batches = sum(batches_per_loader)
        self.out_pc_queue = MPMCQueue(1, loader_number, 1)

    def next_epoch(self):
        self.thread_id = (
            self.thread_id + self.total_batches) % self.loader_number

    def run(self):
        th.cuda.set_device(self.rank)
        s = th.cuda.Stream(device=self.rank)
        set_thread_local_stream(
            s, self.thread_id + self.sampler_number, 1)
        with th.cuda.stream(s):
            for i in range(self.num_epochs):
                for i in range(self.batches):
                    sample_result = self.in_mpmc_queue.get(self.thread_id)
                    set_queue_size(self.out_pc_queue.size())
                    input_nodes = sample_result[0]
                    seeds = sample_result[1]
                    blocks = sample_result[2]
                    batch_inputs, batch_labels = load_subtensor(
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

def calculate_ratio(gb, size, entry_size_in_byte):
    n_cached = gb * 1024 * 1024 * 1024 / entry_size_in_byte
    ret = n_cached / size * 100
    return min(100., ret)


class Data(object):
    def __init__(self, part_config, rank, batch_size, args):
        # load partitioned graph
        g, node_feats, edge_feats, gpb, _, _, _ = dgl_load_partition(
            part_config, rank)
        process = psutil.Process(os.getpid())
        print('Host memory usage after load partition {} GB'.format(
            process.memory_info().rss / 1e9))
        g = add_self_loop(g)

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

        self.in_feats = node_feats['_N/features'].shape[1]

        print('Rank {}, Host memory usage before cache feats: {} GB'.format(
            rank, process.memory_info().rss / 1e9))
        cache_feats(args.feat_mode, g,
                        node_feats['_N/features'], args.cache_ratio)
        del node_feats['_N/features']
        print('Rank {}, Host memory usage after cache feats: {} GB'.format(
            rank, process.memory_info().rss / 1e9))

        self.num_vertices = int(gpb._max_node_ids[-1])

        print('Graph cache ratio {}, feature cache ratio {}'.format(
            args.graph_cache_ratio, args.cache_ratio))

        print('rank {}, # global: {}, # local: {}'.format(
            rank, self.num_vertices, n_local_nodes))
        print('# in feats:', self.in_feats)
        train_nid = th.masked_select(
            g.nodes()[:n_local_nodes], node_feats['_N/train_mask'])
        train_nid = rebalance_train_nids(
            train_nid, batch_size, g.ndata[NID])
        train_label = allgather_train_labels(node_feats['_N/labels'])
        self.n_classes = len(
            th.unique(train_label[th.logical_not(th.isnan(train_label))]))
        print('#labels:', self.n_classes)
        print('Rank {}, subgraph nodes {} B, subgraph edges {} B'.format(
            rank, g.number_of_nodes() / 1e9, g.number_of_edges() / 1e9))

        th.distributed.barrier()
        # tansfer graph and train nodes to gpu
        self.device = th.device('cuda:%d' % rank)
        self.train_nid = train_nid.to(self.device)
        # train_g = g.reverse().formats(['csr'])
        train_g = g.formats(['csr'])
        g = None
        self.train_g = csr_to_global_id(train_g, train_g.ndata[NID])
        self.global_nid_map = train_g.ndata[NID].to(self.device)
        cache_graph(train_g, args.graph_cache_ratio)
        train_g = None
        print('Rank {}, pytorch memory usage after move train_g to device : {} GB'.format(
            rank, th.cuda.memory_allocated(rank) / 1e9))
        self.train_label = train_label.to(self.device)
        # todo: transfer gpb to gpu
        self.min_vids = [0] + list(gpb._max_node_ids)
        self.min_vids = th.tensor(self.min_vids, dtype=th.int64).to(self.device)
        self.min_eids = [0] + list(gpb._max_edge_ids)
        self.min_eids = th.tensor(self.min_eids, dtype=th.int64).to(self.device)

def load_partition(part_config, rank, batch_size, args):
    return Data(part_config, rank, batch_size, args)
