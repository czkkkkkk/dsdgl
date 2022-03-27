from __future__ import absolute_import

import scipy.sparse as sp
from scipy.sparse import coo_matrix
import numpy as np
import os

from dgl.data.dgl_dataset import DGLDataset, DGLBuiltinDataset
from dgl.data.utils import _get_dgl_url, generate_mask_tensor, load_graphs, save_graphs, deprecate_property
import dgl.backend as F
from dgl.convert import from_scipy
from dgl.transforms import reorder_graph

class TwitterDataset(DGLDataset):
  raw_dir = '/data/konect/'
  url = '/data/konect/twitter/out.twitter'

  def __init__(self, raw_dir=None, force_reload=False,
               verbose=False, transform=None):
    self.num_classes = 3
    super(TwitterDataset, self).__init__(name='twitter',
                                         url=TwitterDataset.url,
                                         raw_dir=TwitterDataset.raw_dir,
                                         force_reload=force_reload,
                                         verbose=verbose,
                                         transform=transform)

  def get_rand_type(self):
    val = np.random.uniform()
    if val < 0.1:
      return 0
    elif val < 0.4:
      return 1
    return 2

  def process(self):
    row = []
    col = []
    num_nodes = 0
    with open(TwitterDataset.url, 'r') as f:
      line = f.readline()
      num_nodes = 0
      num_edges = 0
      cur_node = 0
      for line in f:
        id1, id2 = line.split()
        id1 = int(id1)
        id2 = int(id2)
        if id1 > num_nodes:
          num_nodes = id1
        if id2 > num_nodes:
          num_nodes = id2
        row.append(id1)
        col.append(id2)
        num_edges += 1
      num_nodes += 1
      print('number nodes:', num_nodes)
      print('number edges:', num_edges)
    row = np.array(row)
    col = np.array(col)
    node_types = []
    for i in range(0, num_nodes):
      node_types.append(self.get_rand_type())
    node_types = np.array(node_types)
    coo = coo_matrix((np.zeros_like(row), (row, col)), shape=(num_nodes, num_nodes))
    graph = from_scipy(coo)
    print("finish construct graph")
    graph = dgl.to_bidirected(graph)
    print("finish to_bidirected graph")
    # graph = dgl.to_simple(graph)
    # print("finish to_simple graph")
    graph.ndata['node_type'] = F.tensor(node_types, dtype=F.data_type_dict['int32'])
    # features = np.random.rand(num_nodes, 128)
    # labels = np.random.randint(0, self.num_classes, size=num_nodes)
    # train_mask = (node_types == 0)
    # val_mask = (node_types == 1)
    # test_mask = (node_types == 2)
    # graph.ndata['train_mask'] = generate_mask_tensor(train_mask)
    # graph.ndata['val_mask'] = generate_mask_tensor(val_mask)
    # graph.ndata['test_mask'] = generate_mask_tensor(test_mask)
    # graph.ndata['feat'] = F.tensor(features, dtype=F.data_type_dict['float32'])
    # graph.ndata['label'] = F.tensor(labels, dtype=F.data_type_dict['int64'])
    # graph = reorder_graph(graph, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)
    self._graph = graph

  def has_cache(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    print("check cache", graph_path)
    if os.path.exists(graph_path):
      print("using cached data")
      return True
    return False
  
  def save(self):
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    save_graphs(graph_path, self._graph)

  def load(self):
    print("loading graph")
    graph_path = os.path.join(self.save_path, 'dgl_graph.bin')
    graphs, _ = load_graphs(graph_path)
    print("finish loading graph")
    self._graph = graphs[0]
    # self._graph.ndata['train_mask'] = generate_mask_tensor(self._graph.ndata['train_mask'].numpy())
    # self._graph.ndata['val_mask'] = generate_mask_tensor(self._graph.ndata['val_mask'].numpy())
    # self._graph.ndata['test_mask'] = generate_mask_tensor(self._graph.ndata['test_mask'].numpy())

  def __getitem__(self, idx):
    assert idx == 0, "Reddit Dataset only has one graph"
    return self._graph

  def __len__(self):
    return 1

import dgl
if __name__ == '__main__':
  dataset = TwitterDataset()
  graph = dataset[0]
  print(graph)