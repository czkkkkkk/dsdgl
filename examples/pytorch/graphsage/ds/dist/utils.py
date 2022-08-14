
import threading
from threading import Thread
from queue import Queue
import torch as th
import dgl
from pynvml import *
import time

class PCQueue(object):
  def __init__(self, capacity):
    self.capacity = threading.Semaphore(capacity)
    self.product = threading.Semaphore(0)
    self.buffer = Queue()
  
  def get(self):
    self.product.acquire()
    item = self.buffer.get()
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
  def __init__(self, dataloader, pc_queue, rank, num_epochs):
    Thread.__init__(self)
    self.rank = rank
    self.dataloader = dataloader
    self.pc_queue = pc_queue
    self.num_epochs = num_epochs
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    dgl.ds.set_thread_local_stream(s)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        for step, (input_nodes, seeds, blocks) in enumerate(self.dataloader):
          s.synchronize()
          self.pc_queue.put((input_nodes, seeds, blocks))
        self.pc_queue.stop_produce(1)


class SubtensorLoader(Thread):
  def __init__(self, features, labels, min_vids, in_pc_queue, out_pc_queue, rank, num_epochs):
    Thread.__init__(self)
    self.rank = rank
    self.features = features
    self.labels = labels
    self.min_vids = min_vids
    self.in_pc_queue = in_pc_queue
    self.out_pc_queue = out_pc_queue
    self.num_epochs = num_epochs
  
  def run(self):
    th.cuda.set_device(self.rank)
    s = th.cuda.Stream(device=self.rank)
    dgl.ds.set_thread_local_stream(s)
    with th.cuda.stream(s):
      for i in range(self.num_epochs):
        while True:
          sample_result = self.in_pc_queue.get()
          if sample_result is None:
            break
          input_nodes = sample_result[0]
          seeds = sample_result[1]
          blocks = sample_result[2]
          batch_inputs, batch_labels = dgl.ds.load_subtensor(self.features, self.labels, input_nodes, seeds, self.min_vids)
          s.synchronize()
          self.out_pc_queue.put((batch_inputs, batch_labels, blocks))
        self.out_pc_queue.stop_produce(1)
  
class GPUMonitor(Thread):
  def __init__(self, rank):
    Thread.__init__(self)
    nvmlInit()
    self.stop_signal = False
    self.samples = {}
    self.rank = rank
    self.token = 'default'
  
  def run(self):
    handle = nvmlDeviceGetHandleByIndex(self.rank)
    while not self.stop_signal:
      token = self.token
      if token not in self.samples:
        self.samples[token] = []
      self.samples[token].append(nvmlDeviceGetUtilizationRates(handle))
      time.sleep(0.1)
  
  def set_token(self, token):
    self.token = token

  def stop(self):
    self.stop_signal = True
    time.sleep(2)
    print('Rank {} reports GPU utilization')
    for token, samples in self.samples.items():
      util = 0
      memory = 0
      for info in samples:
        util += info.gpu
        memory = max(memory, info.memory)
      util /= len(samples)
      print('Rank {}, stage {}, sample size {}, avg GPU util {}, max_memory util {}'.format(self.rank, token, len(samples), util, memory))



