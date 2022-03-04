import threading
from queue import Queue

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
    print("training data remain:", self.buffer.qsize())
    return item
  
  def put(self, item):
    self.capacity.acquire()
    self.buffer.put(item)
    self.product.release()
  
  def stop_produce(self, num):
    for i in range(num):
      self.put(None)

def divide(capacity, number):
  divide_capacity = []
  for i in range(number):
    divide_capacity.append(capacity // number)
  for i in range(capacity % number):
    divide_capacity[i] += 1
  return divide_capacity

class MPMCQueue_simple(object):
  def __init__(self, capacity, p_number, c_number):
    self.capacity = capacity
    self.p_number = p_number
    self.c_number = c_number
    divide_capacity = divide(capacity, p_number)
    self.capacity_per_producer = []
    for i in range(p_number):
      self.capacity_per_producer.append(threading.Semaphore(divide_capacity[i]))
    self.capacity_per_consumer = []
    for i in range(c_number):
      self.capacity_per_consumer.append(threading.Semaphore(0))
    self.buffer = Queue()
  
  def put(self, item, p_id):
    self.capacity_per_producer[p_id].acquire()
    self.buffer.put(item)
    self.capacity_per_consumer[p_id].release()
  
  def get(self, c_id):
    self.capacity_per_consumer[c_id].acquire()
    item = self.buffer.get()
    self.capacity_per_producer[c_id].release()
    return item

# a pc queue for multiple producer and consumer
# every producer and consumer has logically seperate queue
class MPMCQueue(object):
  def __init__(self, capacity, p_number, c_number):
    self.capacity = capacity
    self.p_number = p_number
    self.c_number = c_number
    divide_capacity = divide(capacity, p_number)
    self.capacity_per_producer = []
    for i in range(p_number):
      self.capacity_per_producer.append(threading.Semaphore(divide_capacity[i]))
    self.capacity_per_consumer = []
    for i in range(c_number):
      self.capacity_per_consumer.append(threading.Semaphore(0))
    self.buffer = Queue()
    self.p_seq = capacity
    self.c_seq = 0
    self.p_lock = threading.Lock()
    self.c_lock = threading.Lock()
  
  def get_p_seq(self):
    self.p_lock.acquire()
    p_seq = self.p_seq
    self.p_seq += 1
    self.p_lock.release()
    return p_seq
  
  def get_c_seq(self):
    self.c_lock.acquire()
    c_seq = self.c_seq
    self.c_seq += 1
    self.c_lock.release()
    return c_seq
  
  def put(self, item, p_id):
    self.capacity_per_producer[p_id].acquire()
    self.buffer.put(item)
    seq = self.get_c_seq()
    self.capacity_per_consumer[seq % self.c_number].release()
  
  def get(self, c_id):
    self.capacity_per_consumer[c_id].acquire()
    item = self.buffer.get()
    seq = self.get_p_seq()
    self.capacity_per_producer[seq % self.p_number].release()
    return item
  
  def size(self):
    print("training data remain:", self.buffer.qsize())
