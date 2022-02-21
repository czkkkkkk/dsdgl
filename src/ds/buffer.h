#ifndef DS_BUFFER_H
#define DS_BUFFER_H

#include <atomic>
#include <queue>
#include <semaphore.h>

namespace dgl {
namespace ds {

class SpinLock {
public:
  SpinLock() : flag_(false) {}

  void Lock(){
    bool expect = false;
    while (!flag_.compare_exchange_weak(expect, true)) {
      expect = false;
    }
  }

  void Unlock() {
    flag_.store(false);
  }

private:
  std::atomic<bool> flag_;
};

template <typename T>
class ThreadSafeQueue {
public:
  ThreadSafeQueue() {}

  void Put(T item) {
    lock_.Lock();
    que_.push(item);
    lock_.Unlock();
  }

  T Get() {
    lock_.Lock();
    T item = que_.front();
    que_.pop();
    lock_.Unlock();
    return item;
  }

private:
  SpinLock lock_;
  std::queue<T> que_;

};

template <typename T>
class Buffer {
public:
  Buffer(int cap) {
    sem_init(&produce_, 0, cap);
    sem_init(&consume_, 0, 0);
  }

  void Put(T item) {
    sem_wait(&produce_);
    que_.Put(item);
    sem_post(&consume_);
  }

  T Get() {
    sem_wait(&consume_);
    T item = que_.Get();
    sem_post(&produce_);
    return item;
  }

private:
  ThreadSafeQueue<T> que_;
  sem_t produce_, consume_;

};

}
}

#endif