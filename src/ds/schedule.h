#ifndef DGL_DS_SCHEDULE_H
#define DGL_DS_SCHEDULE_H

#include <dgl/runtime/container.h>
#include <dgl/packed_func_ext.h>
#include <stdio.h>
#include <stdint.h>
#include <unistd.h>

#include "context.h"
#include "utils.h"
#include "buffer.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {

class Scheduler {
  static constexpr int STOP_SIGNAL = -2;
public:
  // use stream_t value as token
  Scheduler() : comm_tokens_(50), comm_requests_(50), cur_token_(-1), alive_(true) {}
  ~Scheduler() {
    auto* ds_context = DSContext::Global();
    int rank = ds_context->rank;
    if(rank == 0) {
      comm_requests_.Put(STOP_SIGNAL);
    }
    while(alive_);
  }

  void Schedule() {
    auto* ds_context = DSContext::Global();
    int next_token, expect = -1;
    while (true) {
      next_token = comm_tokens_.Get();
      if (next_token == STOP_SIGNAL) {
        break;
      }
      while (!cur_token_.compare_exchange_weak(expect, next_token)) {
        expect = -1;
      }
    }
    alive_ = false;
  }

  void Coordinate() {
    auto* ds_context = DSContext::Global();
    int rank = ds_context->rank;
    int next_token;
    while (true) {
      if (rank == 0) {
        next_token = comm_requests_.Get();
      }
      ds_context->comm_coordinator->Barrier();
      ds_context->comm_coordinator->Broadcast(next_token);
      comm_tokens_.Put(next_token);
      if(next_token == STOP_SIGNAL) {
        break;
      }
    }
  }

  void TryComm(int token) {
    auto* ds_context = DSContext::Global();
    if(!ds_context->enable_comm_control || ds_context->world_size == 1) {
      return;
    }
    if (ds_context->rank == 0) {
      comm_requests_.Put(token);
    }
    //wait for my turn to communicate
    while (cur_token_.load() != token);
  }

  void FinishComm() {
    auto* ds_context = DSContext::Global();
    if(!ds_context->enable_comm_control || ds_context->world_size == 1) {
      return;
    }
    cur_token_.store(-1);
  }

  static Scheduler* Global() {
    static Scheduler instance;
    return &instance;
  }

private:
  Buffer<int> comm_tokens_;
  Buffer<int> comm_requests_;
  std::atomic<int> cur_token_, alive_;

};

}
}

#endif