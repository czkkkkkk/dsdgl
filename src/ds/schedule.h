#ifndef DGL_DS_SCHEDULE_H
#define DGL_DS_SCHEDULE_H

#include <stdio.h>
#include <unistd.h>
#include "context.h"
#include "buffer.h"

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace ds {

enum CommToken {
  COMM_INIT,
  COMM_SAMPLE,
  COMM_LOAD,
  COMM_WAIT
};

class Scheduler {
public:
  Scheduler() : comm_tokens_(50), comm_requests_(50), cur_token_(COMM_INIT) {}

  void Schedule() {
    auto* ds_context = DSContext::Global();
    CommToken next_token, expect = COMM_INIT;
    while (true) {
      next_token = comm_tokens_.Get();
      while (!cur_token_.compare_exchange_weak(expect, next_token)) {
        expect = COMM_INIT;
      }
    }
  }

  void Coordinate() {
    auto* ds_context = DSContext::Global();
    int rank = ds_context->rank;
    CommToken next_token;
    while (true) {
      if (rank == 0) {
        next_token = comm_requests_.Get();
      }
      ds_context->comm_coordinator->Barrier();
      ds_context->comm_coordinator->Broadcast(next_token);
      comm_tokens_.Put(next_token);
    }
  }

  void TryComm(CommToken token) {
    auto* ds_context = DSContext::Global();
    CommToken expect = token;
    if (ds_context->rank == 0) {
      comm_requests_.Put(token);
    }
    while (!cur_token_.compare_exchange_weak(expect, COMM_WAIT)) {
      expect = token;
    }
  }

  void FinishComm() {
    auto* ds_context = DSContext::Global();
    CommToken expect = COMM_WAIT;
    bool flag = cur_token_.compare_exchange_weak(expect, COMM_INIT);
    CHECK_EQ(flag, true);
  }

  static Scheduler* Global() {
    static Scheduler instance;
    return &instance;
  }

private:
  Buffer<CommToken> comm_tokens_;
  Buffer<CommToken> comm_requests_;
  std::atomic<CommToken> cur_token_;

};

}
}

#endif