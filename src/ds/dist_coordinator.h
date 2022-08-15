#ifndef DGL_DS_DIST_COORDINATOR_H_
#define DGL_DS_DIST_COORDINATOR_H_

#include "zmq.hpp"

#include "./coordinator.h"
#include "base/zmq_helpers.h"

namespace dgl {
namespace ds {

enum DistCommEvent { kPeerExchange };

class DistCoordinator {
 public:
  DistCoordinator(int rank, int world_size, int global_rank,
                  int global_world_size, int node_rank, Coordinator* coor);
  template <typename T>
  T PeerExchange(const T& val) {
    auto bs = std::make_shared<BinStream>();
    *bs << DistCommEvent::kPeerExchange << global_rank_ << peer_global_rank_
        << val;
    zmq_send_binstream(sender_.get(), *bs);
    auto recved =
        std::make_shared<BinStream>(zmq_recv_binstream(recver_.get()));

    DistCommEvent e;
    int peer_global_rank, my_global_rank;
    *recved >> e >> peer_global_rank >> my_global_rank;
    CHECK_EQ(e, DistCommEvent::kPeerExchange);
    CHECK_EQ(peer_global_rank, peer_global_rank_);
    CHECK_EQ(my_global_rank, global_rank_);
    T ret;
    *recved >> ret;
    return ret;
  }

 private:
  int rank_, world_size_, global_rank_, global_world_size_;
  int peer_global_rank_;
  int node_rank_;

  std::unique_ptr<zmq::context_t> zmq_ctx_;
  std::unique_ptr<zmq::socket_t> sender_, recver_;
};

}  // namespace ds
}  // namespace dgl

#endif