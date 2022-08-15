#include "dist_coordinator.h"

#include "base/zmq_helpers.h"
#include "utils.h"

namespace dgl {
namespace ds {

DistCoordinator::DistCoordinator(int rank, int world_size, int global_rank,
                                 int global_world_size, int node_rank,
                                 Coordinator* coor) {
  // get bind address;
  // send bind address to corresponding peer
  // connect
  rank_ = rank;
  world_size_ = world_size;
  global_rank_ = global_rank;
  global_world_size_ = global_world_size;
  node_rank_ = node_rank;
  peer_global_rank_ = (global_rank + world_size) % global_world_size;
  zmq_ctx_ = std::unique_ptr<zmq::context_t>(new zmq::context_t());
  auto my_addr = GetEnvParam("MY_ADDR", std::string(""));
  int port = GetAvailablePort();
  auto recv_bind_addr = std::string("tcp://*:") + std::to_string(port);
  recver_ =
      std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PULL));
  recver_->bind(recv_bind_addr);

  auto recv_addr = std::string("tcp://") + my_addr + ":" + std::to_string(port);
  auto peer_addr = coor->RingExchange(peer_global_rank_, recv_addr);
  sender_ =
      std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PUSH));
  sender_->connect(peer_addr);
}

}  // namespace ds
}  // namespace dgl