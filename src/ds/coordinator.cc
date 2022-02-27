#include "./coordinator.h"

#include <unistd.h>
#include <zmq.hpp>

#include <dmlc/logging.h>

#include "base/zmq_helpers.h"
#include "utils.h"

namespace dgl {
namespace ds {


Coordinator::Coordinator(int rank, int world_size, int port) {
  rank_ = rank;
  n_peers_ = world_size;
  is_root_ = rank == 0? true: false;
  zmq_ctx_ = std::unique_ptr<zmq::context_t>(new zmq::context_t());
  LOG(INFO) << "Coordinator initializing port: " << port;
  _Initialize(port);
}

void Coordinator::_Initialize(int mport) {
  // 1. Each rank setups the socket receving msg from the root
  int port = GetAvailablePort();
  recv_addr_ =
      std::string("tcp://") + GetHostName() + ":" + std::to_string(port);
  auto recv_bind_addr = std::string("tcp://*:") + std::to_string(port);
  recv_from_root_ = std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PULL));
  recv_from_root_->bind(recv_bind_addr);


  // 2. Root setup the socket receving msg from all ranks
  // NOTE: currently assume all ranks are on the same machine
  int master_port = mport;
  auto root_bind_addr = std::string("tcp://*:") + std::to_string(master_port);
  auto root_conn_addr = std::string("tcp://") + GetHostName() + ":" + std::to_string(master_port);
  if (is_root_) {
    root_receiver_ = std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PULL));
    root_receiver_->bind(root_bind_addr);
  }

  // 3. Each rank connects to the root
  send_to_root_ = std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PUSH));
  send_to_root_->connect(root_conn_addr);
  LOG(INFO) << "Rank " << rank_ << " try to connect to root on addr "
            << recv_addr_;
  int pid = getpid();
  zmq_sendmore_int32(send_to_root_.get(), rank_);
  zmq_sendmore_string(send_to_root_.get(), GetHostName());
  zmq_sendmore_int32(send_to_root_.get(), pid);
  zmq_send_string(send_to_root_.get(), recv_addr_);

  // 4. Root builds up peer informations and broadcase them
  if (IsRoot()) {
    peer_infos_.resize(n_peers_);

    for (int i = 0; i < n_peers_; ++i) {
      to_peers_.emplace_back(nullptr);
    }
    std::map<std::string, int> hosts_cnt;
    for (int i = 0; i < n_peers_; ++i) {
      int peer_rank = zmq_recv_int32(root_receiver_.get());
      std::string peer_hostname = zmq_recv_string(root_receiver_.get());
      int pid = zmq_recv_int32(root_receiver_.get());
      std::string peer_addr = zmq_recv_string(root_receiver_.get());
      LOG(INFO) << "Get the info of peer " + std::to_string(peer_rank) + " on address "
                + peer_addr;
      CHECK_LT(peer_rank, n_peers_);

      to_peers_[peer_rank] = std::unique_ptr<zmq::socket_t>(new zmq::socket_t(*zmq_ctx_, ZMQ_PUSH));
      to_peers_[peer_rank]->connect(peer_addr);

      hosts_cnt[peer_hostname] = 0;
      peer_infos_[peer_rank].pid = pid;
      peer_infos_[peer_rank].hostname = peer_hostname;
      peer_infos_[peer_rank].rank = peer_rank;
    }
    for (int i = 0; i < n_peers_; ++i) {
      const std::string &hostname = peer_infos_[i].hostname;
      peer_infos_[i].dev_id = hosts_cnt[hostname]++;
    }
  }
  Broadcast(peer_infos_);
  LOG(INFO) << "My rank is " + std::to_string(rank_) + ", my device id is " + std::to_string(peer_infos_[rank_].dev_id);
}

void Coordinator::SendIntTo(int peer_id, int val) {
  zmq::socket_t *send_socket;
  if (peer_id == -1) {
    send_socket = send_to_root_.get();
  } else {
    CHECK(is_root_);  // Only root is allowed to send to other peers
    send_socket = to_peers_[peer_id].get();
  }
  zmq_send_int32(send_socket, val);
}

void Coordinator::SendBinstreamTo(int peer_id,
                                  const std::shared_ptr<BinStream> &payload) {
  zmq::socket_t *send_socket;
  if (peer_id == -1) {
    send_socket = send_to_root_.get();
  } else {
    CHECK(is_root_);  // Only root is allowed to send to other peers
    send_socket = to_peers_[peer_id].get();
  }
  zmq_send_binstream(send_socket, *payload);
}

int Coordinator::RecvIntFromRoot() {
  return zmq_recv_int32(recv_from_root_.get());
}

std::shared_ptr<BinStream> Coordinator::RecvBinstreamFromRoot() {
  auto ret =
      std::make_shared<BinStream>(zmq_recv_binstream(recv_from_root_.get()));
  return ret;
}

int Coordinator::RootRecvInt() { return zmq_recv_int32(root_receiver_.get()); }

std::shared_ptr<BinStream> Coordinator::RootRecvBinStream() {
  return std::make_shared<BinStream>(zmq_recv_binstream(root_receiver_.get()));
}

void Coordinator::Barrier() {
  if (IsRoot()) {
    for (int i = 0; i < n_peers_; ++i) {
      SendIntTo(i, 1);
    }
  }
  int val = RecvIntFromRoot();
  DCHECK_EQ(val, 1);
  SendIntTo(-1, 2);
  if (IsRoot()) {
    for (int i = 0; i < n_peers_; ++i) {
      int val = RootRecvInt();
      DCHECK_EQ(val, 2);
    }
    for (int i = 0; i < n_peers_; ++i) {
      SendIntTo(i, 3);
    }
  }
  val = RecvIntFromRoot();
  DCHECK_EQ(val, 3);
}

}
}