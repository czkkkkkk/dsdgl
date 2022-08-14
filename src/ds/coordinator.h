#ifndef DGL_DS_COORDINATOR_H_
#define DGL_DS_COORDINATOR_H_

#include <memory>
#include <vector>
#include <dmlc/logging.h>
#include <chrono>
#include <thread>

#include "zmq.hpp"

#include "./base/bin_stream.h"

namespace dgl {
namespace ds {

enum CommEvent { Allgather = 0, Scatter = 1, Gather = 2, Broadcast = 3, RingExchange = 4, CommEnd = 5 };

struct ProcInfo {
  int pid;
  int dev_id;
  int rank;
  std::string hostname;

  bool SameDevice(const ProcInfo& rhs) {
    return hostname == rhs.hostname && dev_id == rhs.dev_id;
  }
  BinStream &serialize(BinStream &bs) const {
    bs << pid << dev_id << rank << hostname;
    return bs;
  }
  BinStream &deserialize(BinStream &bs) {
    bs >> pid >> dev_id >> rank >> hostname;
    return bs;
  }
};

class Coordinator {
 public:
  Coordinator(int rank, int workd_size, int port, const std::string& root_addr, const std::string& my_addr);

  int GetWorldSize() const { return n_peers_; }
  int GetRank() const { return rank_; }
  int GetDevId() const { return peer_infos_[rank_].dev_id; }
  bool IsRoot() const { return is_root_; }
  const std::string &GetHostname() const { return peer_infos_[rank_].hostname; }

  const std::vector<ProcInfo> &GetPeerInfos() const { return peer_infos_; }

  void SendIntTo(int peer_id, int val);
  void SendBinstreamTo(
      int peer_id,
      const std::shared_ptr<BinStream> &payload);  // -1 means send to root

  int RecvIntFromRoot();
  std::shared_ptr<BinStream> RecvBinstreamFromRoot();

  int RootRecvInt();                               // For root
  std::shared_ptr<BinStream> RootRecvBinStream();  // For root

  void Barrier();

  template <typename T>
  void Allgather(std::vector<T> &vec) {
    CHECK(vec.size() == n_peers_);
    auto bs = std::make_shared<BinStream>();
    *bs << CommEvent::Allgather << rank_ << vec[rank_];
    SendBinstreamTo(-1, bs);
    if (is_root_) {
      std::vector<T> root_vals(n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto recv_bs = RootRecvBinStream();
        CommEvent event;
        int rank;
        T val;
        *recv_bs >> event >> rank >> val;
        CHECK(event == CommEvent::Allgather);
        root_vals[rank] = val;
      }
      auto broadcast_bs = std::make_shared<BinStream>();
      *broadcast_bs << root_vals;
      for (int i = 0; i < n_peers_; ++i) {
        SendBinstreamTo(i, broadcast_bs);
      }
    }
    auto allinfos = RecvBinstreamFromRoot();
    *allinfos >> vec;
  }
  template <typename T>
  T Scatter(const std::vector<T> &vec) {
    if (is_root_) {
      CHECK(vec.size() == n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::Scatter << vec[i];
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    T ret;
    *my_msg >> e >> ret;
    CHECK(e == CommEvent::Scatter);
    Barrier();
    return ret;
  }
  template <typename T>
  std::vector<T> Gather(const T& val) {
    auto bs = std::make_shared<BinStream>();
    *bs << CommEvent::Gather << rank_ << val;
    SendBinstreamTo(-1, bs);
    std::vector<T> ret;
    if(IsRoot()) {
      ret.resize(n_peers_);
      for(int i = 0; i < n_peers_; ++i) {
        auto bs = RootRecvBinStream();
        CommEvent e;
        int rank;
        T v;
        *bs >> e >> rank >> v;
        CHECK(e == CommEvent::Gather);
        ret[rank] = v;
      }
    }
    Barrier();
    return ret;
  }

  template <typename T>
  std::vector<std::vector<T>> GatherLargeVector(const std::vector<T>& vec) {
    constexpr size_t per_block_size = 100000000;
    std::vector<std::vector<T>> ret(n_peers_);
    std::vector<size_t> sizes = Gather(vec.size());
    size_t round = 0;
    if(IsRoot()) {
      size_t max_size = 0;
      for(auto v: sizes) {
        max_size = std::max(max_size, v);
      }
      round = (max_size + per_block_size - 1) / per_block_size;
    }
    Broadcast(round);
    for(size_t i = 0; i < round; ++i) {
      size_t start = i * per_block_size;
      size_t end = std::min(start + per_block_size, vec.size());
      std::vector<T> to_send;
      if(start < end) {
        to_send = std::vector<T>(vec.begin() + start, vec.begin() + end);
      }
      auto recvs = Gather(to_send);
      if(IsRoot()) {
        for(int i = 0; i < n_peers_; ++i) {
          ret[i].insert(ret[i].end(), recvs[i].begin(), recvs[i].end());
        }
      }
    }
    return ret;
  }

  template <typename T>
  void Broadcast(T &val) {
    if (IsRoot()) {
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::Broadcast << val;
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    *my_msg >> e >> val;
    CHECK(e == CommEvent::Broadcast);
  }
  template <typename T>
  T RingExchange(int needed_peer_id, const T &val) {
    auto bs = std::make_shared<BinStream>();
    *bs << CommEvent::RingExchange << rank_ << needed_peer_id << val;
    SendBinstreamTo(-1, bs);
    if (IsRoot()) {
      std::vector<T> gathered_val(n_peers_);
      std::vector<int> send_ids(n_peers_);
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = RootRecvBinStream();
        CommEvent e;
        int peer_rank, needed_peer;
        T v;
        *bs >> e >> peer_rank >> needed_peer >> v;
        CHECK_EQ(e, CommEvent::RingExchange);
        send_ids[peer_rank] = needed_peer;
        gathered_val[peer_rank] = v;
      }
      for (int i = 0; i < n_peers_; ++i) {
        auto bs = std::make_shared<BinStream>();
        *bs << CommEvent::RingExchange << send_ids[i]
            << gathered_val[send_ids[i]];
        SendBinstreamTo(i, bs);
      }
    }
    auto my_msg = RecvBinstreamFromRoot();
    CommEvent e;
    int peer_id;
    T v;
    *my_msg >> e >> peer_id >> v;
    CHECK_EQ(peer_id, needed_peer_id);
    return v;
  }

 private:
  void _Initialize(int port, const std::string& root_addr, const std::string& my_addr);

  bool is_root_;
  int rank_;
  int n_peers_;
  std::unique_ptr<zmq::context_t> zmq_ctx_;
  std::vector<std::unique_ptr<zmq::socket_t>> to_peers_;  // For root
  std::unique_ptr<zmq::socket_t> root_receiver_;          // For root

  std::unique_ptr<zmq::socket_t> send_to_root_;    // For all
  std::unique_ptr<zmq::socket_t> recv_from_root_;  // For all
  std::string recv_addr_;                          // For all
  std::vector<ProcInfo> peer_infos_;
};

}
}

#endif