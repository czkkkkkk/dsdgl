#include "ds/cuda/alltoall.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include <chrono>
#include <thread>
#include "mpi.h"

#include "ds/utils.h"
#include "./test_utils.h"
#include "ds/context.h"
#include "runtime/cuda/cuda_common.h"

using namespace dgl::ds;
using namespace dgl;
using namespace dgl::aten;
using namespace dgl::runtime;

template<typename T>
using Vec3d = std::vector<std::vector<std::vector<T>>>;
template<typename T>
using Vec2d = std::vector<std::vector<T>>;


template<typename T>
std::vector<T> RandVec(int size) {
  std::vector<T> ret(size);
  for(int i = 0; i < size; ++i) {
    ret[i] = rand() % 100 + 1;
  }
  return ret;
}

template<typename T>
std::pair<std::vector<T>, std::vector<int64_t>> Flatten(const std::vector<std::vector<T>>& vecs, int expand_size) {
  std::vector<T> ret;
  std::vector<int64_t> offset(1, 0);
  for(const auto& vec: vecs) {
    for(auto v: vec) {
      ret.push_back(v);
    }
    CHECK(vec.size() % expand_size == 0);
    offset.push_back(offset.back() + vec.size() / expand_size);
  }
  return {ret, offset};
}

template<typename T>
Vec3d<T> _GetExpOutput(int rank, int world_size, const Vec3d<T>& input_all) {
  Vec3d<T> ret(world_size, Vec2d<T>(world_size, std::vector<T>()));
  for(int i = 0; i < world_size; ++i) {
    for(int j = 0; j < world_size; ++j) {
      ret[i][j] = input_all[j][i];
    }
  }
  return ret;
}

template<typename T>
Vec3d<T> Expand(const Vec3d<T>& input_all, int expand_size) {
  Vec3d<T> ret(input_all.size(), Vec2d<T>(input_all.size(), std::vector<T>()));
  for(int i = 0; i < input_all.size(); ++i) {
    for(int j = 0; j < input_all[i].size(); ++j) {
      for(auto v: input_all[i][j]) {
        for(int t = 0; t < expand_size; ++t) {
          ret[i][j].push_back(v);

        }
      }
    }
  }
  return ret;

}

template<typename T>
void _TestAlltoall(int rank, int world_size, const Vec3d<T>& input_all, int expand_size=1) {
  auto expand_input = Expand(input_all, expand_size);
  auto exp_output_all = _GetExpOutput(rank, world_size, expand_input);
  auto input = expand_input[rank];
  auto exp_output = exp_output_all[rank];
  std::vector<T> host_sendbuff;
  std::vector<int64_t> host_send_offset;
  std::tie(host_sendbuff, host_send_offset) = Flatten(input, expand_size);
  std::vector<T> exp_recvbuff;
  std::vector<int64_t> exp_recv_offset;
  std::tie(exp_recvbuff, exp_recv_offset) = Flatten(exp_output, expand_size);

  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  IdArray sendbuff = IdArray::FromVector(host_sendbuff).CopyTo({kDLGPU, rank}, stream);
  IdArray send_offset = IdArray::FromVector(host_send_offset).CopyTo({kDLGPU, rank}, stream);
  IdArray recvbuff, recv_offset;
  std::tie(recvbuff, recv_offset) = Alltoall(sendbuff, send_offset, expand_size, rank, world_size, DSContext::Global()->nccl_comm);
  CheckVectorEq(recv_offset.ToVector<int64_t>(), exp_recv_offset);
  CheckVectorEq(recvbuff.ToVector<T>(), exp_recvbuff);
}

TEST(DSSampling, Alltoall64bits) {
  using IdType = int64_t;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
  SetEnvParam("USE_NCCL", 0);
  if(world_size == 2) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}}, {{1}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
  }
  if(world_size == 3) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}, {3, 8}}, {{1}, {100}, {100}}, {{1}, {100}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
  }
  Vec3d<IdType> input_all;
  input_all.resize(world_size, Vec2d<IdType>(world_size));
  for(int i = 0; i < world_size; ++i) {
    for(int j = 0; j < world_size; ++j) {
      int size = GetEnvParam("ALLTOALL_BENCHMARK_SIZE", 512);
      input_all[i][j] = RandVec<IdType>(size);
    }
  }
  _TestAlltoall(rank, world_size, input_all);
  _TestAlltoall(rank, world_size, input_all, 602);
}

TEST(DSSampling, AlltoallNCCL) {
  using IdType = int;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
  SetEnvParam("USE_NCCL", 1);
  if(world_size == 2) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}}, {{1}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
    input_all.resize(world_size, Vec2d<IdType>(world_size));
    for(int i = 0; i < world_size; ++i) {
      for(int j = 0; j < world_size; ++j) {
        int size = rand() % 10000;
        input_all[i][j] = RandVec<IdType>(size);
      }
    }
    _TestAlltoall(rank, world_size, input_all);
  }
  if(world_size == 3) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}, {3, 8}}, {{1}, {100}, {100}}, {{1}, {100}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
  }
}

TEST(DSSampling, Alltoall32bits) {
  using IdType = int;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
  SetEnvParam("USE_NCCL", 0);
  if(world_size == 2) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}}, {{1}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
    _TestAlltoall(rank, world_size, input_all, 602);
  }
  if(world_size == 3) {
    Vec3d<IdType> input_all = {{{1, 3}, {3, 5}, {3, 8}}, {{1}, {100}, {100}}, {{1}, {100}, {100}}};
    _TestAlltoall(rank, world_size, input_all);
  }
  Vec3d<IdType> input_all;
  input_all.resize(world_size, Vec2d<IdType>(world_size));
  for(int i = 0; i < world_size; ++i) {
    for(int j = 0; j < world_size; ++j) {
      int size = rand() % 100000;
      input_all[i][j] = RandVec<IdType>(size);
    }
  }
  _TestAlltoall(rank, world_size, input_all);
  _TestAlltoall(rank, world_size, input_all, 602);
}

template<typename T>
void _AlltoallBenchmark(int rank, int world_size, int size, int expand_size=1) {
  auto stream = CUDAThreadEntry::ThreadLocal()->stream;
  std::vector<int64_t> send_offset(world_size + 1, 0);
  for(int i = 1; i <= world_size; ++i) {
    send_offset[i] = i * size;
  }
  std::vector<T> sendbuff(send_offset[world_size] * expand_size, rank);
  auto context = DLContext({kDLGPU, rank});
  auto dgl_sendbuff = IdArray::FromVector<T>(sendbuff).CopyTo(context, stream);
  auto dgl_send_offset = IdArray::FromVector<int64_t>(send_offset).CopyTo(context, stream);
  IdArray recvbuff, recv_offset;
  // warmup
  for(int i = 0; i < 5; ++i) {
    std::tie(recvbuff, recv_offset) = Alltoall(dgl_sendbuff, dgl_send_offset, expand_size, rank, world_size, DSContext::Global()->nccl_comm);
  }
  CUDACHECK(cudaDeviceSynchronize());
  int num_iters = 20;
  auto start_ts = std::chrono::high_resolution_clock::now();
  for(int i = 0; i < num_iters; ++i) {
    std::tie(recvbuff, recv_offset) = Alltoall(dgl_sendbuff, dgl_send_offset, expand_size, rank, world_size, DSContext::Global()->nccl_comm);
  }
  CUDACHECK(cudaDeviceSynchronize());
  auto end_ts = std::chrono::high_resolution_clock::now();
  auto bytes = size * world_size * sizeof(T) / 1e9;
  auto time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_ts - start_ts).count();
  bool use_nccl = GetEnvParam("USE_NCCL", 0);
  LOG(INFO) << "Alltoall(NCCL?: " << use_nccl << ") " << "Benchmark: #send size of each GPU: " << bytes << " GB, # element size: " << sizeof(T) << " bytes, # alltoall time: " << time_in_ms / num_iters << " ms, # speeds: " << bytes / time_in_ms * 1000 * num_iters << " GB/s";
}
TEST(DSSampling, AlltoallBenchmark) {
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
  int size = GetEnvParam("ALLTOALL_BENCHMARK_SIZE", 100000);
  _AlltoallBenchmark<int>(rank, world_size, size);
  _AlltoallBenchmark<int64_t>(rank, world_size, size);
}