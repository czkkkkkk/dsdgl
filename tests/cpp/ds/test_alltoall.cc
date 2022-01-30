#include "ds/cuda/alltoall.h"

#include <gtest/gtest.h>
#include <cstdlib>
#include "mpi.h"

#include "ds/utils.h"
#include "./test_utils.h"
#include "ds/context.h"

using namespace dgl::ds;

template<typename T>
using Vec3d = std::vector<std::vector<std::vector<T>>>;
template<typename T>
using Vec2d = std::vector<std::vector<T>>;

template<typename T>
std::vector<T> RandVec(int size) {
  std::vector<T> ret(size);
  for(int i = 0; i < size; ++i) {
    ret[i] = rand() % 100;
  }
  return ret;
}

template<typename T>
std::pair<std::vector<T>, std::vector<int64_t>> Flatten(const std::vector<std::vector<T>>& vecs) {
  std::vector<T> ret;
  std::vector<int64_t> offset(1, 0);
  for(const auto& vec: vecs) {
    for(auto v: vec) {
      ret.push_back(v);
    }
    offset.push_back(offset.back() + vec.size());
  }
  return {ret, offset};
}

template<typename T>
Vec3d<T> _GetExpOutput(int rank, int world_size, const Vec3d<T>& input_all) {
  Vec3d<T> ret(world_size);
  for(int i = 0; i < world_size; ++i) {
    ret[i].resize(world_size);
  }
  for(int i = 0; i < world_size; ++i) {
    for(int j = 0; j < world_size; ++j) {
      ret[i][j] = input_all[j][i];
    }
  }
  return ret;
}

template<typename T>
void _TestAlltoall(int rank, int world_size, const Vec3d<T>& input_all) {
  auto exp_output_all = _GetExpOutput(rank, world_size, input_all);
  auto input = input_all[rank];
  auto exp_output = exp_output_all[rank];
  std::vector<T> sendbuff;
  std::vector<int64_t> send_offset;
  std::tie(sendbuff, send_offset) = Flatten(input);
  std::vector<T> exp_recvbuff;
  std::vector<int64_t> exp_recv_offset;
  std::tie(exp_recvbuff, exp_recv_offset) = Flatten(exp_output);
  T* dev_sendbuff;
  int64_t *dev_send_offset;
  T* dev_recvbuff;
  int64_t *dev_recv_offset;
  DSCudaMalloc(&dev_recvbuff, exp_recv_offset[world_size]);
  DSCudaMalloc(&dev_recv_offset, world_size + 1);
  DSMallocAndCopy(&dev_sendbuff, sendbuff);
  DSMallocAndCopy(&dev_send_offset, send_offset);
  auto* ds_context = DSContext::Global();
  CHECK(ds_context->initialized);
  Alltoall(dev_sendbuff, dev_send_offset, dev_recvbuff, dev_recv_offset, sizeof(T), &ds_context->comm_info, rank, world_size);
  auto recv_offset = DSDeviceVecToHost(dev_recv_offset, world_size + 1);
  CheckVectorEq(recv_offset, exp_recv_offset);
  auto recvbuff = DSDeviceVecToHost(dev_recvbuff, exp_recv_offset[world_size]);
  CheckVectorEq(recvbuff, exp_recvbuff);
}

TEST(DSSampling, Alltoall) {
  using IdType = int64_t;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
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

template<typename T>
void _AlltoallBenchmark(int rank, int world_size, int size) {
  std::vector<T> sendbuff(world_size * size, rank);
  std::vector<int64_t> send_offset(world_size + 1, 0);
  for(int i = 1; i <= world_size; ++i) {
    send_offset[i] = i * size;
  }


}
TEST(DSSampling, AlltoallBenchmark) {
  using IdType = int;
  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  srand(110);
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