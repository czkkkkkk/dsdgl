#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include "dmlc/logging.h"

#include "ds/cuda/ds_kernel.h"

using IdType = unsigned long long int;

template<typename T>
std::string VecToString(const std::vector<T>& vec) {
  std::string ret = "[";
  for(int i = 0; i < vec.size(); ++i) {
    if(i > 0) ret += ", ";
    ret += std::to_string(vec[i]);
  }
  ret += "]";
  return ret;
}
IdType* ToDevice(const std::vector<IdType>& vec) {
  IdType* ret;
  cudaMalloc(&ret, sizeof(IdType) * vec.size());
  cudaMemcpy(ret, vec.data(), sizeof(IdType) * vec.size(), cudaMemcpyHostToDevice);
  return ret;
}
std::vector<IdType> FromDevice(IdType* ptr, size_t size) {
  std::vector<IdType> ret(size);
  cudaMemcpy(ret.data(), ptr, sizeof(IdType) * size, cudaMemcpyDeviceToHost);
  return ret;
}

template<typename T>
void CheckVectorEq(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for(int i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(lhs[i], rhs[i]);
  }
}

void _TestCluster(int world_size, const std::vector<IdType>& vid_base, const std::vector<IdType>& seeds) {
  cudaSetDevice(0);
  int n_seeds = seeds.size();
  // Calculate the expectations
  auto exp_seeds = seeds;
  std::sort(exp_seeds.begin(), exp_seeds.end());
  std::vector<IdType> exp_send_offset(world_size + 1), exp_send_sizes(world_size);
  for(int rank = 0, sptr = 0; rank < world_size; ++rank) {
    exp_send_offset[rank] = sptr;
    while (sptr < n_seeds && (rank == world_size - 1 || exp_seeds[sptr] < vid_base[rank + 1])) {
      ++sptr;
    }
    exp_send_sizes[rank] = sptr - exp_send_offset[rank];
  }
  exp_send_offset[world_size] = n_seeds;

  auto* device_vid_base = ToDevice(vid_base);
  auto* device_seeds = ToDevice(seeds);
  
  std::vector<IdType> send_offset(world_size + 1), send_sizes(world_size);
  IdType* device_send_sizes;
  Cluster(world_size, device_vid_base, n_seeds, device_seeds, send_offset.data(), send_sizes.data(), &device_send_sizes);
  auto real_seeds = FromDevice(device_seeds, n_seeds);
  CheckVectorEq(real_seeds, exp_seeds);
  CheckVectorEq(send_offset, exp_send_offset);
  CheckVectorEq(send_sizes, exp_send_sizes);
  auto real_device_send_sizes = FromDevice(device_send_sizes, world_size);
  CheckVectorEq(real_device_send_sizes, exp_send_sizes);
  cudaFree(device_vid_base);
  cudaFree(device_seeds);
  cudaFree(device_send_sizes);
}

TEST(Sampling, Cluster){
  _TestCluster(2, {0, 3}, {3, 0, 4});
  _TestCluster(3, {0, 3, 8}, {5, 3, 0, 4, 7});
  _TestCluster(4, {0, 4, 8, 8}, {1});
  _TestCluster(4, {0, 4, 8, 8}, {1, 8});
};