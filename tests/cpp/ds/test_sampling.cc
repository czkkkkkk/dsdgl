#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <nccl.h>
#include <thread>
#include <numeric>

#include "ds/cuda/ds_kernel.h"
#include "dmlc/logging.h"

using namespace dgl::ds;

using IdType = unsigned long long int;

/*
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

struct ShuffleArgs {
  std::vector<IdType> send_offset, send_sizes, seeds;
};
struct ShuffleOutput {
  int n_frontier;
  std::vector<IdType> frontier, recv_offset; 
};

void _TestShuffleThread(ShuffleArgs args, ShuffleOutput exp_output, int rank, int world_size, ncclComm_t comm) {
  cudaSetDevice(rank);
  // Prepare input
  auto* dev_send_offset = ToDevice(args.send_offset);
  auto* dev_send_sizes = ToDevice(args.send_sizes);
  auto* dev_seeds = ToDevice(args.seeds);
  
  // Prepare output
  IdType n_frontier;
  std::vector<IdType> recv_offset(world_size + 1);
  IdType* frontier;
  Shuffle(world_size, args.send_offset.data(), args.send_sizes.data(), dev_send_sizes, dev_seeds, n_frontier, recv_offset.data(), &frontier, rank, comm);
  EXPECT_EQ(n_frontier, exp_output.n_frontier);
  CheckVectorEq(FromDevice(frontier, exp_output.frontier.size()), exp_output.frontier);
  CheckVectorEq(recv_offset, exp_output.recv_offset);

  cudaFree(dev_send_offset);
  cudaFree(dev_send_sizes);
  cudaFree(dev_seeds);
}

void _TestShuffle(int world_size, const std::vector<ShuffleArgs>& inputs, const std::vector<ShuffleOutput>& exp_outputs) {
  std::vector<ncclComm_t> comms(world_size);
  std::vector<int> devs(world_size);
  std::iota(devs.begin(), devs.end(), 0);
  NCCLCHECK(ncclCommInitAll(comms.data(), world_size, devs.data()));
  std::vector<std::thread> ths;
  for(int rank = 0; rank < world_size; ++ rank) {
    ths.emplace_back([world_size, rank, comm=comms[rank], args=inputs[rank], exp_output=exp_outputs[rank]]() {
      _TestShuffleThread(args, exp_output, rank, world_size, comm);
    });
  }
  for(auto& t: ths) {
    t.join();
  }
  for(auto comm: comms) {
    NCCLCHECK(ncclCommDestroy(comm));
  }
}

TEST(Sampling, Shuffle) {
  int world_size = 2;
  std::vector<ShuffleArgs> args;
  std::vector<ShuffleOutput> exp_output;
  args.push_back(ShuffleArgs({{0, 1, 2}, {1, 1}, {0, 2}}));
  args.push_back(ShuffleArgs({{0, 1, 2}, {1, 1}, {1, 3}}));
  exp_output.push_back(ShuffleOutput({2, {0, 1}, {0, 1, 2}}));
  exp_output.push_back(ShuffleOutput({2, {2, 3}, {0, 1, 2}}));
  _TestShuffle(world_size, args, exp_output);

  world_size = 3;
  args.clear();
  exp_output.clear();

  args.push_back(ShuffleArgs({{0, 1, 2, 3}, {1, 1, 1}, {0, 2, 3}}));
  args.push_back(ShuffleArgs({{0, 0, 0, 0}, {0, 0, 0}, {}}));
  args.push_back(ShuffleArgs({{0, 0, 0, 1}, {0, 0, 1}, {1}}));
  exp_output.push_back(ShuffleOutput({1, {0}, {0, 1, 1, 1}}));
  exp_output.push_back(ShuffleOutput({1, {2}, {0, 1, 1, 1}}));
  exp_output.push_back(ShuffleOutput({2, {3, 1}, {0, 1, 1, 2}}));
  _TestShuffle(world_size, args, exp_output);
  // TODO: Add a test to consider the case when the output frontier is 0
}
*/