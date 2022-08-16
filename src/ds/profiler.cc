#include "./profiler.h"

#include <sstream>

#include "./context.h"
#include "../runtime/cuda/cuda_common.h"
#include "./utils.h"

#define WARP_SIZE 32
#define CACHE_LINE 32
#define PCIE_OVERHEAD 18

namespace dgl {
namespace ds {


std::pair<int64_t, int64_t> ProcessAccess(std::vector<int64_t>& accesses) {
  int64_t ret = 0, saved = 0;
  std::sort(accesses.begin(), accesses.end());
  int64_t prev_accessed_line = -1;

  for(auto pos: accesses) {
    if(pos * 8 / CACHE_LINE == prev_accessed_line) {
      saved++;
      continue;
    }
    prev_accessed_line = pos * 8 / CACHE_LINE;
    ret++;
  }
  return {ret, saved};
}
void Profiler::UpdateDSSamplingLocalCount(IdArray sampled_index, int fanout) {
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  auto *context = DSContext::Global();
  int size = sampled_index->shape[0];
  auto host_sampled_index = sampled_index.CopyTo({kDLCPU, 0}, thr_entry->stream);
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  int64_t *ptr = host_sampled_index.Ptr<int64_t>();
  CHECK(size % fanout == 0);

  for(int i = 0; i < size / fanout; ++i) {
    int64_t count = 0, saved = 0;
    bool local = ptr[i*fanout] > -1;
    for(int j = 0; j < fanout; j += WARP_SIZE) {
      std::vector<int64_t> accesses;
      for(int k = 0; k < WARP_SIZE && j + k < fanout && i * fanout + j + k < size; ++k) {
        if(local) {
          CHECK(ptr[i*fanout+j+k] >= 0);
          accesses.push_back(ptr[i*fanout+j+k]);
        }
        else {
          CHECK(ptr[i*fanout+j+k] < -1);
          accesses.push_back(ENCODE_ID(ptr[i*fanout+j+k]));
        }
      }
      auto pair = ProcessAccess(accesses);
      count += pair.first;
      saved += pair.second;
    }
    if(local) {
      ds_sampling_local_count_ += count * CACHE_LINE + count * PCIE_OVERHEAD;
    } else {
      ds_sampling_pcie_count_ += count * CACHE_LINE + count * PCIE_OVERHEAD;
    }
    saved_count_ += saved * CACHE_LINE + saved * PCIE_OVERHEAD;
  }
}
void Profiler::UpdateDSSamplingNvlinkCount(IdArray send_offset, int fanout) {
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  auto *context = DSContext::Global();
  int rank = context->rank;
  int world_size = context->world_size;
  auto host_send_offset = send_offset.CopyTo({kDLCPU, 0}, thr_entry->stream);
  auto* host_send_offset_ptr = host_send_offset.Ptr<int64_t>();
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  for(int r = 0; r < world_size; ++r) {
    int64_t size = host_send_offset_ptr[r+1] - host_send_offset_ptr[r];
    if(r != rank) {
      ds_sampling_nvlink_count_ += size * fanout * 8;
      ds_sampling_nvlink_node_count_ += size * 8;
    }
  }
}

void Profiler::UpdateUVASamplingCount(IdArray sampled_index, int fanout) {
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  auto *context = DSContext::Global();
  int size = sampled_index->shape[0];
  auto host_sampled_index = sampled_index.CopyTo({kDLCPU, 0}, thr_entry->stream);
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  int64_t *ptr = host_sampled_index.Ptr<int64_t>();
  CHECK(size % fanout == 0);

  for(int i = 0; i < size / fanout; ++i) {
    for(int j = 0; j < fanout; j += WARP_SIZE) {
      std::vector<int64_t> accesses;
      for(int k = 0; k < WARP_SIZE && j + k < fanout && i * fanout + j + k < size; ++k) {
        accesses.push_back(ptr[i*fanout+j+k]);
      }
      auto pair = ProcessAccess(accesses);
      uva_sampling_pcie_count_ += pair.first * CACHE_LINE;
      saved_count_ += pair.second * CACHE_LINE;
    }
  }
}

int64_t GatherSum(int64_t val) {
  auto *context = DSContext::Global();
  std::vector<int64_t> vals = context->coordinator->Gather(val);
  int64_t ret = 0;
  if(context->rank == 0) {
    for(auto v: vals) {
      ret += v;
    }
  }
  return ret;
}
void Profiler::Report(int num_epochs) {
  auto *context = DSContext::Global();
  auto local_count = GatherSum(ds_sampling_local_count_);
  auto nvlink_count = GatherSum(ds_sampling_nvlink_count_);
  auto nvlink_node_count = GatherSum(ds_sampling_nvlink_node_count_);
  auto saved = GatherSum(saved_count_);
  auto pcie_count = GatherSum(ds_sampling_pcie_count_);

  if(context->rank == 0) {
    std::stringstream ss;
    ss << "[Rank: " << context->rank << "] is reporting profiler results\n";
    ss << " # DS Sampling Local Access: " << local_count / 1e9 / num_epochs << " GB/epoch\n";
    ss << " # DS Sampling Nvlink Access: " << nvlink_count / 1e9 / num_epochs << " GB/epoch\n";
    ss << " # DS Sampling Nvlink Node Access: " << nvlink_node_count / 1e9 / num_epochs << " GB/epoch\n";
    ss << " # DS Sampling PCIe Access: " << pcie_count / 1e9 / num_epochs << " GB/epoch\n";
    ss << " # DS Sampling Saved Access: " << saved / 1e9 / num_epochs << " GB/epoch\n";
    LOG(INFO) << ss.str();
  }
}

}
}