#ifndef DGL_DS_KERNEL_CONTROLLER_
#define DGL_DS_KERNEL_CONTROLLER_

#include <cuda_runtime.h>
#include <dmlc/logging.h>

#include "./context.h"

namespace dgl {
namespace ds {

class KernelController {
 public:
  static constexpr int HIGH_PRIORITY_QUEUE_THRESHOLD = 2;
  static constexpr int LOW_PRIORITY_MAX_THREADS = 512 * 32;
  static void SetQueueSize(int queue_size) {
    auto* context = DSContext::Global();
    int role = CUDAThreadEntry::ThreadLocal()->role;
    if(role == SAMPLER_ROLE) {
      context->sampler_queue_size = queue_size;
    } else {
      CHECK(role == LOADER_ROLE);
      context->loader_queue_size = queue_size;
    }
  }
  static void AdjustKernelSize(dim3& grid, dim3& block) {
    auto* context = DSContext::Global();
    if (!context->enable_kernel_control) {
      return;
    }
    int role = CUDAThreadEntry::ThreadLocal()->role;
    bool adjust;
    if(role == SAMPLER_ROLE) {
      adjust = context->sampler_queue_size >= HIGH_PRIORITY_QUEUE_THRESHOLD;
    } else {
      CHECK(role == LOADER_ROLE);
      adjust = context->loader_queue_size >= HIGH_PRIORITY_QUEUE_THRESHOLD;
    }
    if(adjust) {
      CHECK(grid.y == 1 && grid.z == 1);
      int n_blocks = grid.x;
      int n_threads = block.x * block.y * block.z;
      grid.x = std::min(n_blocks, (LOW_PRIORITY_MAX_THREADS + n_threads - 1) / n_threads);
    }
  }
};

}
}

#endif