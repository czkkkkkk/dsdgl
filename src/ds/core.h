#ifndef DGL_DS_CORE_H_
#define DGL_DS_CORE_H_

namespace dgl {
namespace ds {

void Initialize(int rank, int world_size, int global_rank, int global_world_size, int thread_num=2, bool enable_kernel_control=false, bool enable_comm_control=true, bool enable_profiler=false);

}
}

#endif