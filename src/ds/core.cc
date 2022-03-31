#include "nccl.h"
#include "../c_api_common.h"
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/ndarray.h>
#include "dmlc/logging.h"
#include <cuda_runtime.h>
#include <thread>

#include "context.h"
#include "./conn/nvmlwrap.h"
#include "./utils.h"
#include "../runtime/cuda/cuda_common.h"
#include "schedule.h"
#include "./kernel_controller.h"

using namespace dgl::runtime;

namespace dgl {
namespace ds {

std::string NCCLIdToString(ncclUniqueId id) {
  return std::string(id.internal, id.internal + NCCL_UNIQUE_ID_BYTES);
}

ncclUniqueId StringToNCCLId(const std::string& str) {
  ncclUniqueId ret;
  CHECK_EQ(str.length(), NCCL_UNIQUE_ID_BYTES);
  memcpy(ret.internal, str.data(), NCCL_UNIQUE_ID_BYTES);
  return ret;
}

void SetupGpuCommunicationEnv(CommInfo *comm_info) {
  auto* ds_context = DSContext::Global();
  int rank = ds_context->rank;
  int world_size = ds_context->world_size;
  auto* coor = ds_context->coordinator.get();
  int n_block = GetEnvParam("N_BLOCK", 16);
  std::vector<std::shared_ptr<Connection>> conns;
  for(int r = 0; r < world_size; ++r) {
    conns.push_back(Connection::GetConnection(coor->GetPeerInfos()[rank], coor->GetPeerInfos()[r]));
  }
  BuildCommInfo(n_block, conns, coor, comm_info);
}

void InitNcclComm(ncclComm_t *nccl_comm, DSContext *ds_context, int world_size, int rank) {
  ncclUniqueId nccl_id;
  if (rank == 0) {
    ncclGetUniqueId(&nccl_id);
  }
  std::string nccl_id_str = NCCLIdToString(nccl_id);
  ds_context->coordinator->Broadcast(nccl_id_str);
  nccl_id = StringToNCCLId(nccl_id_str);
  ncclCommInitRank(nccl_comm, world_size, nccl_id, rank);
}

void Initialize(int rank, int world_size, int thread_num, bool enable_kernel_control, bool enable_comm_control, bool enable_profiler) {
  LOG(INFO) << "Rank [" << rank << "] initializing DS context";
  auto* ds_context = DSContext::Global();
  ds_context->initialized = true;
  ds_context->rank = rank;
  ds_context->world_size = world_size;
  int master_port = GetEnvParam("MASTER_PORT", 12633);
  int comm_port = GetEnvParam("COMM_PORT", 12644);
  ds_context->coordinator = std::unique_ptr<Coordinator>(new Coordinator(rank, world_size, master_port));
  ds_context->comm_coordinator = std::unique_ptr<Coordinator>(new Coordinator(rank, world_size, comm_port));
  cudaSetDevice(rank);

  ds_context->thread_num = thread_num;

  ds_context->enable_kernel_control = enable_kernel_control;
  LOG(INFO) << "Enable kernel control? " << enable_kernel_control;

  ds_context->enable_comm_control = enable_comm_control;

  int use_nccl = GetEnvParam("USE_NCCL", 0);
  if (!use_nccl) {
    // Build our communication environment
    ds_context->comm_info.resize(thread_num);
    wrapNvmlInit();
    for (int i=0; i<thread_num; i++) {
      ds_context->comm_info[i] = std::unique_ptr<CommInfo>(new CommInfo());
      SetupGpuCommunicationEnv(ds_context->comm_info[i].get());
    }
  } else {
    // Build NCCL environment
    ds_context->nccl_comm.resize(thread_num);
    for (int i=0; i<thread_num; i++) {
      InitNcclComm(&(ds_context->nccl_comm[i]), ds_context, world_size, rank);
    }
  }

  //init scheduler
  std::thread scheduler(&Scheduler::Schedule, Scheduler::Global());
  std::thread coordinator(&Scheduler::Coordinate, Scheduler::Global());
  scheduler.detach();
  coordinator.detach();

  LOG(INFO) << "Rank " + std::to_string(rank) + " successfully builds nccl communicator";

  ds_context->enable_profiler = enable_profiler;
  if (enable_profiler) {
    ds_context->profiler = std::unique_ptr<Profiler>(new Profiler());
  }
  LOG(INFO) << "Enable profiler? " << enable_profiler;
}

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSInitialize")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  int rank = args[0];
  int world_size = args[1];
  int thread_num = args[2];
  bool enable_kernel_control = args[3];
  bool enable_comm_control = args[4];
  bool enable_profiler = args[5];
  Initialize(rank, world_size, thread_num, enable_kernel_control, enable_comm_control, enable_profiler);
});

// Set dgl thread local stream
DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSSetStream")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  void* s = args[0];
  int thread_id = args[1];
  int role = args[2];
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  thr_entry->thread_id = thread_id;
  thr_entry->stream = (cudaStream_t)s;
  thr_entry->role = role;
  LOG(INFO) << "Set local stream: " << thr_entry->stream;
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  
  // Create data copy stream to pipeline with kernels
  CUDACHECK(cudaStreamCreate(&thr_entry->data_copy_stream));
  CUDACHECK(cudaDeviceSynchronize());

  auto* ds_thread_local = DSThreadEntry::ThreadLocal();
  ds_thread_local->pinned_array_counter = 0;
  for(int i = 0; i < N_PINNED_ARRAY; ++i) {
    ds_thread_local->pinned_array[i] = CreatePinnedArray(DLDataType({kDLInt, 64, 1}), THREAD_LOCAL_PINNED_ARRAY_SIZE);
  }
});

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSSetQueueSize")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  int queue_size = args[0];
  KernelController::SetQueueSize(queue_size);
});

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSAllgatherTrainLabels")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  auto* coor = DSContext::Global()->coordinator.get();
  IdArray labels = args[0];
  auto labels_vec = labels.ToVector<int64_t>();
  auto gathered_labels = coor->Gather(labels_vec);
  std::vector<int64_t> flatten_labels;
  if(coor->IsRoot()) {
    flatten_labels = Flatten(gathered_labels);
  }
  coor->Broadcast(flatten_labels);
  *rv = IdArray::FromVector(flatten_labels, labels->ctx);
});

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSProfilerReport")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  int num_epochs = args[0];
  CHECK(DSContext::Global()->enable_profiler);
  DSContext::Global()->profiler->Report(num_epochs);
});

}
}
