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

void SetupGpuCommunicationEnv() {
  auto* ds_context = DSContext::Global();
  int rank = ds_context->rank;
  int world_size = ds_context->world_size;
  auto* coor = ds_context->coordinator.get();
  int n_block = GetEnvParam("N_BLOCK", 16);
  std::vector<std::shared_ptr<Connection>> conns;
  for(int r = 0; r < world_size; ++r) {
    conns.push_back(Connection::GetConnection(coor->GetPeerInfos()[rank], coor->GetPeerInfos()[r]));
  }
  BuildCommInfo(n_block, conns, coor, &ds_context->comm_info);
  std::vector<std::shared_ptr<Connection>> conns_load;
  for(int r = 0; r < world_size; ++r) {
    conns_load.push_back(Connection::GetConnection(coor->GetPeerInfos()[rank], coor->GetPeerInfos()[r]));
  }
  BuildCommInfo(n_block, conns_load, coor, &ds_context->comm_info_load);
}

void Initialize(int rank, int world_size) {
  LOG(INFO) << "Rank [" << rank << "] initializing DS context";
  auto* ds_context = DSContext::Global();
  ds_context->initialized = true;
  ds_context->rank = rank;
  ds_context->world_size = world_size;
  ds_context->coordinator = std::unique_ptr<Coordinator>(new Coordinator(rank, world_size));
  ds_context->comm_coordinator = std::unique_ptr<Coordinator>(new Coordinator(rank, world_size, 12307));
  cudaSetDevice(rank);

  int use_nccl = GetEnvParam("USE_NCCL", 0);
  wrapNvmlInit();

  // Build our communication environment
  SetupGpuCommunicationEnv();
  // Build NCCL environment
  ncclUniqueId nccl_id, nccl_id_load;
  if (rank == 0) {
    ncclGetUniqueId(&nccl_id);
    ncclGetUniqueId(&nccl_id_load);
  }
  std::string nccl_id_str = NCCLIdToString(nccl_id);
  std::string nccl_id_load_str = NCCLIdToString(nccl_id_load);
  ds_context->coordinator->Broadcast(nccl_id_str);
  nccl_id = StringToNCCLId(nccl_id_str);
  ds_context->coordinator->Broadcast(nccl_id_load_str);
  nccl_id_load = StringToNCCLId(nccl_id_load_str);

  if(world_size >= 1) {
    ncclCommInitRank(&ds_context->nccl_comm, world_size, nccl_id, rank);
    ncclCommInitRank(&ds_context->nccl_comm_load, world_size, nccl_id_load, rank);
  }

  //init scheduler
  std::thread scheduler(&Scheduler::Schedule, Scheduler::Global());
  std::thread coordinator(&Scheduler::Coordinate, Scheduler::Global());
  scheduler.detach();
  coordinator.detach();

  LOG(INFO) << "Rank " + std::to_string(rank) + " successfully builds nccl communicator";
}

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSInitialize")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  int rank = args[0];
  int world_size = args[1];
  Initialize(rank, world_size);
});

// Set dgl thread local stream
DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSSetStream")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  void* s = args[0];
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  thr_entry->stream = (cudaStream_t)s;
  LOG(INFO) << "Set local stream: " << thr_entry->stream;
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  
  // Create data copy stream to pipeline with kernels
  CUDACHECK(cudaStreamCreate(&thr_entry->data_copy_stream));
});

}
}
