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

void Initialize(int rank, int world_size, int sample_worker_num, int subtensor_load_num) {
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

  // int use_nccl = GetEnvParam("USE_NCCL", 0);
  // wrapNvmlInit();
  // Build our communication environment
  // SetupGpuCommunicationEnv();
  // Build NCCL environment
  
  // init nccl communicators for sample workers
  ds_context->sample_worker_num = sample_worker_num;
  ds_context->sample_nccl_comm = std::unique_ptr<ncclComm_t[]>(new ncclComm_t[sample_worker_num]);
  for (int i=0; i<sample_worker_num; i++) {
    InitNcclComm(&(ds_context->sample_nccl_comm[i]), ds_context, world_size, rank);
  }

  // init nccl communicators for load workers
  ds_context->load_worker_num = subtensor_load_num;
  ds_context->load_nccl_comm = std::unique_ptr<ncclComm_t[]>(new ncclComm_t[subtensor_load_num]);
  for (int i=0; i<subtensor_load_num; i++) {
    InitNcclComm(&(ds_context->load_nccl_comm[i]), ds_context, world_size, rank);
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
  int sample_worker_num = args[2];
  int subtensor_load_num = args[3];
  Initialize(rank, world_size, sample_worker_num, subtensor_load_num);
});

// Set dgl thread local stream
DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSSetStream")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
  void* s = args[0];
  int thread_id = args[1];
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  thr_entry->thread_id = thread_id;
  thr_entry->stream = (cudaStream_t)s;
  LOG(INFO) << "Set local stream: " << thr_entry->stream;
  CUDACHECK(cudaStreamSynchronize(thr_entry->stream));
  
  // Create data copy stream to pipeline with kernels
  CUDACHECK(cudaStreamCreate(&thr_entry->data_copy_stream));
});

}
}
