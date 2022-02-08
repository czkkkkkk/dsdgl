#include "./scan.h"

  
#include <dgl/runtime/device_api.h>

using namespace dgl;
using namespace dgl::runtime;

namespace dgl {
namespace ds {

static const int ELE_PER_BLOCK = 1024;
static const int THREADS_PER_BLOCK = 512;

__device__
void _GetValue(IdType* workspace, int tid, int size, int rank, IdType* part_ids, IdType *ret) {
  if(tid < size) {
    if(part_ids != nullptr) {
      *ret = part_ids[tid] == rank;
    } else {
      *ret = workspace[tid];
    }
  } else {
    *ret = 0;
  }
}

// Inplace exclusive multi-way scan
__global__
void _MultiWayCtaScanKernel(IdType* workspace, int size, IdType* part_ids, int world_size, IdType* sums) {
  __shared__ IdType temp[ELE_PER_BLOCK];
  int tid = threadIdx.x;
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int n_blocks = gridDim.y;

  int block_inc = size * rank;
  int thread_inc = ELE_PER_BLOCK * bid;
  workspace += block_inc + thread_inc;
  part_ids += thread_inc;
  if (ELE_PER_BLOCK * bid + ELE_PER_BLOCK > size) {
    size = size % ELE_PER_BLOCK;
  } else {
    size = ELE_PER_BLOCK;
  }
  int ai = tid;
  int bi = tid + ELE_PER_BLOCK / 2;
  _GetValue(workspace, ai, size, rank, part_ids, temp + ai);
  _GetValue(workspace, bi, size, rank, part_ids, temp + bi);

  int offset = 1;
  for(int d = ELE_PER_BLOCK >> 1; d > 0; d >>= 1) {
    __syncthreads();
    if(tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      temp[bi] += temp[ai];
    }
    offset <<= 1;
  }
  
  if(tid == 0) {
    if(sums != nullptr) {
      sums[bid+rank*n_blocks] = temp[ELE_PER_BLOCK-1];
    }
    temp[ELE_PER_BLOCK-1] = 0;
  }
  for(int d = 1; d < ELE_PER_BLOCK; d <<= 1) {
    offset >>= 1;
    __syncthreads();
    if(tid < d) {
      int ai = offset * (2 * tid + 1) - 1;
      int bi = offset * (2 * tid + 2) - 1;
      IdType t = temp[ai];
      temp[ai] = temp[bi];
      temp[bi] += t;
    }
  }
  __syncthreads();
  if(ai < size) workspace[ai] = temp[ai];
  if(bi < size) workspace[bi] = temp[bi];
}

__global__
void _ScanAddKernel(IdType* workspace, IdType* sums, int size, int world_size) {
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int tid = blockDim.x * bid + threadIdx.x;
  int n_blocks = gridDim.y;
  int stride = gridDim.y * blockDim.x;
  workspace += size * rank;
  while(tid < size) {
    workspace[tid] += sums[rank*n_blocks+bid];
    tid += stride;
  }
}

void _MultiWayScanRecursive(IdType* workspace, int size, IdType* part_ids, int world_size, DeviceAPI *device, DGLContext ctx) {
  int n_blocks = (size + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;
  IdType *sums = nullptr;
  if (size > ELE_PER_BLOCK) {
    sums = (IdType*)device->AllocWorkspace(ctx, world_size * n_blocks * sizeof(IdType));
  }
  const dim3 grid(world_size, n_blocks);
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _MultiWayCtaScanKernel<<<grid, THREADS_PER_BLOCK, 0, thr_entry->stream>>>(workspace, size, part_ids, world_size, sums);
  if(size > ELE_PER_BLOCK) {
    _MultiWayScanRecursive(sums, n_blocks, nullptr, world_size, device, ctx);
    auto* thr_entry = CUDAThreadEntry::ThreadLocal();
    _ScanAddKernel<<<grid, THREADS_PER_BLOCK * 2, 0, thr_entry->stream>>>(workspace, sums, size, world_size);
    device->FreeWorkspace(ctx, sums);
  }
}

__global__
void _PermutateKernel(IdType* workspace, IdType* input, IdType* part_offset, IdType* part_ids, int size, int world_size, IdType* sorted, IdType* index) {
  int rank = blockIdx.x;
  int bid = blockIdx.y;
  int tid = bid * ELE_PER_BLOCK + threadIdx.x;
  if(tid < size && part_ids[tid] == rank) {
    IdType tval = input[tid];
    IdType pos = workspace[rank * size + tid] + part_offset[rank];
    sorted[pos] = tval;
    index[pos] = tid;
  }
}

static void _Permutate(IdType *workspace, IdType* input, IdType* part_offset, IdType* part_ids, int size, int world_size, IdType *sorted, IdType* index) {
  int n_blocks = (size + ELE_PER_BLOCK - 1) / ELE_PER_BLOCK;
  const dim3 grid(world_size, n_blocks);
  auto* thr_entry = CUDAThreadEntry::ThreadLocal();
  _PermutateKernel<<<grid, ELE_PER_BLOCK, 0, thr_entry->stream>>>(workspace, input, part_offset, part_ids, size, world_size, sorted, index);
}

std::pair<IdArray, IdArray> MultiWayScan(IdArray input, IdArray part_offset, IdArray part_ids, int world_size) {
  if(input->shape[0] == 0) {
    return {NullArray(input->dtype, input->ctx), NullArray(input->dtype, input->ctx)};
  }
  int size = input->shape[0];
  auto device = DeviceAPI::Get(input->ctx);
  size_t workspace_size = world_size * size * sizeof(IdType);
  IdType* workspace = (IdType*)device->AllocWorkspace(input->ctx, workspace_size);
  _MultiWayScanRecursive(workspace, size, part_ids.Ptr<IdType>(), world_size, device, input->ctx);

  IdArray sorted = IdArray::Empty({input->shape[0]}, input->dtype, input->ctx);
  IdArray index = IdArray::Empty({input->shape[0]}, input->dtype, input->ctx);

  _Permutate(workspace, input.Ptr<IdType>(), part_offset.Ptr<IdType>(), part_ids.Ptr<IdType>(), size, world_size, sorted.Ptr<IdType>(), index.Ptr<IdType>());
  device->FreeWorkspace(input->ctx, workspace);
  return {sorted, index};
}

}
}