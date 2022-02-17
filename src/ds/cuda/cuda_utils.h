#ifndef DGL_DS_CUDA_UTILS_H_
#define DGL_DS_CUDA_UTILS_H_

#include <cuda_runtime.h>


#include "../utils.h"

namespace dgl {
namespace ds {

template<typename T>
std::string DeviceVecToString(const T* ptr, size_t size) {
  std::vector<T> vec(size);
  cudaMemcpy(vec.data(), ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
  return VecToString(vec);
}

}
}

#endif