#ifndef DGL_DS_UTILS_H_
#define DGL_DS_UTILS_H_

#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <dmlc/logging.h>
#include <dgl/array.h>

#include <cuda_runtime.h>

#define CUDACHECK(cmd)                                      \
  do {                                                      \
    cudaError_t e = cmd;                                    \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define CUDACHECKERR(e)                                     \
  do {                                                      \
    if (e != cudaSuccess) {                                 \
      LOG(FATAL) << "Cuda error " << cudaGetErrorString(e); \
    }                                                       \
  } while (false);

#define SYSCHECK(call, name)                                     \
do {                                                           \
  int ret = -1;                                                \
  while (ret == -1) {                                          \
    SYSCHECKVAL(call, name, ret);                              \
    if (ret == -1) {                                           \
      LOG(ERROR) << "Got " << strerror(errno) << ", retrying"; \
    }                                                          \
  }                                                            \
} while (0);

#define SYSCHECKVAL(call, name, retval)                                    \
  do {                                                                     \
    retval = call;                                                         \
    if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK &&          \
        errno != EAGAIN) {                                                 \
      LOG(ERROR) << "Call to " << name << " failed : " << strerror(errno); \
    }                                                                      \
  } while (0);

#define SYSCHECKNTIMES(call, name, times, usec, exptype)                    \
  do {                                                                      \
    int ret = -1;                                                           \
    int count = 0;                                                          \
    while (ret == -1 && count < times) {                                    \
      SYSCHECKVALEXP(call, name, ret, exptype);                             \
      count++;                                                              \
      if (ret == -1) {                                                      \
        usleep(usec);                                                       \
      }                                                                     \
    }                                                                       \
    if (ret == -1) {                                                        \
      LOG(ERROR) << "Call to " << name << " timeout : " << strerror(errno); \
    }                                                                       \
  } while (0);

#define SYSCHECKVALEXP(call, name, retval, exptype)                        \
  do {                                                                     \
    retval = call;                                                         \
    if (retval == -1 && errno != EINTR && errno != EWOULDBLOCK &&          \
        errno != EAGAIN && errno != exptype) {                             \
      LOG(ERROR) << "Call to " << name << " failed : " << strerror(errno); \
    }                                                                      \
  } while (0);

using namespace dgl::runtime;

namespace dgl {
namespace ds {

template <typename T>
void DSCudaMalloc(T **ptr) {
  cudaMalloc(ptr, sizeof(T));
}

template <typename T>
void DSCudaMalloc(T **ptr, int size) {
  cudaMalloc(ptr, sizeof(T) * size);
  cudaMemset(*ptr, 0, sizeof(T) * size);
}

template <typename T>
void DSMallocAndCopy(T **ret, const T *src, int size) {
  DSCudaMalloc(ret, size);
  cudaMemcpy(*ret, src, sizeof(T) * size, cudaMemcpyHostToDevice);
}

template <typename T>
void DSMallocAndCopy(T **ret, const std::vector<T> &src) {
  DSMallocAndCopy(ret, src.data(), src.size());
}

template <typename T>
std::vector<T> DSDeviceVecToHost(T* ptr, size_t size) {
  std::vector<T> ret(size);
  cudaMemcpy(ret.data(), ptr, sizeof(T) * size, cudaMemcpyDeviceToHost);
  return ret;
}

static inline void DSCudaHostAlloc(void **ptr, void **devPtr, size_t size) {
  CUDACHECK(cudaHostAlloc(ptr, size, cudaHostAllocMapped));
  memset(*ptr, 0, size);
  *devPtr = *ptr;
}

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

template <typename T>
T GetEnvParam(const std::string &key, T default_value) {
  auto new_key = std::string("DGL_DS_") + key;
  char *ptr = std::getenv(new_key.c_str());
  if (ptr == nullptr) return default_value;
  std::stringstream converter(ptr);
  T ret;
  converter >> ret;
  return ret;
}

template <typename T>
T GetEnvParam(const char *str, T default_value) {
  return GetEnvParam<T>(std::string(str), default_value);
}

template <typename T>
void SetEnvParam(const std::string& key, T value) {
  auto new_key = std::string("DGL_DS_") + key;
  setenv(new_key.c_str(), std::to_string(value).c_str(), 1);
}
template <typename T>
void SetEnvParam(const char* key, T value) {
  SetEnvParam<T>(std::string(key), value);
}

int GetAvailablePort();
std::string GetHostName();

/**
 * @brief Create an IdArray on shared memory. The memory is created based on the array on the root process. However, other processes must have the correct data type for the array.
 * @param arr IdArray
 * @param shm_name the name of the shared memory
 * @return IdArray with shared memory
 */
IdArray CreateShmArray(IdArray arr, const std::string& shm_name);

}
}

#endif