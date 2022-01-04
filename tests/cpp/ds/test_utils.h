#ifndef TEST_DS_UTILS_H_
#define TEST_DS_UTILS_H_

#include <vector>
#include <cuda_runtime.h>
#include <string>

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

template<typename T>
void CheckVectorEq(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for(int i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(lhs[i], rhs[i]);
  }
}


#endif