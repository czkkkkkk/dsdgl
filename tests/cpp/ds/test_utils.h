#ifndef TEST_DS_UTILS_H_
#define TEST_DS_UTILS_H_

#include <vector>
#include <cuda_runtime.h>
#include <string>

template<typename T>
void CheckVectorEq(const std::vector<T>& lhs, const std::vector<T>& rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());
  for(size_t i = 0; i < lhs.size(); ++i) {
    EXPECT_EQ(lhs[i], rhs[i]);
  }
}


#endif