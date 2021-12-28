#ifndef DGL_DS_UTILS_H_
#define DGL_DS_UTILS_H_

#include <string>
#include <vector>

namespace dgl {
namespace ds {

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

}
}
#endif