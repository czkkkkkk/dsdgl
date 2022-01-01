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

template <typename T>
T GetEnvParam(const std::string &key, T default_value) {
  auto gccl_str = std::string("DGL_DS_") + key;
  char *ptr = std::getenv(gccl_str.c_str());
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

int GetAvailablePort();
std::string GetHostName();

}
}

#endif