#ifndef DGL_DS_DEVICE_MANAGER_H
#define DGL_DS_DEVICE_MANAGER_H

#include <memory>
#include <map>
#include <dgl/array.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <dgl/aten/array_ops.h>

using namespace dgl;

namespace dgl {
namespace ds{

class MemoryManager {
 public:
  struct ArrayInfo {
    NDArray array;
    std::vector<int64_t> shape;
    DLDataType dtype;
    DLContext ctx;
  };
  NDArray Empty(const std::string& name, const std::vector<int64_t>& shape, DLDataType dtype, DLContext ctx) {
    return _GetArray(name, shape, dtype, ctx, [&shape, dtype, ctx]() {
      return NDArray::Empty(shape, dtype, ctx);
    });
  }
  template<typename T>
  NDArray Full(const std::string& name, T val, int64_t length, DLContext ctx) {
    auto dtype = DLDataTypeTraits<T>::dtype;
    std::vector<int64_t> shape = {length};
    NDArray ret = _GetArray(name, shape, dtype, ctx, [val, length, ctx]() {
      return aten::Full(val, length, ctx);
    });
    if(ctx.device_type == kDLGPU) {
      auto ptr = thrust::device_ptr<T>(ret.Ptr<T>());
      thrust::fill(ptr, ptr + shape[0], val);
    } else {
      auto ptr = ret.Ptr<T>();
      std::fill(ptr, ptr + shape[0], val);
    }
    return ret;
  }

  void ClearUseCount() {
    use_count_.clear();
  }

  static MemoryManager* Global() {
    static MemoryManager instance;
    return &instance;
  }

 private:
  template<typename Lambda>
  NDArray _GetArray(const std::string& name, const std::vector<int64_t>& shape, DLDataType dtype, DLContext ctx, Lambda func) {
    CHECK(use_count_[name] == 0) << "Using the same array name twice";
    if(arrays_.count(name) == 0 || shape[0] > arrays_[name].shape[0]) {
      auto array = func();
      ArrayInfo info{array, shape, dtype, ctx};
      arrays_[name] = info;
      use_count_[name] = 1;
      return array;
    }
    use_count_[name] += 1;
    _CheckInfo(name, shape, dtype, ctx);
    auto array = arrays_[name].array;
    if(shape[0] < array->shape[0]) {
      auto ret = array.CreateView(shape, dtype);
      return ret;
    }
    return array;
  }

  void _CheckInfo(const std::string& name, const std::vector<int64_t>& shape, DLDataType dtype, DLContext ctx) {
    const auto& array_info = arrays_[name];
    CHECK_GE(array_info.shape[0], shape[0]) << "Name: " << name;
    CHECK(array_info.dtype == dtype);
    CHECK(array_info.ctx.device_type == ctx.device_type);
  }

  std::map<std::string, int> use_count_;
  std::map<std::string, ArrayInfo> arrays_;
};

}
}

#endif