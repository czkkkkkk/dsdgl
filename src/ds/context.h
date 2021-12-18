#ifndef DGL_DS_CONTEXT_H_
#define DGL_DS_CONTEXT_H_

#include <nccl.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <memory>

using namespace dgl::runtime;

namespace dgl {
namespace ds {

struct DSContext {
  ncclComm_t nccl_comm;
};

class DSContextObject : public Object {
 public:
  DSContextObject() {
    ds_context_ = std::make_shared<DSContext>();
  }
  DSContext* GetContext() const { return ds_context_.get(); }

  static constexpr const char* _type_key = "DSContext";
  DGL_DECLARE_OBJECT_TYPE_INFO(DSContextObject, Object);
 private:
  std::shared_ptr<DSContext> ds_context_;

};

class DSContextRef : public ObjectRef {
 public:
  DSContextRef() = default;
  explicit DSContextRef(std::shared_ptr<Object> obj): ObjectRef(obj) {}

  const DSContextObject* operator->() const {
    return static_cast<const DSContextObject*>(obj_.get());
  }
  using ContainerType = DSContextObject;
};
}
}

#endif