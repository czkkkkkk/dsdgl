#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>

namespace dgl {
namespace ds {

DGL_REGISTER_GLOBAL("ds._CAPI_DGLDSSampleNeighbors")
.set_body([] (DGLArgs args, DGLRetValue *rv) {
    HeteroGraphRef hg = args[0];
    const auto& nodes = ListValueToVector<IdArray>(args[1]);
    // TODO

  });

}
}