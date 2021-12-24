#include "context.h"

namespace dgl {
namespace ds {

DGL_REGISTER_GLOBAL("ds.context._CAPI_DGLCreateDSContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  int rank = args[0];
  auto o = std::make_shared<DSContextObject>(rank);
  *rv = o;
});

}
}