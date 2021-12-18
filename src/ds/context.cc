#include "context.h"


namespace dgl {
namespace ds {

DGL_REGISTER_GLOBAL("ds._CAPI_DGLCreateDSContext")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  auto o = std::make_shared<DSContextObject>();
  *rv = o;
});

}
}