#include "context.h"

namespace dgl {
namespace ds {

typedef dmlc::ThreadLocalStore<DSThreadEntry> DSThreadStore;

DSThreadEntry* DSThreadEntry::ThreadLocal() {
  return DSThreadStore::Get();
}

}
}