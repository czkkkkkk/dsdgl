#ifndef DGL_DS_CUDA_SCAN_H_
#define DGL_DS_CUDA_SCAN_H_

#include <dgl/array.h>
#include "./ds_kernel.h"

namespace dgl {
namespace ds {

std::pair<IdArray, IdArray> MultiWayScan(IdArray input, IdArray part_offset, IdArray part_ids, int world_size);
  


}
}

#endif