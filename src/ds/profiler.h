#ifndef DGL_DS_PROFILER_H_
#define DGL_DS_PROFILER_H_

#include <dgl/array.h>


namespace dgl {
namespace ds {

using namespace dgl::runtime;
using namespace dgl::aten;

class Profiler {
 public:
  void UpdateFeatCacheRate(int hit, int miss) { /* TODO */ };

  void UpdateDSSamplingLocalCount(IdArray sampled_index, int fanout);
  void UpdateDSSamplingNvlinkCount(IdArray send_offset, int fanout);
  void UpdateUVASamplingCount(IdArray sampled_index, int fanout);
  void Report(int num_epochs);

 private:
  int64_t ds_sampling_local_count_ = 0;
  int64_t ds_sampling_nvlink_count_ = 0; 
  int64_t ds_sampling_nvlink_node_count_ = 0;
  int64_t ds_sampling_pcie_count_ = 0;
  int64_t saved_count_ = 0;
  int64_t uva_sampling_pcie_count_ = 0;
  int64_t uva_sampling_saved_count_ = 0;

};

}
}

#endif