#include "./connection.h"

#include <cuda_runtime.h>
#include <nvml.h>

#include "../utils.h"
#include "./nvmlwrap.h"
#include "./topo.h"
#include "./nvlink.h"
#include "./p2p_connection.h"
#include "./jump_connection.h"

namespace dgl {
namespace ds {

#define NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE 32

void GetBusId(char bus_id[], int dev_id) {
  CUDACHECK(cudaDeviceGetPCIBusId(bus_id, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE,
                                  dev_id));
  nvmlDevice_t nvmlDevice;
  CHECK(wrapNvmlDeviceGetHandleByPciBusId(bus_id, &nvmlDevice));
  nvmlPciInfo_t pciInfo;
  CHECK(wrapNvmlDeviceGetHandleByPciInfo(nvmlDevice, &pciInfo));
  strncpy(bus_id, pciInfo.busId, NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE);
}

bool CanP2p(const ProcInfo& my_info, const ProcInfo& peer_info) {
  /* Determine if we can communicate with the peer through p2p */
  // Do not use P2P across root complexes by default (provided CUDA permits
  // it)
  int p2pLevel = PATH_SOC;
  std::string tr_level = GetEnvParam("TRANSPORT_LEVEL", std::string("p2p"));
  if (tr_level != "p2p") return false;

  // Rule out different nodes
  if (my_info.hostname != peer_info.hostname) return false;

  // Do not detect topology if we're on the same GPU. Note this is not really
  // supported.
  if (my_info.dev_id == peer_info.dev_id) {
    return true;
  }

  // See if CUDA can do P2P
  int p2p;
  if (cudaDeviceCanAccessPeer(&p2p, my_info.dev_id, peer_info.dev_id) !=
      cudaSuccess) {
    LOG(FATAL) << "peer query failed between dev" << my_info.dev_id
               << " and dev " << peer_info.dev_id;
    return false;
  }
  if (p2p == 0) return false;

  // Check for NVLink/NVswitch
  char my_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE],
      peer_bus_id[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
  GetBusId(my_bus_id, my_info.dev_id);
  GetBusId(peer_bus_id, peer_info.dev_id);
  int nvlinkp2p = getNvlinkGpu(my_bus_id, peer_bus_id);
  if (nvlinkp2p > 0) {
    return true;
  }
  return false;
}

std::shared_ptr<Connection> Connection::GetConnection(ProcInfo r1, ProcInfo r2) {
  if(CanP2p(r1, r2)) {
    return std::make_shared<P2pConnection>(r1, r2);
  }
  LOG(FATAL) << "Currently can only support GPUs with nvlinks";
  CHECK(r1.hostname == r2.hostname) << "Currently only support GPUs on the same machine";
  return nullptr;
  // return std::make_shared<JumpConnection>();
}

}
}