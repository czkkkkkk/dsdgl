#ifndef DGL_DS_BLOCK_COMM_INFO_
#define DGL_DS_BLOCK_COMM_INFO_

#include <memory>
#include <stdalign.h>
#include <cuda_runtime.h>

#include "../coordinator.h"
#include "../conn/connection.h"

namespace dgl {
namespace ds {

#define MAX_CONN_INFO_SIZE 256
#define CACHE_LINE_SIZE 128
#define MEM_ALIGN 4096

// NOTE: RECV_BUFFER_SIZE * N_BLOCKs must greater than the communication size of 
// a connection. We may need to reduce the recv buffer size by partition the communication
// into stages
// 10MB
static const size_t RECV_BUFFER_SIZE = 50 * 1024 * 1024;


struct SendDevMem {
  union {
    struct {
      uint64_t done;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad2[MEM_ALIGN];
  };
};

struct __align__(16) RecvDevMem {
  union {
    struct {
      uint64_t ready;
      char pad1[CACHE_LINE_SIZE - sizeof(uint64_t)];
    };
    char pad3[MEM_ALIGN];
  };
  char buff[1];  // Actually larger than that
};

struct ConnInfo {
  uint64_t *next_ready;
  uint64_t *prev_done;
  
  uint64_t *my_ready, *my_done;

  void *my_recv_buff, *next_recv_buff;
};

struct BlockCommInfo {

  SendDevMem **send_dev_mem;
  RecvDevMem **recv_dev_mem;

  ConnInfo *conn_info;

};

void BuildBlockCommInfo(int block_id, const std::vector<std::shared_ptr<Connection>>& conns, Coordinator* coordinator, BlockCommInfo* block_comm_info);

}
}

#endif
