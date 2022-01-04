#include <gtest/gtest.h>
#include <dmlc/logging.h>

#include "ds/utils.h"
#include "ds/core.h"
#include "mpi.h"

using namespace dgl::ds;

GTEST_API_ int main(int argc, char **argv) {
  testing::InitGoogleTest(&argc, argv);
  LOG(INFO) << "MPI test is enabled";
  MPI_Init(&argc, &argv);
  LOG(INFO) << "MPI test is not enabled";

  int rank, world_size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  SetEnvParam("MASTER_PORT", 12307);
  Initialize(rank, world_size);

  int result = RUN_ALL_TESTS();

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

  return result;
}