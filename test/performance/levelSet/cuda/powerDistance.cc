/* -*- C++ -*- */

#include "stlib/levelSet/powerDistanceCuda.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/container/EquilateralArray.h"
#include "stlib/cuda/check.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include <cassert>
#include <cmath>

using namespace stlib;

std::string programName;

void
exitOnError()
{
  std::cerr
      << "Usage:\n"
      << programName << "\n";
  exit(1);
}

int
main(int argc, char* argv[])
{
  // Parse the command line arguments and options.
  ads::ParseOptionsArguments parser(argc, argv);
  programName = parser.getProgramName();
  // There should be no arguments.
  if (! parser.areArgumentsEmpty()) {
    exitOnError();
  }
  // There should be no more options.
  if (! parser.areOptionsEmpty()) {
    std::cerr << "Error: unmatched options:\n";
    parser.printOptions(std::cerr);
    exitOnError();
  }

  container::EquilateralArray<float, 3, levelSet::PatchExtent> patch;
  const std::array<float, 3> lowerCorner = {{0, 0, 0}};
  const float spacing = 1;
  std::vector<geom::BallSquared<float, 3> > balls(10);
  const float Scaling = (levelSet::PatchExtent - 1.) / RAND_MAX;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = 0; j != 3; ++j) {
      balls[i].center[j] = Scaling * rand();
    }
    balls[i].squaredRadius = 1;
  }

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  levelSet::powerDistanceCuda(&patch, lowerCorner, spacing, balls);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "Elapsed time = " << elapsedTime << " ms.\n";

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  return 0;
}
