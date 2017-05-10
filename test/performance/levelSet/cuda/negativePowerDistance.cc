/* -*- C++ -*- */

#include "stlib/levelSet/powerDistanceCuda.h"
#include "stlib/levelSet/flood.h"
#include "stlib/levelSet/marchingSimplices.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer.h"
#include "stlib/container/EquilateralArray.h"
#include "stlib/cuda/check.h"
#include "stlib/geom/mesh/iss/file_io.h"
#include "stlib/geom/mesh/iss/distinct_points.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <vector>

#include <cassert>
#include <cmath>

#include "common.h"

using namespace stlib;

int
main(int argc, char* argv[])
{
  // Parse the program name and options.
  ads::ParseOptionsArguments parser(argc, argv);
  parseNameAndOptions(&parser);

  // Read the set of balls.
  std::vector<geom::Ball<Number, Dimension> > balls;
  readBalls(&parser, &balls);

  //
  // Determine an appropriate grid and domain.
  //
  if (verbose) {
    std::cout << "Constructing the grid..." << std::endl;
  }
  ads::Timer timer;
  timer.tic();
  // Place a bounding box around the balls.
  BBox domain;
  domain.bound(balls.begin(), balls.end());
  domain.offset(2 * targetGridSpacing);
  // Make the grid.
  Grid grid(domain, targetGridSpacing);
  if (verbose) {
    std::cout << "  Time = " << timer.toc() << std::endl;
    std::cout << "  Lower corner = " << grid.lowerCorner << '\n'
              << "  Spacing = " << grid.spacing << '\n'
              << "  Extents = " << grid.extents() << '\n';
  }

  //
  // Calculate the level set using the GPU.
  //
  if (verbose) {
    std::cout << "Calculate the level set..." << std::endl;
  }
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  levelSet::negativePowerDistanceCuda(&grid, balls);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  if (verbose) {
    std::cout << "Elapsed time for CUDA kernel = " << elapsedTime << " ms.\n";
  }

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));

  // Note: Comment out the rest of main when using computeprof.

  // Flood fill.
  floodFill(&grid, balls);
  // Compute the content.
  const std::pair<Number, Number> cb = contentAndBoundary(&parser, grid);
  // Write the level set output file if a file name was specified.
  writeLevelSet(&parser, grid);

  if (verbose) {
    std::cout << "Volume and surface area:\n";
  }
  std::cout << cb.first << ' ' << cb.second << '\n';

  return 0;
}
