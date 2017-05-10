/* -*- C -*- */

#include "clipAreaGpu.h"
#include "clipAreaKernel.h"

#include "stlib/ads/timer.h"

#include <cuda_runtime_api.h>

#include <iostream>
#include <numeric>

namespace
{

void
exitOnError(cudaError_t error)
{
  std::cerr << "Encountered the error number " << error << ".\n";
  error = cudaThreadExit();
  if (error != cudaSuccess) {
    std::cerr << "Encountered error number " << error
              << " in cudaThreadExit().\n";
  }
  exit(1);
}

}

std::size_t
calculateAreaGpu(const std::vector<float3>& referenceMesh,
                 const std::vector<float3>& centers,
                 const std::vector<std::size_t>& clippingSizes,
                 const std::vector<std::size_t>& clippingIndices)
{
  // Allocate device memory for the reference mesh.
  float* referenceMeshDevice;
  cudaError_t error;
  error = cudaMalloc((void**)&referenceMeshDevice,
                     3 * referenceMesh.size() * sizeof(float));
  if (error != cudaSuccess) {
    exitOnError(error);
  }
  {
    // Reorder the point coordinates.
    const std::size_t N = referenceMesh.size();
    std::vector<float> coordinates(3 * N);
    for (std::size_t i = 0; i != N; ++i) {
      coordinates[i] = referenceMesh[i].x;
      coordinates[i + N] = referenceMesh[i].y;
      coordinates[i + 2 * N] = referenceMesh[i].z;
    }
    // Copy the reference mesh to the device.
    error = cudaMemcpy(referenceMeshDevice, &coordinates[0],
                       3 * N * sizeof(float), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) {
      exitOnError(error);
    }
  }

  // Allocate device memory for the ball centers.
  float3* centersDevice;
  error = cudaMalloc((void**)&centersDevice,
                     centers.size() * sizeof(float3));
  if (error != cudaSuccess) {
    exitOnError(error);
  }
  // Copy the ball centers to the device.
  error = cudaMemcpy(centersDevice, &centers[0],
                     centers.size() * sizeof(float3),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    exitOnError(error);
  }

  // Make the array of clipping delimiters.
  std::vector<unsigned> delimiters(clippingSizes.size() + 1);
  delimiters[0] = 0;
  std::partial_sum(clippingSizes.begin(), clippingSizes.end(),
                   delimiters.begin() + 1);
  // Allocate device memory for the clipping delimiters.
  unsigned* delimitersDevice;
  error = cudaMalloc((void**)&delimitersDevice,
                     delimiters.size() * sizeof(unsigned));
  if (error != cudaSuccess) {
    exitOnError(error);
  }
  // Copy the delimiters to the device.
  error = cudaMemcpy(delimitersDevice, &delimiters[0],
                     delimiters.size() * sizeof(unsigned),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    exitOnError(error);
  }

  // Make the array of clipping centers.
  std::vector<float3> clippingCenters(clippingIndices.size());
  for (std::size_t i = 0; i != clippingCenters.size(); ++i) {
    clippingCenters[i] = centers[clippingIndices[i]];
  }
  // Allocate device memory for the clipping centers.
  float3* clippingCentersDevice;
  error = cudaMalloc((void**)&clippingCentersDevice,
                     clippingCenters.size() * sizeof(float3));
  if (error != cudaSuccess) {
    exitOnError(error);
  }
  // Copy the delimiters to the device.
  error = cudaMemcpy(clippingCentersDevice, &clippingCenters[0],
                     clippingCenters.size() * sizeof(float3),
                     cudaMemcpyHostToDevice);
  if (error != cudaSuccess) {
    exitOnError(error);
  }

  // Allocate device memory for the active count.
  unsigned* activeCountsDevice;
  error = cudaMalloc((void**)&activeCountsDevice,
                     centers.size() * sizeof(unsigned));
  if (error != cudaSuccess) {
    exitOnError(error);
  }

  // Invoke the kernel.
  ads::Timer timer;
  timer.tic();
  clipKernel(referenceMeshDevice, referenceMesh.size(), centersDevice,
             centers.size(), delimitersDevice, clippingCentersDevice,
             activeCountsDevice);
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    exitOnError(error);
  }
  cudaThreadSynchronize();
  const double elapsedTime = timer.toc();
  std::cout << "Time for kernel execution = " << elapsedTime << ".\n";

  cudaFree(referenceMeshDevice);
  cudaFree(centersDevice);
  cudaFree(delimitersDevice);
  cudaFree(clippingCentersDevice);

  // Copy result from device memory to host memory
  std::vector<unsigned> activeCounts(centers.size());
  error = cudaMemcpy(&activeCounts[0], activeCountsDevice,
                     activeCounts.size() * sizeof(unsigned),
                     cudaMemcpyDeviceToHost);
  if (error != cudaSuccess) {
    exitOnError(error);
  }

  cudaFree(activeCountsDevice);

  return std::accumulate(activeCounts.begin(), activeCounts.end(),
                         std::size_t(0));
}
