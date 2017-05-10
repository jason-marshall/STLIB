/* -*- C++ -*- */

#include "stlib/levelSet/vanDerWaalsCuda.h"

#ifndef __CUDA_ARCH__
#include "stlib/levelSet/ballsCuda.h"
#include <cassert>
#endif

namespace stlib
{
namespace levelSet {

namespace {
// We use 32 threads in order to optimize operations such as the warp vote.
// Each thread is responsible for two x-oriented columns
// in the patch.
const unsigned NumThreads = 32;
// The maximum number of influencing balls for a patch. Must be a multiple of 
// NumThreads.
const unsigned MaxInfluencingBalls = 128;
}


__device__
unsigned
countNegative(const float* g) {
   __shared__ unsigned counts[NumThreads];
   counts[threadIdx.x] = 0;
   for (unsigned i = 0; i != 2 * PatchExtent; ++i) {
      counts[threadIdx.x] += g[i] < 0.f;
   }
   // No need to synch here because we use 32 threads.
   // No need to synchronize in between as long as we access the memory via
   // a volatile.
   if (threadIdx.x < 16) {
      volatile unsigned* v = counts;
      v[threadIdx.x] += v[threadIdx.x + 16];
      v[threadIdx.x] += v[threadIdx.x + 8];
      v[threadIdx.x] += v[threadIdx.x + 4];
      v[threadIdx.x] += v[threadIdx.x + 2];
      v[threadIdx.x] += v[threadIdx.x + 1];
   }
   return counts[0];
}


/*
  \pre numInfluencingBalls != 0.

  \param lowerCorner The lower corner of this patch.
  \param spacing The grid spacing between adjacent points.
  \param balls 
  \param patchVolume The accumulated volume for the grid points in this patch.
*/
__device__
void
vanDerWaalsKernel(const float3 lowerCorner,
                  const float spacing,
                  const float4* balls,
                  const unsigned numInfluencingBalls,
                  const unsigned* ballIndices,
                  unsigned* negativeCount) {
   // The overall thread index.
   const unsigned tid = threadIdx.x;
   // Load the influencing balls into shared memory.
   __shared__ float4 b[MaxInfluencingBalls];
   for (unsigned offset = 0; offset != MaxInfluencingBalls;
        offset += NumThreads) {
      if (tid + offset < numInfluencingBalls) {
         b[tid + offset] = balls[ballIndices[tid + offset]];
      }
   }

   // The Cartesian locations of the grid points in the columns for this thread.
   const float z0 = lowerCorner.z + spacing * (tid / PatchExtent);
   const float z1 = z0 + 4 * spacing;
   // Only the y coordinate is constant. We set the x and z coordinates later.
   float3 p = {0, lowerCorner.y + spacing * (tid % PatchExtent), 0};

   // The grid points in the column.
   float g[2 * PatchExtent];
   // Initialize the grid values to infinity.
   for (int i = 0; i != 2 * PatchExtent; ++i) {
      g[i] = 1.f / 0.f;
   }

   // The center of the patch.
   const float3 patchCenter =
      {lowerCorner.x + (PatchExtent - 1) * 0.5f * spacing,
       lowerCorner.y + (PatchExtent - 1) * 0.5f * spacing,
       lowerCorner.z + (PatchExtent - 1) * 0.5f * spacing};
   const float patchRadius = (PatchExtent - 1) * 0.5f * spacing * sqrt(3.f);

   // Compute the distances to the influencing balls. We work with the power
   // distance.
   float d;
   float dyz;
   // Loop over the influencing balls.
   for (int j = 0; j != numInfluencingBalls; ++j) {
      // If the influencing ball is not close to the patch, we don't need
      // to compute the distances. Note that this calculation is duplicated,
      // however, it isn't any faster to spread this distance calculation 
      // across the threads.
      if ((patchCenter.x - b[j].x) * (patchCenter.x - b[j].x) +
          (patchCenter.y - b[j].y) * (patchCenter.y - b[j].y) +
          (patchCenter.z - b[j].z) * (patchCenter.z - b[j].z) >=
          (patchRadius + b[j].w) * (patchRadius + b[j].w)) {
         continue;
      }

      // Loop over the grid points in the first column.
      p.z = z0;
      dyz = (p.y - b[j].y) * (p.y - b[j].y) + (p.z - b[j].z) * (p.z - b[j].z);
      p.x = lowerCorner.x;
      for (int i = 0; i != PatchExtent; ++i) {
         d = (p.x - b[j].x) * (p.x - b[j].x) + dyz - b[j].w * b[j].w;
         if (d < g[i]) {
            g[i] = d;
         }
         p.x += spacing;
      }
      // Loop over the grid points in the second column.
      p.z = z1;
      dyz = (p.y - b[j].y) * (p.y - b[j].y) + (p.z - b[j].z) * (p.z - b[j].z);
      p.x = lowerCorner.x;
      for (int i = PatchExtent; i != 2 * PatchExtent; ++i) {
         d = (p.x - b[j].x) * (p.x - b[j].x) + dyz - b[j].w * b[j].w;
         if (d < g[i]) {
            g[i] = d;
         }
         p.x += spacing;
      }
      // CONTINUE: Try an early exit for all negative.
   }

   const unsigned count = countNegative(g);
   if (tid == 0) {
      atomicAdd(negativeCount, count);
   }
}


// Select a single patch using the block index. Then call a kernel for that
// patch.
__global__
void
vanDerWaalsKernel(const unsigned yExtents,
                  const float3 lowerCorner,
                  const float spacing,
                  const float4* balls,
                  const unsigned* ballIndexOffsets,
                  const unsigned* packedBallIndices,
                  unsigned* negativeCount) {
   // Convert the 2-D block index into a single patch index.
   const unsigned i = blockIdx.x + blockIdx.y * gridDim.x;
   const unsigned begin = ballIndexOffsets[i];
   const unsigned end = ballIndexOffsets[i+1];
   // Check the case that there are no influencing balls. In this case the
   // patch is far away from the protein and there is no contribution to
   // the volume.
   if (begin == end) {
      return;
   }
   const float3 patchLowerCorner =
      {lowerCorner.x + PatchExtent * blockIdx.x * spacing,
       lowerCorner.y + PatchExtent * (blockIdx.y % yExtents) * spacing,
       lowerCorner.z + PatchExtent * (blockIdx.y / yExtents) * spacing};
   vanDerWaalsKernel(patchLowerCorner, spacing, balls, end - begin,
                     &packedBallIndices[begin], negativeCount);
}


#ifndef __CUDA_ARCH__
float
vanDerWaalsCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies) {
   const std::size_t numPatches = product(grid.gridExtents);
   assert(dependencies.getNumberOfArrays() == numPatches);
   // The number of influencing balls for each patch is limited.
   for (std::size_t i = 0; i != dependencies.getNumberOfArrays(); ++i) {
      assert(dependencies.size(i) <= MaxInfluencingBalls);
   }
   
   // Allocate device memory for the balls and copy the memory.
   float4* ballsDev;
   CUDA_CHECK(cudaMalloc((void**)&ballsDev, balls.size() * sizeof(float4)));
   {
      std::vector<float4> buffer(balls.size());
      for (std::size_t i = 0; i != balls.size(); ++i) {
         buffer[i].x = balls[i].center[0];
         buffer[i].y = balls[i].center[1];
         buffer[i].z = balls[i].center[2];
         buffer[i].w = balls[i].radius;
      }
      CUDA_CHECK(cudaMemcpy(ballsDev, &buffer[0],
                            buffer.size() * sizeof(float4),
                            cudaMemcpyHostToDevice));
   }

   // Allocate device memory for the ball index offsets and packed ball indices.
   // Copy the data to the device.
   unsigned* ballIndexOffsetsDev;
   unsigned* packedBallIndicesDev;
   allocateBallIndicesCuda(dependencies, &ballIndexOffsetsDev,
                           &packedBallIndicesDev);
   
   // Allocate device memory for the count of negative grid points.
   unsigned* negativeCountDev;
   CUDA_CHECK(cudaMalloc((void**)&negativeCountDev, sizeof(unsigned)));
   CUDA_CHECK(cudaMemset(negativeCountDev, 0, sizeof(unsigned)));

   const float3 lowerCorner = {grid.lowerCorner[0], grid.lowerCorner[1],
                               grid.lowerCorner[2]};
   // Use a 2-D grid of blocks. (In CUDA we are limited to a 2-D grid.)
   // The second coordinate encodes the second and third patch coordinates.
   // Let (i, j, k) be the patch coordinates, and (x, y) be the block
   // coordinates.
   // x = i
   // y = j + k * grid.patchExtent[1]

   {
      // Check that the maximum grid size is sufficient.
      cudaDeviceProp prop;
      CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
      assert(grid.gridExtents[0] <= prop.maxGridSize[0]);
      assert(grid.gridExtents[1] * grid.gridExtents[2] <= prop.maxGridSize[1]);
   }
   const dim3 GridDim(grid.gridExtents[0],
                      grid.gridExtents[1] * grid.gridExtents[2]);
   // Launch the kernel.
   vanDerWaalsKernel<<<GridDim,NumThreads>>>
      (grid.gridExtents[1], lowerCorner, grid.spacing, ballsDev,
       ballIndexOffsetsDev, packedBallIndicesDev, negativeCountDev);
   // Copy the count of negative grid points back to the host.
   unsigned negativeCount;
   CUDA_CHECK(cudaMemcpy(&negativeCount, negativeCountDev, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

   // Free the device memory.
   CUDA_CHECK(cudaFree(ballsDev));
   CUDA_CHECK(cudaFree(ballIndexOffsetsDev));
   CUDA_CHECK(cudaFree(packedBallIndicesDev));
   CUDA_CHECK(cudaFree(negativeCountDev));

   // Return the total volume.
   return negativeCount * grid.spacing * grid.spacing * grid.spacing;
}
#endif


#ifndef __CUDA_ARCH__
float
vanDerWaalsCuda(const std::vector<geom::Ball<float, 3> >& balls,
                float targetGridSpacing) {
   const std::size_t D = 3;
   typedef GridGeometry<D, PatchExtent, float> Grid;
   typedef Grid::BBox BBox;
   typedef geom::Ball<float, D> Ball;

   //
   // Define the grid geometry for computing the van der Waals volume.
   //
   // Place a bounding box around the balls comprising the molecule.
   BBox targetDomain;
   targetDomain.bound(balls.begin(), balls.end());
   // Define the grid geometry.
   const Grid grid(targetDomain, targetGridSpacing);

   container::StaticArrayOfArrays<unsigned> dependencies;
   patchDependencies(grid, balls.begin(), balls.end(), &dependencies);

#if 0   
   cudaEvent_t start, stop;
   CUDA_CHECK(cudaEventCreate(&start));
   CUDA_CHECK(cudaEventCreate(&stop));
   CUDA_CHECK(cudaEventRecord(start, 0));
#endif

   // Compute the van der Waals volume.
   const float volume = vanDerWaalsCuda(grid, balls, dependencies);

#if 0
   CUDA_CHECK(cudaEventRecord(stop, 0));
   CUDA_CHECK(cudaEventSynchronize(stop));
   float elapsedTime;
   CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
   std::cout << "Elapsed time to compute van der Waals volume = "
             << elapsedTime << " ms.\n";
   CUDA_CHECK(cudaEventDestroy(start));
   CUDA_CHECK(cudaEventDestroy(stop));
#endif

   return volume;
}
#endif

} // namespace levelSet
}
