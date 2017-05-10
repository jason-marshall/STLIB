/* -*- C++ -*- */

#include "stlib/levelSet/positiveDistanceCuda.h"

#ifndef __CUDA_ARCH__
#include "stlib/levelSet/gridCuda.h"
#include "stlib/levelSet/ballsCuda.h"
#include <cassert>
#endif

namespace stlib
{
namespace levelSet {


// Note: Storing ballsSquared in constant memory does not significantly affect
// performance.
__device__
void
positiveDistanceKernel(float* patch,
                       const float3 lowerCorner,
                       const float spacing,
                       const float4* balls,
                       unsigned numInfluencingBalls,
                       const unsigned* ballIndices) {
   // Calculate the Cartesian location of the grid point.
   const float3 p = {lowerCorner.x + spacing * threadIdx.x,
                     lowerCorner.y + spacing * threadIdx.y,
                     lowerCorner.z + spacing * threadIdx.z};
   // Initialize the grid value to infinity.
   float g = 1./0;
   // Find the minimum power distance over the set of balls.
   float d;
   __shared__ float4 b[32];
   // Convert the multi-index to a single offset.
   const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
   unsigned block;
   while (numInfluencingBalls != 0) {
      block = 32;
      if (numInfluencingBalls < 32) {
         block = numInfluencingBalls;
      }
      // Load the balls into shared memory.
      if (tid < block) {
         // Even though these are uncoalesced, this is an insignificant cost.
         b[tid] = balls[ballIndices[tid]];
      }
      __syncthreads();
      // Process the loaded balls.
      for (unsigned i = 0; i != block; ++i) {
         d = sqrt((p.x - b[i].x) * (p.x - b[i].x) +
                  (p.y - b[i].y) * (p.y - b[i].y) +
                  (p.z - b[i].z) * (p.z - b[i].z)) - b[i].w;
         if (d < g) {
            g = d;
         }
      }
      ballIndices += block;
      numInfluencingBalls -= block;
   }
   // Record the patch value. 
   patch[tid] = g;
}


// Select a single patch using the block index. Then call a kernel for that
// patch.
__global__
void
positiveDistanceKernel(const unsigned numRefined,
                       float* patches,
                       const uint3* indices,
                       const float3 lowerCorner,
                       const float spacing,
                       const float4* balls,
                       const unsigned* ballIndexOffsets,
                       const unsigned* packedBallIndices) {
   // Convert the 2-D block index into a single patch index.
   const unsigned i = blockIdx.x + blockIdx.y * gridDim.x;
   if (i >= numRefined) {
      return;
   }
   const unsigned begin = ballIndexOffsets[i];
   const unsigned end = ballIndexOffsets[i+1];
   const unsigned NumThreads = PatchExtent * PatchExtent * PatchExtent;
   const float3 patchLowerCorner =
      {lowerCorner.x + PatchExtent * indices[i].x * spacing,
       lowerCorner.y + PatchExtent * indices[i].y * spacing,
       lowerCorner.z + PatchExtent * indices[i].z * spacing};
   positiveDistanceKernel(patches + i * NumThreads,
                          patchLowerCorner, spacing, balls,
                          end - begin, &packedBallIndices[begin]);
}


#ifndef __CUDA_ARCH__
void
positiveDistanceCuda(Grid<float, 3, PatchExtent>* grid,
                     const std::vector<geom::Ball<float, 3> >& balls,
                     float offset, float maxDistance) {
   // Dispense with the trivial case.
   if (grid->empty()) {
      return;
   }

   // Determine the patch/ball dependencies.
   container::StaticArrayOfArrays<unsigned> dependencies;
   positiveDistanceDependencies(*grid, balls, offset, maxDistance,
                                &dependencies);

   // Refine the appropriate patches and set the rest to have an unknown
   // distance.
   grid->refine(dependencies);
   const std::size_t numRefined = grid->numRefined();

   // Allocate device memory for the refined patches and their indices.
   float* patchesDev;
   uint3* indicesDev;
   allocateGridCuda(*grid, numRefined, &patchesDev, &indicesDev);

   // Allocate device memory for the balls and copy the memory.
   float4* ballsDev;
   CUDA_CHECK(cudaMalloc((void**)&ballsDev, balls.size() * sizeof(float4)));
   {
      std::vector<float4> buffer(balls.size());
      for (std::size_t i = 0; i != balls.size(); ++i) {
         buffer[i].x = balls[i].center[0];
         buffer[i].y = balls[i].center[1];
         buffer[i].z = balls[i].center[2];
         buffer[i].w = balls[i].radius + offset;
      }
      CUDA_CHECK(cudaMemcpy(ballsDev, &buffer[0],
                            buffer.size() * sizeof(float4),
                            cudaMemcpyHostToDevice));
   }

   // Allocate device memory for the ball index offsets and packed ball indices.
   // Copy the data to the device.
   unsigned* ballIndexOffsetsDev;
   unsigned* packedBallIndicesDev;
   allocateBallIndicesCuda(dependencies, numRefined, &ballIndexOffsetsDev,
                           &packedBallIndicesDev);
   
   const float3 lowerCorner = {grid->lowerCorner[0], grid->lowerCorner[1],
                               grid->lowerCorner[2]};
   // Use a 2-D grid of blocks. Because the number of refined patches may 
   // exceed the maximum allowed single grid dimension.
   cudaDeviceProp prop;
   CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
   const std::size_t Len = prop.maxGridSize[0];
   const dim3 GridDim(std::min(numRefined, Len), (numRefined + Len - 1) / Len);
   // A thread for each patch grid point.
   const dim3 ThreadsPerBlock(PatchExtent, PatchExtent, PatchExtent);
   // Launch the kernel.
   positiveDistanceKernel<<<GridDim,ThreadsPerBlock>>>
      (numRefined, patchesDev, indicesDev, lowerCorner, grid->spacing, ballsDev,
       ballIndexOffsetsDev, packedBallIndicesDev);
   // Copy the patch data back to the host.
   CUDA_CHECK(cudaMemcpy(grid->data(), patchesDev,
                         grid->numVertices() * sizeof(float),
                         cudaMemcpyDeviceToHost));

   // Free the device memory.
   CUDA_CHECK(cudaFree(patchesDev));
   CUDA_CHECK(cudaFree(indicesDev));
   CUDA_CHECK(cudaFree(ballsDev));
   CUDA_CHECK(cudaFree(ballIndexOffsetsDev));
   CUDA_CHECK(cudaFree(packedBallIndicesDev));
}
#endif


#ifndef __CUDA_ARCH__
void
positiveDistanceCuda
(const std::size_t numRefined,
 float* patchesDev,
 uint3* indicesDev,
 const float3 lowerCorner,
 const float spacing,
 const std::vector<geom::Ball<float, 3> >& balls,
 const float offset,
 const container::StaticArrayOfArrays<unsigned>& dependencies) {
   // Allocate device memory for the balls and copy the memory.
   float4* ballsDev;
   CUDA_CHECK(cudaMalloc((void**)&ballsDev, balls.size() * sizeof(float4)));
   {
      std::vector<float4> buffer(balls.size());
      for (std::size_t i = 0; i != balls.size(); ++i) {
         buffer[i].x = balls[i].center[0];
         buffer[i].y = balls[i].center[1];
         buffer[i].z = balls[i].center[2];
         buffer[i].w = balls[i].radius + offset;
      }
      CUDA_CHECK(cudaMemcpy(ballsDev, &buffer[0],
                            buffer.size() * sizeof(float4),
                            cudaMemcpyHostToDevice));
   }

   // Allocate device memory for the ball index offsets and packed ball indices.
   // Copy the data to the device.
   unsigned* ballIndexOffsetsDev;
   unsigned* packedBallIndicesDev;
   allocateBallIndicesCuda(dependencies, numRefined, &ballIndexOffsetsDev,
                           &packedBallIndicesDev);
   
   // Use a 2-D grid of blocks. Because the number of refined patches may 
   // exceed the maximum allowed single grid dimension.
   cudaDeviceProp prop;
   CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
   const std::size_t Len = prop.maxGridSize[0];
   const dim3 GridDim(std::min(numRefined, Len), (numRefined + Len - 1) / Len);
   // A thread for each patch grid point.
   const dim3 ThreadsPerBlock(PatchExtent, PatchExtent, PatchExtent);
   // Launch the kernel.
   positiveDistanceKernel<<<GridDim,ThreadsPerBlock>>>
      (numRefined, patchesDev, indicesDev, lowerCorner, spacing, ballsDev,
       ballIndexOffsetsDev, packedBallIndicesDev);

   // Free the device memory for the balls.
   CUDA_CHECK(cudaFree(ballsDev));
   CUDA_CHECK(cudaFree(ballIndexOffsetsDev));
   CUDA_CHECK(cudaFree(packedBallIndicesDev));
}
#endif


} // namespace levelSet
}
