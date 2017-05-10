/* -*- C++ -*- */

#include "stlib/levelSet/exteriorVolumeCuda.h"

#ifndef __CUDA_ARCH__
#include "stlib/levelSet/gridCuda.h"
#include "stlib/levelSet/ballsCuda.h"
#include <cassert>
#endif

namespace stlib
{
namespace levelSet {


// This is used to compute the volume from the signed distance.
// Return the function
//        1,             if x <= -1,
// f(x) = 0.5 - 0.5 * x, if -1 < x < 1,
//        0,             if x >= 1.
__device__
float
volumeFractionFromDistance(const float x) {
   if (x <= -1.f) {
      return 1.f;
   }
   if (x >= 1.f) {
      return 0.f;
   }
   return 0.5f - 0.5f * x;
}


__device__
void
reducePatch(float* x, const unsigned tid) {
   unsigned size = PatchExtent * PatchExtent * PatchExtent;
   while (size > 64) {
      size /= 2;
      if (tid < size) {
         x[tid] += x[tid + size];
      }
      __syncthreads();
   }
   // No need to synchronize after this as long as we access the memory via
   // a volatile.
   if (tid < 32) {
      volatile float* v = x;
      v[tid] += v[tid + 32];
      v[tid] += v[tid + 16];
      v[tid] += v[tid + 8];
      v[tid] += v[tid + 4];
      v[tid] += v[tid + 2];
      v[tid] += v[tid + 1];
   }
}


// p is an array of length PatchExtent^3.
// m is an array of length PatchExtent^3 / 2.
__device__
float
maxPatchValue(const float* p, float* m, const unsigned tid) {
   unsigned size = PatchExtent * PatchExtent * PatchExtent / 2;
   // Special case for the first step.
   if (tid < size) {
      m[tid] = max(p[tid], p[tid + size]);
   }
   while (size > 16) {
      size /= 2;
      if (tid < size) {
         m[tid] = max(m[tid], m[tid + size]);
      }
      __syncthreads();
   }
   // No need to synchronize here because a half warp executes simultaneously.
   if (tid < 8) {
      m[tid] = max(m[tid], m[tid + 8]);
   }
   if (tid < 4) {
      m[tid] = max(m[tid], m[tid + 4]);
   }
   if (tid < 2) {
      m[tid] = max(m[tid], m[tid + 2]);
   }
   if (tid < 1) {
      m[tid] = max(m[tid], m[tid + 1]);
   }
   __syncthreads();
   return m[0];
}


// Note: Storing the balls in constant memory does not significantly affect
// performance.
/*
  - patchVolume: The accumulated volume for the grid points in this patch.
*/
__device__
void
exteriorVolumeKernel(const float3 lowerCorner,
                     const float spacing,
                     const float4* balls,
                     unsigned numInfluencingBalls,
                     const unsigned* ballIndices,
                     float* patchVolume) {
   // Convert the multi-index to a single offset.
   const unsigned tid = threadIdx.x + threadIdx.y * blockDim.x +
      threadIdx.z * blockDim.x * blockDim.y;
   // Calculate the Cartesian location of the grid point.
   const float3 p = {lowerCorner.x + spacing * threadIdx.x,
                     lowerCorner.y + spacing * threadIdx.y,
                     lowerCorner.z + spacing * threadIdx.z};

   // Shared memory for the grid points in this patch.
   __shared__ float g[PatchExtent * PatchExtent * PatchExtent];
#if 0
   __shared__ float maxValues[PatchExtent * PatchExtent * PatchExtent / 2];
#endif

   // Initialize the grid value to infinity.
   g[tid] = 1.f/0.f;
   // Find the minimum distance over the set of balls.
   float d;
   // Shared memory for loading balls 32 at a time.
   __shared__ float4 b[32];
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
      if (g[tid] > 0) {
         // Process the loaded balls.
         for (unsigned i = 0; i != block; ++i) {
            d = sqrt((p.x - b[i].x) * (p.x - b[i].x) +
                     (p.y - b[i].y) * (p.y - b[i].y) +
                     (p.z - b[i].z) * (p.z - b[i].z)) - b[i].w;
            if (d < g[tid]) {
               g[tid] = d;
            }
         }
      }
      // CONTINUE: Investigate breaking when the distance is negative.
#if 0
      // If all of the points are negative, we can stop processing this patch.
      if (maxPatchValue(g, maxValues, tid) <= 0) {
         break;
      }
#endif
      ballIndices += block;
      numInfluencingBalls -= block;
   }
   // Reverse the sign to get the exterior.
   g[tid] = -g[tid];

#if 0
   // Constants used for computing volume.
   // The content of a voxel.
   const float voxelVolume = spacing * spacing * spacing;
   // The radius of the ball that has the same content as a voxel.
   // 4 * pi r^3 / 3 = dx^3
   const float inverseVoxelRadius = 1.f / (0.6203504908994001f * spacing);
   // Compute the volume at each grid point.
   g[tid] = voxelVolume * 
      volumeFractionFromDistance(g[tid] * inverseVoxelRadius);
#else
   if (g[tid] < 0) {
      g[tid] = spacing * spacing * spacing;
   }
   else {
      g[tid] = 0;
   }
#endif
   __syncthreads();

   // Sum the volumes to get the total for the patch.
   reducePatch(g, tid);
   if (tid == 0) {
      *patchVolume = g[0];
   }
}


// Select a single patch using the block index. Then call a kernel for that
// patch.
__global__
void
exteriorVolumeKernel(const unsigned numRefined,
                     const uint3* indices,
                     const float3 lowerCorner,
                     const float spacing,
                     const float4* balls,
                     const unsigned* ballIndexOffsets,
                     const unsigned* packedBallIndices,
                     float* patchVolumes) {
   // Convert the 2-D block index into a single patch index.
   const unsigned i = blockIdx.x + blockIdx.y * gridDim.x;
   if (i >= numRefined) {
      return;
   }
   const unsigned begin = ballIndexOffsets[i];
   const unsigned end = ballIndexOffsets[i+1];
   const float3 patchLowerCorner =
      {lowerCorner.x + PatchExtent * indices[i].x * spacing,
       lowerCorner.y + PatchExtent * indices[i].y * spacing,
       lowerCorner.z + PatchExtent * indices[i].z * spacing};
   exteriorVolumeKernel(patchLowerCorner, spacing, balls,
                          end - begin, &packedBallIndices[begin],
                          patchVolumes + i);
}


#ifndef __CUDA_ARCH__
float
exteriorVolumeCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<bool>& isActive,
 const std::vector<geom::Ball<float, 3> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies) {
   const std::size_t numRefined = dependencies.getNumberOfArrays();
   // CONTINUE REMOVE
   assert(numRefined == std::count(isActive.begin(), isActive.end(), true));

   // Allocate device memory for the active patch indices.
   uint3* indicesDev;
   allocateGridIndicesCuda(grid, numRefined, isActive, &indicesDev);

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
   
   // Allocate device memory for the patch volumes.
   float* patchVolumesDev;
   CUDA_CHECK(cudaMalloc((void**)&patchVolumesDev, numRefined * sizeof(float)));

   const float3 lowerCorner = {grid.lowerCorner[0], grid.lowerCorner[1],
                               grid.lowerCorner[2]};
   // Use a 2-D grid of blocks. Because the number of active patches may 
   // exceed the maximum allowed single grid dimension.
   cudaDeviceProp prop;
   CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
   const std::size_t Len = prop.maxGridSize[0];
   const dim3 GridDim(std::min(numRefined, Len), (numRefined + Len - 1) / Len);
   // A thread for each patch grid point.
   const dim3 ThreadsPerBlock(PatchExtent, PatchExtent, PatchExtent);
   // Launch the kernel.
   exteriorVolumeKernel<<<GridDim,ThreadsPerBlock>>>
      (numRefined, indicesDev, lowerCorner, grid.spacing, ballsDev,
       ballIndexOffsetsDev, packedBallIndicesDev, patchVolumesDev);
   // Copy the patch volumes data back to the host.
   std::vector<float> patchVolumes(numRefined);
   CUDA_CHECK(cudaMemcpy(&patchVolumes[0], patchVolumesDev,
                         patchVolumes.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));

   // Free the device memory.
   CUDA_CHECK(cudaFree(indicesDev));
   CUDA_CHECK(cudaFree(ballsDev));
   CUDA_CHECK(cudaFree(ballIndexOffsetsDev));
   CUDA_CHECK(cudaFree(packedBallIndicesDev));
   CUDA_CHECK(cudaFree(patchVolumesDev));

   // Return the total volume.
   return std::accumulate(patchVolumes.begin(), patchVolumes.end(), float(0));
}
#endif


} // namespace levelSet
}
