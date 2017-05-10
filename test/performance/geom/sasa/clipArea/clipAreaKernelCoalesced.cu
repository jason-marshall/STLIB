/* -*- C -*- */

#include "clipAreaKernel.h"

#include <assert.h>
#include <stdio.h>

#if 0
__device__
bool
isInside(const Ball ball, const float3 p) {
   return (ball.center.x - p.x) * (ball.center.x - p.x) +
      (ball.center.y - p.y) * (ball.center.y - p.y) +
      (ball.center.z - p.z) * (ball.center.z - p.z) < ball.squaredRadius;
}
#endif

__device__
bool
isInside(const float3 center, const float3 p) {
   return (center.x - p.x) * (center.x - p.x) +
      (center.y - p.y) * (center.y - p.y) +
      (center.z - p.z) * (center.z - p.z) < 1;
}

// Calculate the number of active points after clipping.
__global__
void
clip(const float3* referenceMesh, const unsigned meshSize,
     const float3* centers, unsigned centersSize,
     const unsigned* delimiters, const float3* clippingCenters,
     unsigned* activeCounts) {
   // The n_th thread block works on the n_th ball.
   const float3 center = centers[blockIdx.x];
   // Count the clipped points in parallel.
   __shared__ unsigned clippedCounts[ThreadsPerBlock];
   clippedCounts[threadIdx.x] = 0;
   float3 p;
   bool clipped;
   const unsigned begin = delimiters[blockIdx.x];
   const unsigned end = delimiters[blockIdx.x + 1];
   if (end - begin <= ThreadsPerBlock) {
      // Copy the clipping centers into shared memory.
      __shared__ float3 cc[ThreadsPerBlock];
      const unsigned size = end - begin;
      if (threadIdx.x < size) {
         cc[threadIdx.x] = clippingCenters[begin + threadIdx.x];
      }
      __syncthreads();
      __shared__ float referenceMeshBlock[3 * ThreadsPerBlock];
      //for (unsigned i = threadIdx.x; i < meshSize; i += blockDim.x) {
      const float* points = &referenceMesh[0].x;
      for (unsigned i = 0; i != meshSize; i += blockDim.x) {
#if 0
         // Translate the reference mesh point to lie on the ball.
         p = referenceMesh[i];
         p.x += center.x;
         p.y += center.y;
         p.z += center.z;
#else
         // Copy a block of the mesh points into shared memory.
         unsigned n = 3 * i + threadIdx.x;
         referenceMeshBlock[threadIdx.x] = points[n];
         referenceMeshBlock[threadIdx.x + ThreadsPerBlock]
            = points[n + ThreadsPerBlock];
         referenceMeshBlock[threadIdx.x + 2 * ThreadsPerBlock]
            = points[n + 2 * ThreadsPerBlock];
         __syncthreads();
         // Translate the reference mesh point to lie on the ball.
         n = 3 * threadIdx.x;
         p.x = referenceMeshBlock[n] + center.x;
         p.y = referenceMeshBlock[n+1] + center.y;
         p.z = referenceMeshBlock[n+2] + center.z;
#endif
         // Determine if the point is clipped.
         clipped = false;
         for (unsigned j = 0; j != size; ++j) {
            clipped = clipped | isInside(cc[j], p);
         }
         clippedCounts[threadIdx.x] += clipped;
      }
   }
   else {
      for (unsigned i = threadIdx.x; i < meshSize; i += blockDim.x) {
         // Translate the reference mesh point to lie on the ball.
         p = referenceMesh[i];
         p.x += center.x;
         p.y += center.y;
         p.z += center.z;
         // Determine if the point is clipped.
         clipped = false;
         for (unsigned j = begin; j != end; ++j) {
            clipped = clipped || isInside(clippingCenters[j], p);
         }
         clippedCounts[threadIdx.x] += clipped;
      }
   }
   // Calculate the total number of clipped points in the first thread.
   __syncthreads();
   if (threadIdx.x == 0) {
      unsigned count = 0;
      for (unsigned i = 0; i != ThreadsPerBlock; ++i) {
         count += clippedCounts[i];
      }
      activeCounts[blockIdx.x] = meshSize - count;
   }
}

extern "C"
void
clipKernel(const float3* referenceMesh, const unsigned meshSize, 
           const float3* centers, unsigned centersSize, 
           const unsigned* delimiters, const float3* clippingCenters,
           unsigned* activeCounts) {
   // The mesh size must be a multiple of the number of threads per block.
   assert(meshSize % ThreadsPerBlock == 0);
   // We choose the number of blocks to be the number of balls. The maximum
   // number of blocks is 65535.
   assert(centersSize < (1 << 16 - 1));
   // Invoke the kernel.
   clip<<<centersSize, ThreadsPerBlock>>>
      (referenceMesh, meshSize, centers, centersSize, delimiters,
       clippingCenters, activeCounts);
}
