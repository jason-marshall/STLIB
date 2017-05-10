/* -*- C++ -*- */

#include "stlib/levelSet/solventExcludedCavitiesPosCuda.h"

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
const unsigned NumPointsOnSphere = 1024;
const unsigned NumPointsInBlock = 512;
}

__constant__ float3 pointsOnSphere[NumPointsOnSphere];


// CONTINUE Write a version for __CUDA_ARCH__ >= 200.
// Return true if the solvent does not intersect any of the influencing balls.
// c The center of the probe.
// b The array of influencing balls.
__device__
bool
isProbePlacementValid(const float3 c,
                      const float probeRadius,
                      const float4* b) {
   __shared__ unsigned doesIntersect[32];
   doesIntersect[threadIdx.x] = 0;
   for (unsigned offset = 0; offset != MaxInfluencingBalls;
        offset += NumThreads) {
      doesIntersect[threadIdx.x] |=
         (b[threadIdx.x + offset].x - c.x) * (b[threadIdx.x + offset].x - c.x) +
         (b[threadIdx.x + offset].y - c.y) * (b[threadIdx.x + offset].y - c.y) +
         (b[threadIdx.x + offset].z - c.z) * (b[threadIdx.x + offset].z - c.z) <
         (1.f - 10.f * __FLT_EPSILON__) *
         (b[threadIdx.x + offset].w + probeRadius) *
         (b[threadIdx.x + offset].w + probeRadius);
   }
   volatile unsigned* v = doesIntersect;
   if (threadIdx.x < 16) {
      v[threadIdx.x] |= v[threadIdx.x + 16];
      v[threadIdx.x] |= v[threadIdx.x + 8];
      v[threadIdx.x] |= v[threadIdx.x + 4];
      v[threadIdx.x] |= v[threadIdx.x + 2];
      v[threadIdx.x] |= v[threadIdx.x + 1];
   }
   return ! doesIntersect[0];
}


// Return true if the solvent intersects any of the influencing balls.
// c The center of the probe.
// b The array of influencing balls.
__device__
bool
doesIntersect(const float3 c,
              const float probeRadius,
              const unsigned numInfluencingBalls,
              const float4* b) {
   bool intersect = false;
   for (unsigned i = 0; i != numInfluencingBalls; ++i) {
      intersect |=
         (b[i].x - c.x) * (b[i].x - c.x) +
         (b[i].y - c.y) * (b[i].y - c.y) +
         (b[i].z - c.z) * (b[i].z - c.z) <
         (1.f - 10.f * __FLT_EPSILON__) *
         (b[i].w + probeRadius) * (b[i].w + probeRadius);
   }
   return intersect;
}


// Return true if the solvent intersects any of the influencing balls.
// c The center of the probe.
// b The array of influencing balls.
__device__
bool
doesIntersect(const float3 c,
              const float probeRadius,
              const unsigned numInfluencingBalls,
              const float4* b, const char* mayClip) {
   bool intersect = false;
   for (unsigned i = 0; i != numInfluencingBalls; ++i) {
      if (mayClip[i]) {
         intersect |= 
            (b[i].x - c.x) * (b[i].x - c.x) +
            (b[i].y - c.y) * (b[i].y - c.y) +
            (b[i].z - c.z) * (b[i].z - c.z) <
            (b[i].w + probeRadius) * (b[i].w + probeRadius);
      }
   }
   return intersect;
}


__device__
unsigned
countPositive(const float* g) {
   __shared__ unsigned counts[NumThreads];
   counts[threadIdx.x] = 0;
   for (unsigned i = 0; i != 2 * PatchExtent; ++i) {
      counts[threadIdx.x] += g[i] > 0.f;
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


__device__
bool
hasPositive(const float* g) {
#if __CUDA_ARCH__ >= 200
   bool positive = false;
   for (unsigned i = 0; i != 2 * PatchExtent; ++i) {
      positive |= g[i] > 0.f;
   }
   return any(positive);
#else
   __shared__ bool positive[NumThreads];
   positive[threadIdx.x] = false;
   for (unsigned i = 0; i != 2 * PatchExtent; ++i) {
      positive[threadIdx.x] |= g[i] > 0.f;
   }
   // No need to synch here because we use 32 threads.
   // No need to synchronize in between as long as we access the memory via
   // a volatile.
   if (threadIdx.x < 16) {
      volatile bool* v = positive;
      v[threadIdx.x] |= v[threadIdx.x + 16];
      v[threadIdx.x] |= v[threadIdx.x + 8];
      v[threadIdx.x] |= v[threadIdx.x + 4];
      v[threadIdx.x] |= v[threadIdx.x + 2];
      v[threadIdx.x] |= v[threadIdx.x + 1];
   }
   return positive[0];
#endif
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
solventExcludedCavitiesKernel(const float3 lowerCorner,
                              const float spacing,
                              const float probeRadius,
                              const float4* balls,
                              const unsigned numInfluencingBalls,
                              const unsigned* ballIndices,
                              unsigned* positiveCount) {
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
   // The threshold for computing distance for a probe is the radius of
   // the patch plus the probe radius. If the probe is farther than this
   // from the center of the patch, the probe does not touch the patch and
   // there is no need to compute distances. We also use this in determining
   // for which balls to compute the distance.
   const float threshold = (PatchExtent - 1) * 0.5f * spacing * sqrt(3.f) +
      probeRadius;
   const float threshold2 = threshold * threshold;

   // Compute the distances to the influencing balls. Here, we need to work
   // with the Euclidean distance because we will check if each point could
   // be covered by an unobstructed probe.
   float d;
   float dyz;
   // Loop over the influencing balls.
   for (int j = 0; j != numInfluencingBalls; ++j) {
      // If the influencing ball is not close to the patch, we don't
      // need to compute the distances. (We need to compute the
      // distance up to a distance of the probe radius.)  Note that
      // this calculation is duplicated, however, it isn't any faster
      // to spread this distance calculation across the threads.
      if ((patchCenter.x - b[j].x) * (patchCenter.x - b[j].x) +
          (patchCenter.y - b[j].y) * (patchCenter.y - b[j].y) +
          (patchCenter.z - b[j].z) * (patchCenter.z - b[j].z) >=
          (threshold + b[j].w) * (threshold + b[j].w)) {
         continue;
      }

      // Loop over the grid points in the first column.
      p.z = z0;
      dyz = (p.y - b[j].y) * (p.y - b[j].y) + (p.z - b[j].z) * (p.z - b[j].z);
      p.x = lowerCorner.x;
      for (int i = 0; i != PatchExtent; ++i) {
         d = sqrt((p.x - b[j].x) * (p.x - b[j].x) + dyz) - b[j].w;
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
         d = sqrt((p.x - b[j].x) * (p.x - b[j].x) + dyz) - b[j].w;
         if (d < g[i]) {
            g[i] = d;
         }
         p.x += spacing;
      }
   }
   // If the distance is greater than the probe radius, then the point
   // is covered by an unobstructed probe. Thus, the distance is negative.
   for (int i = 0; i != 2 * PatchExtent; ++i) {
      if (g[i] >= probeRadius) {
         g[i] = -1.f;
      }
   }

   // If all of the points are negative, we don't need to compute the distances
   // to the probes.
   if (! hasPositive(g)) {
      return;
   }

   // Whether the probe placement is valid, i.e. if it is close to the patch
   // and it does not intersect an influencing ball. Use char instead of 
   // bool to save space.
   __shared__ char isValid[NumPointsInBlock];
   // The indices of the valid points.
   __shared__ unsigned validIndices[NumPointsInBlock];
   // The probe center.
   float3 c;
   // Loop over the influencing balls.
   for (int i = 0; i != numInfluencingBalls; ++i) {
      // If the influencing ball is further than probeRadius from the patch,
      // we do not need to clip by any of the probes on its surface. This 
      // clipping has already been done when we clipped by unobstructed 
      // probes.
      if ((patchCenter.x - b[i].x) * (patchCenter.x - b[i].x) +
          (patchCenter.y - b[i].y) * (patchCenter.y - b[i].y) +
          (patchCenter.z - b[i].z) * (patchCenter.z - b[i].z) >=
          (threshold + b[i].w) * (threshold + b[i].w)) {
         continue;
      }

      // Note that it does not help performance to determine the subset 
      // of the influencing balls that may clip the probes for the current ball.

      // Separate the points on the sphere into blocks. This decreases the 
      // required amount of shared memory.
      for (unsigned block = 0; block != NumPointsOnSphere;
           block += NumPointsInBlock) {
         // Loop over the points on the sphere. When determining which probe
         // placements are valid, process NumThreads points at a time.
         for (unsigned offset = 0; offset != NumPointsInBlock;
              offset += NumThreads) {
            const unsigned j = offset + tid;
            isValid[j] = 1;
            validIndices[j] = block;
            // Position the probe.
            const unsigned p = block + j;
            c.x = b[i].x + (b[i].w + probeRadius) * pointsOnSphere[p].x;
            c.y = b[i].y + (b[i].w + probeRadius) * pointsOnSphere[p].y;
            c.z = b[i].z + (b[i].w + probeRadius) * pointsOnSphere[p].z;
            // First note the probe positions that are too far from the patch to
            // affect it.
            d = (patchCenter.x - c.x) * (patchCenter.x - c.x) +
               (patchCenter.y - c.y) * (patchCenter.y - c.y) +
               (patchCenter.z - c.z) * (patchCenter.z - c.z);
            if (d >= threshold2) {
               isValid[j] = 0;
            }
         }

#if 0
         // Compute the valid indices with a duplicated calculation across all 
         // threads.
         unsigned numValid = 0;
         for (unsigned j = 0; j != NumPointsInBlock; ++j) {
            // Note that doing this without branching would be slower.
            if (isValid[j]) {
               validIndices[numValid++] = block + j;
            }
         }
#else
         // Compute the valid indices. Parallel algorithm.
         unsigned numValid = 0;
         {
            __shared__ unsigned validCounts[NumThreads];
            validCounts[tid] = 0;
            const unsigned NumPerThread = NumPointsInBlock / NumThreads;
            const unsigned offset = tid * NumPerThread;
            for (unsigned j = offset; j != (tid + 1) * NumPerThread; ++j) {
               if (isValid[j]) {
                  validIndices[offset + validCounts[tid]++] = block + j;
               }
            }
            for (unsigned j = 0; j != NumThreads; ++j) {
               for (unsigned k = 0; k != validCounts[j]; ++k) {
                  validIndices[numValid++] = validIndices[j * NumPerThread + k];
               }
            }
         }
#endif
         // Now validIndices[0..numValid-1] are the valid indices.

         for (unsigned offset = 0; offset < numValid; offset += NumThreads) {
            const unsigned j = validIndices[offset + tid];
            // Position the probe.
            c.x = b[i].x + (b[i].w + probeRadius) * pointsOnSphere[j].x;
            c.y = b[i].y + (b[i].w + probeRadius) * pointsOnSphere[j].y;
            c.z = b[i].z + (b[i].w + probeRadius) * pointsOnSphere[j].z;
            // Check if the probe placement is valid, i.e. if it does not
            // intersect any of the influencing balls.
            if (doesIntersect(c, probeRadius, numInfluencingBalls, b)) {
               isValid[j - block] = 0;
            }
         }

         // Loop over the points on the sphere. 
         for (unsigned vi = 0; vi != numValid; ++vi) {
            const unsigned j = validIndices[vi];
            if (! isValid[j - block]) {
               continue;
            }
            // Position the probe.
            c.x = b[i].x + (b[i].w + probeRadius) * pointsOnSphere[j].x;
            c.y = b[i].y + (b[i].w + probeRadius) * pointsOnSphere[j].y;
            c.z = b[i].z + (b[i].w + probeRadius) * pointsOnSphere[j].z;
            // When we loop over the columns, each thread is working on a
            // different column. Here we will work with the power
            // distance since all we need is the sign of the distance.
            // Loop over the grid points in the first column.
            p.z = z0;
            dyz = (p.y - c.y) * (p.y - c.y) +
               (p.z - c.z) * (p.z - c.z) - probeRadius * probeRadius;
            p.x = lowerCorner.x;
            for (int k = 0; k != PatchExtent; ++k) {
               d = (p.x - c.x) * (p.x - c.x) + dyz;
               if (d < g[k]) {
                  g[k] = d;
               }
               p.x += spacing;
            }
            // Loop over the grid points in the second column.
            p.z = z1;
            dyz = (p.y - c.y) * (p.y - c.y) +
               (p.z - c.z) * (p.z - c.z) - probeRadius * probeRadius;
            p.x = lowerCorner.x;
            for (int k = PatchExtent; k != 2 * PatchExtent; ++k) {
               d = (p.x - c.x) * (p.x - c.x) + dyz;
               if (d < g[k]) {
                  g[k] = d;
               }
               p.x += spacing;
            }
         }

#if __CUDA_ARCH__ >= 200
         // CONTINUE: Consider moving outside the block loop to test less
         // frequently.
         // This test hurts performance for __CUDA_ARCH__ < 200. But slightly
         // helps performance otherwise.
         // If we updated the distance by calculating the distance to one or
         // more probes.
         if (numValid != 0) {
            // If all of the points are negative, we don't need to compute the
            // distances to the rest of the probes.
            if (! hasPositive(g)) {
               return;
            }
         }
#endif
      }
   }

   const unsigned count = countPositive(g);
   if (tid == 0) {
      atomicAdd(positiveCount, count);
   }
}


// Select a single patch using the block index. Then call a kernel for that
// patch.
__global__
void
solventExcludedCavitiesKernel(const unsigned yExtents,
                              const float3 lowerCorner,
                              const float spacing,
                              const float probeRadius,
                              const float4* balls,
                              const unsigned* ballIndexOffsets,
                              const unsigned* packedBallIndices,
                              unsigned* positiveCount) {
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
   solventExcludedCavitiesKernel(patchLowerCorner, spacing, probeRadius, balls,
                                 end - begin, &packedBallIndices[begin],
                                 positiveCount);
}


#ifndef __CUDA_ARCH__
//! Distribute points on a sphere with the golden section spiral algorithm.
/*!
  http://cgafaq.info/wiki/Evenly_distributed_points_on_sphere
*/
void
distributePointsOnSphereWithGoldenSectionSpiral(std::vector<float3>* points) {
   // Check the singular case.
   if (points->size() == 0) {
      return;
   }
   const double Delta = numerical::Constants<double>::Pi() *
                        (3. - std::sqrt(5.));
   double longitude = 0;
   double dz = 2.0 / points->size();
   double z = 1. - dz / 2;
   double r;
   for (std::size_t i = 0; i != points->size(); ++i) {
      r = std::sqrt(1. - z * z);
      (*points)[i].x = r * std::cos(longitude);
      (*points)[i].y = r * std::sin(longitude);
      (*points)[i].z = z;
      z -= dz;
      longitude += Delta;
   }
}
#endif


#ifndef __CUDA_ARCH__
float
solventExcludedCavitiesPosCuda
(const GridGeometry<3, PatchExtent, float>& grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const float probeRadius,
 const container::StaticArrayOfArrays<unsigned>& dependencies) {
   const std::size_t numPatches = product(grid.gridExtents);
   assert(dependencies.getNumberOfArrays() == numPatches);
   // The number of influencing balls for each patch is limited.
   for (std::size_t i = 0; i != dependencies.getNumberOfArrays(); ++i) {
      assert(dependencies.size(i) <= MaxInfluencingBalls);
   }
   
   // Initialize the constant memory.
   {
      std::vector<float3> p(NumPointsOnSphere);
      distributePointsOnSphereWithGoldenSectionSpiral(&p);
      CUDA_CHECK(cudaMemcpyToSymbol(pointsOnSphere, &p[0],
                                    NumPointsOnSphere * sizeof(float3)));
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
   
   // Allocate device memory for the count of positive grid points.
   unsigned* positiveCountDev;
   CUDA_CHECK(cudaMalloc((void**)&positiveCountDev, sizeof(unsigned)));
   CUDA_CHECK(cudaMemset(positiveCountDev, 0, sizeof(unsigned)));

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
   solventExcludedCavitiesKernel<<<GridDim,NumThreads>>>
      (grid.gridExtents[1], lowerCorner, grid.spacing, probeRadius, ballsDev,
       ballIndexOffsetsDev, packedBallIndicesDev, positiveCountDev);
   // Copy the count of positive grid points back to the host.
   unsigned positiveCount;
   CUDA_CHECK(cudaMemcpy(&positiveCount, positiveCountDev, sizeof(unsigned),
                         cudaMemcpyDeviceToHost));

   // Free the device memory.
   CUDA_CHECK(cudaFree(ballsDev));
   CUDA_CHECK(cudaFree(ballIndexOffsetsDev));
   CUDA_CHECK(cudaFree(packedBallIndicesDev));
   CUDA_CHECK(cudaFree(positiveCountDev));

   // Return the total volume.
   return positiveCount * grid.spacing * grid.spacing * grid.spacing;
}
#endif


} // namespace levelSet
}
