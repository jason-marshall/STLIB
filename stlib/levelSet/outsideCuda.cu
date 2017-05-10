/* -*- C++ -*- */

#include "stlib/levelSet/outsideCuda.h"
#include "stlib/levelSet/cuda.h"
#include "stlib/cuda/check.h"

#ifndef __CUDA_ARCH__
// CONTINUE REMOVE
#include <cstdio>
#include <cassert>
#endif

namespace stlib
{
namespace levelSet {

// The leftmost bit in a byte is the most significant.
// With little-endian ordering, the first byte in a multi-byte integer
// is the least significant. (Little end first.)
// The bit ordering for a 32-bit unsigned integer is the following.
// 07 06 05 04 03 02 01 00 15 14 13 12 10 09 08 ...


__global__
void
setNegativePatchesKernel(const unsigned size, const uint3* indices,
                         const uint3 extents, unsigned char* sign) {
   // Convert the 2-D block index into a single array index.
   const unsigned arrayIndex = blockIdx.x + blockIdx.y * gridDim.x;
   if (arrayIndex >= size) {
      return;
   }
   // Select a single patch using the array index.
   const uint3 index = indices[arrayIndex];
   // Set the packed sign bits for this thread.
   sign[index.x + (index.y * PatchExtent + threadIdx.y) * extents.x +
        (index.z * PatchExtent + threadIdx.z) * extents.x * extents.y] = 0;
}


__global__
void
setSignKernel(const unsigned numRefined,
              const float* patches, const uint3* indices,
              const uint3 extents, unsigned char* sign) {
   // Convert the 2-D block index into a single patch index.
   const unsigned patchIndex = blockIdx.x + blockIdx.y * gridDim.x;
   if (patchIndex >= numRefined) {
      return;
   }
   // Select a single patch using the block index.
   const unsigned PatchSize = PatchExtent * PatchExtent * PatchExtent;
   const float* patch = patches + patchIndex * PatchSize;
   const uint3 index = indices[patchIndex];
   // Pack the sign information into an unsigned char.
   unsigned char s = 0;
   const unsigned offset = threadIdx.y * PatchExtent +
      threadIdx.z * PatchExtent * PatchExtent;
   for (unsigned i = PatchExtent; i-- != 0; ) {
      s <<= 1;
      s |= patch[offset + i] > 0;
   }
   sign[index.x + (index.y * PatchExtent + threadIdx.y) * extents.x +
        (index.z * PatchExtent + threadIdx.z) * extents.x * extents.y] = s;
}


__global__
void
markOutsideKernel(const unsigned numRefined,
                  float* patches, const uint3* indices, const uint3 extents,
                  const unsigned char* outside) {
   // Convert the 2-D block index into a single patch index.
   const unsigned patchIndex = blockIdx.x + blockIdx.y * gridDim.x;
   if (patchIndex >= numRefined) {
      return;
   }
   const float NegInf = -1. / 0.;
   // Select a single patch using the block index.
   const unsigned PatchSize = PatchExtent * PatchExtent * PatchExtent;
   float* patch = patches + patchIndex * PatchSize;
   const uint3 index = indices[patchIndex];
   // Unpack the outside information.
   const unsigned char out =
      outside[index.x + (index.y * PatchExtent + threadIdx.y) * extents.x +
        (index.z * PatchExtent + threadIdx.z) * extents.x * extents.y];
   unsigned char mask = 1;
   const unsigned offset = threadIdx.y * PatchExtent +
      threadIdx.z * PatchExtent * PatchExtent;
   for (unsigned i = 0; i != PatchExtent; ++i) {
      if (mask & out) {
         patch[offset + i] = NegInf;
      }
      mask <<= 1;
   }
}


__global__
void
sweepXLocalKernel(const unsigned length, const unsigned* sign,
                  unsigned* outside) {
   // The thread extents are (64, 1, 1).
   // Allow for a maximum of 64 * 32 = 2048 bits in the x direction.
   __shared__ unsigned out[64];
   
   if (threadIdx.x >= length) {
      return;
   }
   
   const unsigned offset = blockIdx.x * length +
      blockIdx.y * length * gridDim.x;
   // Load from global memory.
   out[threadIdx.x] = outside[offset + threadIdx.x];
   __syncthreads();
   const unsigned s = sign[offset + threadIdx.x];
   unsigned shifted;
   // Iterate 8 times. A good choice was determined experimentally.
   for (unsigned i = 0; i != 8; ++i) {
      // Shift the bits left and right.
      shifted = (out[threadIdx.x] >> 1) | (out[threadIdx.x] << 1);
      if (threadIdx.x != 0) {
         shifted |= (0x80000000 & out[threadIdx.x - 1]) >> 31;
      }
      if (threadIdx.x != length - 1) {
         shifted |= (1 & out[threadIdx.x + 1]) << 31;
      }
      out[threadIdx.x] |= shifted & s;
      // In order to ensure communication between the half-warps one would
      // synchronize the threads here. However, we forgo this for better
      // performance.
   }
   // Write to global memory.
   outside[offset + threadIdx.x] = out[threadIdx.x];
}


void
sweepXLocal(const uint3 byteExtents, const unsigned* sign, unsigned* outside) {
   // The length of a row. Convert the number of bytes to the number of
   // unsigned integers.
   const unsigned length = byteExtents.x / 4;
   const dim3 Block(byteExtents.y, byteExtents.z);
   sweepXLocalKernel<<<Block, 64>>>(length, sign, outside);
}


// Length is the size of a row, i.e. the number of 32-bit unsigned integers.
__global__
void
sweepX(const unsigned length, const unsigned* sign, unsigned* outside) {
   // The offset into the beginning of the rows.
   const unsigned offset = blockIdx.x * length +
      blockIdx.y * length * gridDim.x;

   // Positive direction.
   unsigned s, out, mask;
   unsigned isOut = 1;
   for (unsigned i = 0; i != length; ++i) {
      // The least significant bit.
      mask = 1;
      // Get the 32 sign and outside bits.
      s = sign[offset + i];
      out = outside[offset + i];
      // Shortcut if the whole block is outside.
      if (out == 0xFFFFFFFF) {
         isOut = 1;
      }
      // Shortcut if the whole block is inside.
      else if (s == 0) {
         isOut = 0;
      }
      else {
         for (unsigned j = 0; j != 32; ++j) {
            if (isOut) {
               out |= mask & s;
            }
            isOut = out & mask;
            mask <<= 1;
         }
         // Write the new outside bits.
         outside[offset + i] = out;
      }
   }

   // Negative direction.
   isOut = 1;
   for (unsigned i = length; i-- != 0; ) {
      // The most significant bit.
      mask = 0x80000000;
      // Get the 32 sign and outside bits.
      s = sign[offset + i];
      out = outside[offset + i];
      // Shortcut if the whole block is outside.
      if (out == 0xFFFFFFFF) {
         isOut = 1;
      }
      // Shortcut if the whole block is inside.
      else if (s == 0) {
         isOut = 0;
      }
      else {
         for (unsigned j = 0; j != 32; ++j) {
            if (isOut) {
               out |= mask & s;
            }
            isOut = out & mask;
            mask >>= 1;
         }
         // Write the new outside bits.
         outside[offset + i] = out;
      }
   }
}


__device__
void
sweepAdjacentRow(const unsigned length, const unsigned stride,
                 const unsigned* sign, unsigned* outside,
                 unsigned sourceOutside[64],
                 const unsigned offset, const unsigned i) {
   // Load the target.
   const unsigned targetSign = sign[offset + i * stride + threadIdx.x];
   unsigned targetOutside = outside[offset + i * stride + threadIdx.x];

   // Combine the shifted source bits.
   __syncthreads();
   unsigned s = sourceOutside[threadIdx.x];
   unsigned combinedOutside = s | (s >> 1) | (s << 1);
   if (threadIdx.x != 0) {
      combinedOutside |= (0x80000000 & sourceOutside[threadIdx.x - 1]) >> 31;
   }
   if (threadIdx.x != length - 1) {
      combinedOutside |= (1 & sourceOutside[threadIdx.x + 1]) << 31;
   }

   // Propagate.
   targetOutside |= combinedOutside & targetSign;

   // Write the new outside information.
   outside[offset + i * stride + threadIdx.x] = targetOutside;
   // Move the target to the source to avoid loading from global memory.
   sourceOutside[threadIdx.x] = targetOutside;
}


__global__
void
sweepAdjacent(const unsigned length, const unsigned width,
              const unsigned blockStride, const unsigned stride,
              const unsigned* sign, unsigned* outside) {
   // The thread extents are (64, 1, 1).
   // Allow for a maximum of 64 * 32 = 2048 bits in the x direction.
   __shared__ unsigned sourceOutside[64];
   
   if (threadIdx.x >= length) {
      return;
   }
   
   const unsigned offset = blockIdx.x * blockStride;

   // Positive direction.
   // Load the first source.
   outside[offset + threadIdx.x] = 0xFFFFFFFF;
   sourceOutside[threadIdx.x] = 0xFFFFFFFF;
   __syncthreads();
   for (unsigned i = 1; i != width; ++i) {
      sweepAdjacentRow(length, stride, sign, outside, sourceOutside,
                       offset, i);
   }

   // Negative direction.
   // Load the first source.
   outside[offset + (width - 1) * stride + threadIdx.x] = 0xFFFFFFFF;
   sourceOutside[threadIdx.x] = 0xFFFFFFFF;
   for (unsigned i = width - 2; i-- != 0; ) {
      sweepAdjacentRow(length, stride, sign, outside, sourceOutside,
                       offset, i);
   }
}


void
sweepY(const uint3 byteExtents, const unsigned* sign, unsigned* outside) {
   // The length of a row. Convert the number of bytes to the number of
   // unsigned integers.
   const unsigned length = byteExtents.x / 4;
   const unsigned width = byteExtents.y;
   // The offset into the beginning of the rows.
   const unsigned blockStride = length * width;
   const unsigned stride = length;
   sweepAdjacent<<<byteExtents.z, 64>>>(length, width, blockStride, stride,
                                        sign, outside);
}

void
sweepZ(const uint3 byteExtents, const unsigned* sign, unsigned* outside) {
   // The length of a row. Convert the number of bytes to the number of
   // unsigned integers.
   const unsigned length = byteExtents.x / 4;
   const unsigned width = byteExtents.z;
   // The offset into the beginning of the rows.
   const unsigned blockStride = length;
   const unsigned stride = length * byteExtents.y;
   sweepAdjacent<<<byteExtents.y, 64>>>(length, width, blockStride, stride,
                                        sign, outside);
}


__global__
void
sweepDiagonalKernel(const uint3 byteExtents, const unsigned* sign,
                    unsigned* outside) {
   // The length of a row. Convert the number of bytes to the number of
   // unsigned integers.
   const unsigned length = byteExtents.x / 4;
   if (threadIdx.x >= length) {
      return;
   }
   const int j = blockIdx.x + 1;
   const int k = blockIdx.y + 1;
   const unsigned strideK = length * byteExtents.y;

   // The thread extents are (64, 1, 1).
   // Allow for a maximum of 64 * 32 = 2048 bits in the x direction.
   __shared__ unsigned sourceOutside[64];
   
   // Load the target.
   const unsigned index = threadIdx.x + j * length + k * strideK;
   const unsigned targetSign = sign[index];
   unsigned targetOutside = outside[index];

   unsigned s;
   unsigned combinedOutside = 0;
   // For each diagonal direction.
   for (int y = -1; y <= 1; y += 2) {
      for (int z = -1; z <= 1; z += 2) {
         s = sourceOutside[threadIdx.x] =
            outside[threadIdx.x + (j + y) * length + (k + z) * strideK];
         __syncthreads();
         // Combine the shifted source bits.
         combinedOutside |= s | (s >> 1) | (s << 1);
         if (threadIdx.x != 0) {
            combinedOutside |=
               (0x80000000 & sourceOutside[threadIdx.x - 1]) >> 31;
         }
         if (threadIdx.x != length - 1) {
            combinedOutside |= (1 & sourceOutside[threadIdx.x + 1]) << 31;
         }
      }
   }

   // Propagate.
   targetOutside |= combinedOutside & targetSign;
   // Write the new outside information.
   outside[index] = targetOutside;
}


void
sweepDiagonal(const uint3 byteExtents, const unsigned* sign,
              unsigned* outside) {
   const dim3 Block(byteExtents.y - 2, byteExtents.z - 2);
   sweepDiagonalKernel<<<Block, 64>>>(byteExtents, sign, outside);
}


// Length is the size of a row, i.e. the number of 32-bit unsigned integers.
__global__
void
countBitsKernel(const uint3 byteExtents, const unsigned* outside,
                unsigned* sliceCounts) {
   // The block extents are (byteExtents.y, 1, 1).
   // The thread extents are (64, 1, 1).

   // The length of a row. Convert the number of bytes to the number of
   // unsigned integers.
   const unsigned length = byteExtents.x / 4;

   unsigned c = 0;
   if (threadIdx.x < length) {
      // The offset into the beginning of the rows.
      const unsigned offset = blockIdx.x * length;
      const unsigned stride = length * (byteExtents.y);

      unsigned v;
      for (unsigned i = 0; i != byteExtents.z; ++i) {
         v = outside[offset + i * stride + threadIdx.x];
         for (unsigned j = 0; j != 32; ++j, v >>= 1) {
            c += v & 1;
         }
      }
   }

   //
   // Accumulate the counts across the threads.
   //
   // Allow for a maximum of 64 * 32 = 2048 bits in the x direction.
   __shared__ unsigned counts[64];
   counts[threadIdx.x] = c;
   __syncthreads();
   if (threadIdx.x < 32) {
      counts[threadIdx.x] += counts[threadIdx.x + 32];
   }
   __syncthreads();
   // Note that the threads in a half-warp execute together so there is no 
   // need for further synchronization.
   for (unsigned offset = 16; offset; offset >>= 1) {
      if (threadIdx.x < offset) {
         counts[threadIdx.x] += counts[threadIdx.x + offset];
      }
   }
   // Write to global memory.
   sliceCounts[blockIdx.x] = counts[0];
}


#ifndef __CUDA_ARCH__
unsigned
countBits(const uint3 byteExtents, const unsigned* outside) {
   // Allocate memory for the sign array.
   unsigned* sliceCountsDev;
   CUDA_CHECK(cudaMalloc((void**)&sliceCountsDev,
                         byteExtents.y * sizeof(unsigned)));
   countBitsKernel<<<byteExtents.y, 64>>>(byteExtents, outside, sliceCountsDev);
   std::vector<unsigned> sliceCounts(byteExtents.y);
   // Copy the count data back to the host.
   CUDA_CHECK(cudaMemcpy(&sliceCounts[0], sliceCountsDev,
                         sliceCounts.size() * sizeof(unsigned),
                         cudaMemcpyDeviceToHost));
   // Free the device memory.
   CUDA_CHECK(cudaFree(sliceCountsDev));
   return std::accumulate(sliceCounts.begin(), sliceCounts.end(), 0);
}
#endif


#ifndef __CUDA_ARCH__
void
markOutsideAsNegativeInf
(const std::array<std::size_t, 3>& gridExtents,
 const std::size_t numRefined,
 float* patchesDev,
 const uint3* indicesDev,
 const std::vector<std::array<std::size_t, 3> >& negativePatches,
 std::vector<bool>* outsideAtLowerCorners) {
   assert(PatchExtent == 8);
   // Determine the grid extents for the bit arrays. Because we access the 
   // bits using unsigned integers, round up to a multiple of 4 bytes in the
   // first dimension.
   const uint3 byteExtents = 
      {(gridExtents[0] + 3) / 4 * 4,
       PatchExtent * gridExtents[1],
       PatchExtent * gridExtents[2]};
   const std::size_t numBytes = byteExtents.x * byteExtents.y * byteExtents.z;

   // Allocate memory for the sign array.
   unsigned* signDev;
   CUDA_CHECK(cudaMalloc((void**)&signDev, numBytes));
   // Start by marking the sign of the distance as positive. We do this because
   // the bit arrays may be larger than the grid in the x direction. Any 
   // extra is definitely positive (outside).
   CUDA_CHECK(cudaMemset(signDev, 0xFF, numBytes));

   cudaDeviceProp prop;
   CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
   const std::size_t MaxGridExtent = prop.maxGridSize[0];

   // First mark the patches that have all negative distances.
   {
      std::vector<uint3> buffer(negativePatches.size());
      for (std::size_t i = 0; i != buffer.size(); ++i) {
         buffer[i].x = negativePatches[i][0];
         buffer[i].y = negativePatches[i][1];
         buffer[i].z = negativePatches[i][2];
      }
      // Allocate memory for the negative patch indices.
      uint3* negativePatchesDev;
      CUDA_CHECK(cudaMalloc((void**)&negativePatchesDev,
                            buffer.size() * sizeof(uint3)));
      // Copy the memory.
      CUDA_CHECK(cudaMemcpy(negativePatchesDev, &buffer[0],
                            buffer.size() * sizeof(uint3),
                            cudaMemcpyHostToDevice));
      // Use a 2-D grid of blocks. Because the number of negative patches may 
      // exceed the maximum allowed single grid dimension.
      const dim3 GridDim(std::min(negativePatches.size(), MaxGridExtent),
                         (negativePatches.size() + MaxGridExtent - 1)
                         / MaxGridExtent);
      // A thread for each row in the array. That is, each thread sets a 
      // single unsigned char.
      const dim3 ThreadsPerBlock(1, PatchExtent, PatchExtent);
      setNegativePatchesKernel<<<GridDim, ThreadsPerBlock>>>
         (negativePatches.size(), negativePatchesDev, byteExtents,
          reinterpret_cast<unsigned char*>(signDev));

      // Free the device memory for the negative patch indices.
      CUDA_CHECK(cudaFree(negativePatchesDev));
   }

   // Then mark the grid points that are negative.
   {
      // Use a 2-D grid of blocks. Because the number of refined patches may 
      // exceed the maximum allowed single grid dimension.
      const dim3 GridDim(std::min(numRefined, MaxGridExtent),
                         (numRefined + MaxGridExtent - 1) / MaxGridExtent);
      // A thread for each row in the array. That is, each thread packs eight
      // values into an unsigned char.
      const dim3 ThreadsPerBlock(1, PatchExtent, PatchExtent);
      setSignKernel<<<GridDim,ThreadsPerBlock>>>
         (numRefined, patchesDev, indicesDev, byteExtents,
          reinterpret_cast<unsigned char*>(signDev));
   }

#if 0
   {
      // CONTINUE: REMOVE
      printf("sign\nbyteExtents.x = %d\n", byteExtents.x);
      std::vector<unsigned char> s(numBytes);
      CUDA_CHECK(cudaMemcpy(&s[0], signDev,
                            s.size() * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i != byteExtents.x; ++i) {
         printf("%X ", s[i]);
      }
      printf("\n");
      // Pick a point near the middle of the sphere.
      const std::size_t offset = 4 * 8 * byteExtents.x +
         4 * 8 * byteExtents.x * byteExtents.y;
      for (std::size_t i = 0; i != byteExtents.x; ++i) {
         printf("%X ", s[offset + i]);
      }
      printf("\n");
   }
#endif

   // Allocate memory for the outside array.
   unsigned* outsideDev;
   CUDA_CHECK(cudaMalloc((void**)&outsideDev, numBytes));
   CUDA_CHECK(cudaMemset(outsideDev, 0, numBytes));

   unsigned oldCount = 1;
   unsigned count = 0;
   while (count != oldCount) {
      // Sweep in the y directions.
      sweepY(byteExtents, signDev, outsideDev);
      // Sweep in the z directions.
      sweepZ(byteExtents, signDev, outsideDev);
      // Sweep in the x directions.
      sweepXLocal(byteExtents, signDev, outsideDev);
      sweepDiagonal(byteExtents, signDev, outsideDev);
      oldCount = count;
      // Counting the bits accounts for about 10% of the execution time.
      count = countBits(byteExtents, outsideDev);
      // CONTINUE REMOVE
      //printf("count = %d\n", count);
   }
   // Free the device memory for the sign of the distance.
   CUDA_CHECK(cudaFree(signDev));

#if 0
   {
      // CONTINUE: REMOVE
      printf("outside\nbyteExtents.x = %d\n", byteExtents.x);
      std::vector<unsigned char> s(numBytes);
      CUDA_CHECK(cudaMemcpy(&s[0], outsideDev,
                            s.size() * sizeof(unsigned char),
                            cudaMemcpyDeviceToHost));
      for (std::size_t i = 0; i != byteExtents.x; ++i) {
         printf("%X ", s[i]);
      }
      printf("\n");
      // Pick a point near the middle of the sphere.
      const std::size_t offset = 4 * 8 * byteExtents.x +
         4 * 8 * byteExtents.x * byteExtents.y;
      for (std::size_t i = 0; i != byteExtents.x; ++i) {
         printf("%X ", s[offset + i]);
      }
      printf("\n");
   }
#endif

   // Mark the outside points as negative infinity.
   {
      // Use a 2-D grid of blocks. Because the number of refined patches may 
      // exceed the maximum allowed single grid dimension.
      const dim3 GridDim(std::min(numRefined, MaxGridExtent),
                         (numRefined + MaxGridExtent - 1) / MaxGridExtent);
      const dim3 ThreadsPerBlock(1, PatchExtent, PatchExtent);
      markOutsideKernel<<<GridDim, ThreadsPerBlock>>>
         (numRefined, patchesDev, indicesDev, byteExtents,
          reinterpret_cast<const unsigned char*>(outsideDev));
   }

   // Copy the outside array back to the host.
   std::vector<unsigned char> outside(numBytes);
   CUDA_CHECK(cudaMemcpy(&outside[0], outsideDev,
                         outside.size() * sizeof(unsigned char),
                         cudaMemcpyDeviceToHost));
   outsideAtLowerCorners->resize(product(gridExtents));
   for (std::size_t i = 0; i != gridExtents[0]; ++i) {
      for (std::size_t j = 0; j != gridExtents[1]; ++j) {
         for (std::size_t k = 0; k != gridExtents[2]; ++k) {
            (*outsideAtLowerCorners)[i + j * gridExtents[0] +
                                     k * gridExtents[0] * gridExtents[1]] =
               outside[i + j * PatchExtent * byteExtents.x +
                       k * PatchExtent * byteExtents.x * byteExtents.y];
         }
      }
   }

   // Free the device memory for the outside array.
   CUDA_CHECK(cudaFree(outsideDev));
}
#endif


} // namespace levelSet
}
