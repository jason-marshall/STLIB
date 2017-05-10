/* -*- C++ -*- */

#if !defined(__levelSet_ballsCuda_ipp__)
#error This file is an implementation detail of ballsCuda.
#endif

namespace stlib
{
namespace levelSet
{


inline
void
allocateBallIndicesCuda
(const container::StaticArrayOfArrays<unsigned>& dependencies,
 unsigned** ballIndexOffsetsDev,
 unsigned** packedBallIndicesDev)
{
  // Allocate device memory for the ball index offsets.
  CUDA_CHECK(cudaMalloc((void**)ballIndexOffsetsDev,
                        (dependencies.getNumberOfArrays() + 1) *
                        sizeof(unsigned)));
  {
    // In "dependencies", we store pointers that determine the beginning
    // and end of each array. We need to convert these to offsets by
    // subtracting the address of the beginning of the storage.
    std::vector<unsigned> buffer(dependencies.getNumberOfArrays() + 1);
    buffer[0] = 0;
    for (std::size_t i = 0; i != dependencies.getNumberOfArrays(); ++i) {
      buffer[i + 1] = dependencies.end(i) - dependencies.begin();
    }
    // Copy to the device.
    CUDA_CHECK(cudaMemcpy(*ballIndexOffsetsDev, &buffer[0],
                          buffer.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }

  // Allocate device memory for the packed ball indices.
  CUDA_CHECK(cudaMalloc((void**)packedBallIndicesDev,
                        dependencies.size() * sizeof(unsigned)));
  // Copy to the device.
  CUDA_CHECK(cudaMemcpy(*packedBallIndicesDev, &dependencies[0],
                        dependencies.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
}


inline
void
allocateBallIndicesCuda
(const container::StaticArrayOfArrays<unsigned>& dependencies,
 const std::size_t numRefined,
 unsigned** ballIndexOffsetsDev,
 unsigned** packedBallIndicesDev)
{
  // Allocate device memory for the ball index offsets.
  CUDA_CHECK(cudaMalloc((void**)ballIndexOffsetsDev,
                        (numRefined + 1) * sizeof(unsigned)));
  {
    // In "dependencies", we store pointers that determine the beginning
    // and end of each array. We need to convert these to offsets by
    // subtracting the address of the beginning of the storage.
    std::vector<unsigned> buffer(numRefined + 1);
    buffer[0] = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i != dependencies.getNumberOfArrays(); ++i) {
      if (! dependencies.empty(i)) {
        buffer[n + 1] = dependencies.end(i) - dependencies.begin();
        ++n;
      }
    }
    assert(n == numRefined);
    // Copy to the device.
    CUDA_CHECK(cudaMemcpy(*ballIndexOffsetsDev, &buffer[0],
                          buffer.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }

  // Allocate device memory for the packed ball indices.
  CUDA_CHECK(cudaMalloc((void**)packedBallIndicesDev,
                        dependencies.size() * sizeof(unsigned)));
  // Copy to the device.
  CUDA_CHECK(cudaMemcpy(*packedBallIndicesDev, &dependencies[0],
                        dependencies.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
}


inline
void
allocateBallIndicesCuda
(const Grid<float, 3, PatchExtent>& grid,
 const container::StaticArrayOfArrays<unsigned>& dependencies,
 unsigned** ballIndexOffsetsDev,
 unsigned** packedBallIndicesDev)
{
  assert(grid.size() == dependencies.getNumberOfArrays());

  const std::size_t numRefined = grid.numRefined();

  // Allocate device memory for the ball index offsets.
  CUDA_CHECK(cudaMalloc((void**)ballIndexOffsetsDev,
                        (numRefined + 1) * sizeof(unsigned)));
  {
    // In "dependencies", we store pointers that determine the beginning
    // and end of each array. We need to convert these to offsets by
    // subtracting the address of the beginning of the storage.
    std::vector<unsigned> buffer(numRefined + 1);
    buffer[0] = 0;
    std::size_t n = 0;
    for (std::size_t i = 0; i != grid.size(); ++i) {
      if (grid[i].isRefined()) {
        buffer[n + 1] = dependencies.end(i) - dependencies.begin();
        ++n;
      }
    }
    assert(n == numRefined);
    // Copy to the device.
    CUDA_CHECK(cudaMemcpy(*ballIndexOffsetsDev, &buffer[0],
                          buffer.size() * sizeof(unsigned),
                          cudaMemcpyHostToDevice));
  }

  // Allocate device memory for the packed ball indices.
  CUDA_CHECK(cudaMalloc((void**)packedBallIndicesDev,
                        dependencies.size() * sizeof(unsigned)));
  // Copy to the device.
  CUDA_CHECK(cudaMemcpy(*packedBallIndicesDev, &dependencies[0],
                        dependencies.size() * sizeof(unsigned),
                        cudaMemcpyHostToDevice));
}


} // namespace levelSet
}
