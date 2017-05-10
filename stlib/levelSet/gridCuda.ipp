/* -*- C++ -*- */

#if !defined(__levelSet_gridCuda_ipp__)
#error This file is an implementation detail of gridCuda.
#endif

namespace stlib
{
namespace levelSet
{


inline
void
allocateGridIndicesCuda(const GridGeometry<3, PatchExtent, float>& grid,
                        const std::size_t numRefined,
                        const std::vector<bool>& isActive,
                        uint3** indicesDev)
{
  typedef container::SimpleMultiIndexExtentsIterator<3> Iterator;

  // Allocate device memory for the indices of the active patches.
  CUDA_CHECK(cudaMalloc((void**)indicesDev,
                        numRefined * sizeof(uint3)));
  {
    // Compute the lower corners.
    std::vector<uint3> indices(numRefined);
    std::size_t n = 0;
    // Loop over the active patches.
    const Iterator end = Iterator::end(grid.gridExtents);
    for (Iterator i = Iterator::begin(grid.gridExtents); i != end; ++i) {
      const std::size_t index = grid.arrayIndex(*i);
      if (isActive[index]) {
        indices[n].x = (*i)[0];
        indices[n].y = (*i)[1];
        indices[n].z = (*i)[2];
        ++n;
      }
    }
    assert(n == numRefined);
    // Copy to the device.
    CUDA_CHECK(cudaMemcpy(*indicesDev, &indices[0],
                          indices.size() * sizeof(uint3),
                          cudaMemcpyHostToDevice));
  }
}


inline
void
allocateGridCuda(const Grid<float, 3, PatchExtent>& grid,
                 const std::size_t numRefined, float** patchesDev,
                 uint3** indicesDev)
{
  typedef container::SimpleMultiIndexExtentsIterator<3> Iterator;

  // Allocate device memory for the patches.
  CUDA_CHECK(cudaMalloc((void**)patchesDev,
                        grid.numVertices() * sizeof(float)));

  // Allocate device memory for the indices of the refined patches.
  CUDA_CHECK(cudaMalloc((void**)indicesDev,
                        numRefined * sizeof(uint3)));
  {
    // Compute the lower corners.
    std::vector<uint3> indices(numRefined);
    std::size_t n = 0;
    // Loop over the refined patches.
    const Iterator end = Iterator::end(grid.extents());
    for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
      const std::size_t index = grid.arrayIndex(*i);
      if (grid[index].isRefined()) {
        indices[n].x = (*i)[0];
        indices[n].y = (*i)[1];
        indices[n].z = (*i)[2];
        ++n;
      }
    }
    assert(n == numRefined);
    // Copy to the device.
    CUDA_CHECK(cudaMemcpy(*indicesDev, &indices[0],
                          indices.size() * sizeof(uint3),
                          cudaMemcpyHostToDevice));
  }
}


} // namespace levelSet
}
