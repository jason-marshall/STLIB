/* -*- C++ -*- */

#if !defined(__levelSet_solventAccessibleCavitiesCuda_ipp__)
#error This file is an implementation detail of solventAccessibleCavitiesCuda.
#endif

namespace stlib
{
namespace levelSet
{


inline
void
solventAccessibleCavitiesCuda(const std::vector<geom::Ball<float, 3> >& balls,
                              const float probeRadius,
                              const float targetGridSpacing,
                              std::vector<float>* content,
                              std::vector<float>* boundary)
{
  typedef float T;
  const std::size_t D = 3;
  const std::size_t N = 8;

  typedef Grid<T, D, N> Grid;
  typedef Grid::BBox BBox;

  // Place a bounding box around the balls.
  BBox domain;
  domain.bound(balls.begin(), balls.end());
  // We expand by the probe radius plus the threshold for determining seeds.
  domain.offset(probeRadius + targetGridSpacing * std::sqrt(T(D)));
  // Make the grid.
  Grid grid(domain, targetGridSpacing);

  // Calculate the solvent-accessible cavities.
  solventAccessibleCavitiesCuda(&grid, balls, probeRadius);

  // Compute the content (volume) and boundary (surface area).
  levelSet::contentAndBoundary(grid, content, boundary);
}


inline
void
solventAccessibleCavitiesCuda(Grid<float, 3, PatchExtent>* grid,
                              const std::vector<geom::Ball<float, 3> >& balls,
                              const float probeRadius)
{
  typedef Grid<float, 3, PatchExtent>::IndexList IndexList;

  const float Inf = std::numeric_limits<float>::infinity();

  // Dispense with the trivial case.
  if (grid->empty()) {
    return;
  }

  // Determine the patch/ball dependencies.
  container::StaticArrayOfArrays<unsigned> interiorDependencies;
  // Offset the balls' radii to include the volume of calculated distance.
  const float MaxDistance = 2 * grid->spacing;
  const float Offset = probeRadius + MaxDistance;
  std::vector<geom::Ball<float, 3> > offsetBalls(balls);
  for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
    offsetBalls[i].radius += Offset;
  }
  // Calculate the dependencies.
  patchDependencies(*grid, offsetBalls.begin(), offsetBalls.end(),
                    &interiorDependencies);

  // Subtract the MaxDistance to get the plain offset balls.
  for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
    offsetBalls[i].radius -= MaxDistance;
  }
  // Determine the patches that have all negative distances.
  std::vector<IndexList> negativePatches;
  findPatchesWithNegativeDistance(*grid, offsetBalls, interiorDependencies,
                                  &negativePatches);

  // Calculate the sizes for the positive distance dependencies.
  // Start with the sizes for the interior dependencies.
  std::vector<std::size_t> sizes(grid->size());
  for (std::size_t i = 0; i != sizes.size(); ++i) {
    sizes[i] = interiorDependencies.size(i);
  }
  // Clear the negative patches.
  for (std::size_t i = 0; i != negativePatches.size(); ++i) {
    sizes[grid->arrayIndex(negativePatches[i])] = 0;
  }

  // Copy the relevant positive patch dependencies from the interior
  // patch dependencies.
  container::StaticArrayOfArrays<unsigned> positiveDependencies;
  positiveDependencies.rebuild(sizes.begin(), sizes.end());
  for (std::size_t i = 0; i != positiveDependencies.getNumberOfArrays(); ++i) {
    for (std::size_t j = 0; j != positiveDependencies.size(i); ++j) {
      positiveDependencies(i, j) = interiorDependencies(i, j);
    }
  }

  // Refine the appropriate patches and set the rest to have an unknown
  // distance.
  grid->refine(positiveDependencies);
  const std::size_t numRefined = grid->numRefined();

  // Set the fill value for patches with all negative distances.
  for (std::size_t i = 0; i != negativePatches.size(); ++i) {
    (*grid)(negativePatches[i]).fillValue =
      - std::numeric_limits<float>::infinity();
  }

  // Allocate device memory for the refined patches and their indices.
  float* patchesDev;
  uint3* indicesDev;
  allocateGridCuda(*grid, numRefined, &patchesDev, &indicesDev);

  // Compute the solvent-center excluded domain.
  const float3 lowerCorner = {grid->lowerCorner[0], grid->lowerCorner[1],
                              grid->lowerCorner[2]
                             };
  positiveDistanceCuda(numRefined, patchesDev, indicesDev, lowerCorner,
                       grid->spacing, balls, probeRadius,
                       positiveDependencies);

  // Mark the outside component of the solvent-center-accesible domain as
  // negative far-away.
  std::vector<bool> outsideAtLowerCorners;
  markOutsideAsNegativeInf(grid->extents(), numRefined, patchesDev,
                           indicesDev, negativePatches,
                           &outsideAtLowerCorners);
  // Mark the unrefined patches.
  typedef container::SimpleMultiIndexRangeIterator<3> Iterator;
  typedef Grid<float, 3, PatchExtent>::VertexPatch VertexPatch;
  const Iterator pEnd = Iterator::end(grid->extents());
  for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
    VertexPatch& patch = (*grid)(*p);
    if (! patch.isRefined()) {
      if (outsideAtLowerCorners[grid->arrayIndex(*p)]) {
        patch.fillValue = -Inf;
      }
    }
  }

  // Copy the patch data back to the host.
  CUDA_CHECK(cudaMemcpy(grid->data(), patchesDev,
                        grid->numVertices() * sizeof(float),
                        cudaMemcpyDeviceToHost));

  // Free the device memory for the positive distance grid.
  CUDA_CHECK(cudaFree(patchesDev));
  CUDA_CHECK(cudaFree(indicesDev));

  // CONTINUE
  //printInfo(*grid, std::cerr);

  // Record the points at the boundary of the solvent-center-accessible
  // cavities. These are positive points with negative neighbors, and will
  // be used as seed points to compute the solvent-accessible cavities.
  std::vector<geom::Ball<float, 3> > seeds;
  solventAccessibleCavitySeeds(*grid, probeRadius, &seeds);

  // If there are no seed points, there are no solvent-accessible cavities.
  if (seeds.empty()) {
    // Fill the grid with positive far-away values and return.
    grid->clear();
    for (std::size_t i = 0; i != grid->size(); ++i) {
      (*grid)[i].fillValue = Inf;
    }
    return;
  }

  // Compute the solvent-accessible cavities from the seed points.
  negativePowerDistanceCuda(grid, seeds);
}


inline
void
solventAccessibleCavitiesQueueCuda
(Grid<float, 3, PatchExtent>* grid,
 const std::vector<geom::Ball<float, 3> >& balls,
 const float probeRadius)
{
  const float Inf = std::numeric_limits<float>::infinity();

  // Compute the solvent-center excluded domain.
  positiveDistanceCuda(grid, balls, probeRadius, 2 * grid->spacing);

  // Mark the outside component of the solvent-center-accesible domain as
  // negative far-away.
  markOutsideAsNegativeInfQueue(grid);

  // Record the points at the boundary of the solvent-center-accessible
  // cavities. These are positive points with negative neighbors, and will
  // be used as seed points to compute the solvent-accessible cavities.
  std::vector<geom::Ball<float, 3> > seeds;
  solventAccessibleCavitySeeds(*grid, probeRadius, &seeds);

  // If there are no seed points, there are no solvent-accessible cavities.
  if (seeds.empty()) {
    // Fill the grid with positive far-away values and return.
    grid->clear();
    for (std::size_t i = 0; i != grid->size(); ++i) {
      (*grid)[i].fillValue = Inf;
    }
    return;
  }

  // Compute the solvent-accessible cavities from the seed points.
  negativePowerDistanceCuda(grid, seeds);
}


} // namespace levelSet
}
