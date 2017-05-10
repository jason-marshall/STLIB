/* -*- C++ -*- */

#if !defined(__levelSet_solventExcludedCavitiesCuda_ipp__)
#error This file is an implementation detail of solventExcludedCavitiesCuda.
#endif

namespace stlib
{
namespace levelSet
{


inline
void
solventExcludedCavitiesCuda(Grid<float, 3, PatchExtent>* grid,
                            const std::vector<geom::Ball<float, 3> >& balls,
                            const float probeRadius)
{
  const std::size_t D = 3;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // The threshold for accepting a point as a seed.
  // 1.01 * (diagonal length of a voxel)
  // This is also how far we must compute the distance beyond the zero
  // iso-surface of the solvent-accessible surface.
  const float seedThreshold = 1.01 * std::sqrt(float(D)) * grid->spacing;

  // Compute the solvent-accessible domain. For the purpose of refining the
  // grid, the balls will be enlarged by probeRadius + seedThreshold. Any
  // patch that intersects these balls will be refined. The level set will
  // be computed on all refined patches. However, only points outside
  // the balls are guaranteed to have the correct Euclidean distance.
  positiveDistanceCuda(grid, balls, probeRadius, seedThreshold);

  // Determine the seeds used in computing the solvent-excluded domain.
  std::vector<geom::Ball<float, D> > seeds;
  std::vector<bool> areFarAway(grid->size(), false);
  solventExcludedSeeds(*grid, probeRadius, seedThreshold, &seeds, &areFarAway);

  // We will use the solvent-accessible grid (with its set of refined patches)
  // to compute the solvent-excluded domain.
  // From this point on, we will work only with the refined grid points.

  // Now we work with the seeds to calculate the solvent-excluded domain.

  // From the dependencies for all patches, calculate the dependencies
  // for only the refined patches that are close to the solvent-excluded
  // domain.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // Compute the dependencies for all patches.
    container::StaticArrayOfArrays<unsigned> allDependencies;
    // We need to compute the distance a little past the surface of the
    // seeds. Offset by the diagonal length of a voxel.
    std::vector<geom::Ball<float, D> > offsetSeeds(seeds);
    for (std::size_t i = 0; i != offsetSeeds.size(); ++i) {
      offsetSeeds[i].radius += seedThreshold;
    }
    patchDependencies(*grid, offsetSeeds.begin(), offsetSeeds.end(),
                      &allDependencies);

    // Record the number of dependencies for each of the close, refined
    // patches.
    std::vector<std::size_t> sizes(allDependencies.getNumberOfArrays());
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      if ((*grid)[i].isRefined() && ! areFarAway[i]) {
        sizes[i] = allDependencies.size(i);
      }
      else {
        // We ignore dependencies for the rest.
        sizes[i] = 0;
      }
    }
    // Allocate memory.
    dependencies.rebuild(sizes.begin(), sizes.end());
    // Copy the dependencies for the refined patches.
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      if (! dependencies.empty(i)) {
        std::copy(allDependencies.begin(i), allDependencies.end(i),
                  dependencies.begin(i));
      }
    }
    // Put the closest balls for each patch first in the list of
    // dependencies.
    putClosestBallsFirst(*grid, seeds, &dependencies);
  }

#if 0
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
#endif

  // Compute the power distance to the seeds.
  negativePowerDistanceCuda(grid, seeds, dependencies, seedThreshold);

#if 0
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "Elapsed time to compute power distance = "
            << elapsedTime << " ms.\n";
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
#endif

  // Set the distance for the far away refined patches.
  for (std::size_t i = 0; i != grid->size(); ++i) {
    if (areFarAway[i]) {
      std::fill((*grid)[i].begin(), (*grid)[i].end(), -seedThreshold);
    }
  }

  // Reverse the sign of the distance to get the solvent-excluded domain.
  float* const end = grid->data() + grid->numVertices();
  for (float* i = grid->data(); i != end; ++i) {
    *i = - *i;
  }

  // Subtract the van der Waals domain to obtain the solvent-excluded cavities.
  subtract(grid, balls, seedThreshold);
}


// Compute the volume of the solvent-excluded cavities and avoid storing any
// level-set function on a grid. Only a patch at a time will be used.
// Compute the volume using the content from distance algorithms.
inline
float
solventExcludedCavitiesVolumeCuda
(const std::vector<geom::Ball<float, 3> >& moleculeBalls,
 const float probeRadius, const float targetGridSpacing)
{
  const std::size_t D = 3;
  typedef GridGeometry<D, PatchExtent, float> Grid;
  typedef Grid::BBox BBox;
  typedef geom::Ball<float, D> Ball;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  //
  // Define the grid geometry for computing the solvent-excluded cavities.
  //
  // Place a bounding box around the balls comprising the molecule.
  BBox targetDomain;
  targetDomain.bound(moleculeBalls.begin(), moleculeBalls.end());
  // Expand by the probe radius so that we can determine the global cavities.
  // add the target grid spacing to get one more grid point.
  targetDomain.offset(probeRadius + targetGridSpacing);
  // Define the grid geometry.
  const Grid grid(targetDomain, targetGridSpacing);

  // Determine the seeds used in computing the solvent-excluded domain.
  std::vector<Ball> seeds;
  solventExcludedSeeds(grid, moleculeBalls, probeRadius, &seeds);

  // Determine the patches that are relevant for the solvent-excluded
  // cavities. A patch is irrelevant if it is farther than
  // (probeRadius + diagonal) from all balls in the molecule or if it is
  // farther than diagonal inside any ball.
  std::vector<bool> relevant;
  solventExcludedCavitiesPatches(grid, moleculeBalls, probeRadius, &relevant);

  // The combined set of molecule balls and solvent balls.
  std::vector<Ball> balls(moleculeBalls.size() + seeds.size());
  std::copy(moleculeBalls.begin(), moleculeBalls.end(), balls.begin());
  std::copy(seeds.begin(), seeds.end(), balls.begin() + moleculeBalls.size());

  // From the dependencies for all patches, calculate the dependencies
  // for only the relevant patches.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // The threshold for accepting a point as a seed is
    // 1.01 * (diagonal length of a voxel)
    // This is also how far we must compute the distance beyond the zero
    // iso-surface of the solvent-accessible surface.
    const float diagonal = 1.01 * std::sqrt(float(D)) * grid.spacing;

    // Compute the dependencies for all patches.
    container::StaticArrayOfArrays<unsigned> allDependencies;
    // We need to compute the distance a little past the surface of the
    // balls. Offset by the diagonal length of a voxel.
    for (std::size_t i = 0; i != balls.size(); ++i) {
      balls[i].radius += diagonal;
    }
    patchDependencies(grid, balls.begin(), balls.end(), &allDependencies);
    // CONTINUE REMOVE
    assert(allDependencies.getNumberOfArrays() == relevant.size());
    // Return the radii to their correct values.
    for (std::size_t i = 0; i != balls.size(); ++i) {
      balls[i].radius -= diagonal;
    }

    // Record the number of dependencies for each of the relevant patches.
    std::vector<std::size_t> sizes;
    for (std::size_t i = 0; i != relevant.size(); ++i) {
      if (relevant[i]) {
        sizes.push_back(allDependencies.size(i));
      }
    }
    // Allocate memory.
    dependencies.rebuild(sizes.begin(), sizes.end());
    // Copy the dependencies for the relevant patches.
    std::size_t n = 0;
    for (std::size_t i = 0; i != relevant.size(); ++i) {
      if (relevant[i]) {
        std::copy(allDependencies.begin(i), allDependencies.end(i),
                  dependencies.begin(n++));
      }
    }
#if 1
    // Put the closest balls for each patch first in the list of
    // dependencies.
    putClosestBallsFirst(grid, relevant, balls, &dependencies);
#endif
  }

#if 0
  std::cout << "dependencies = "
            << timer.toc() * 1000 << " ms.\n";
  timer.tic();
#endif

#if 0
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
#endif

  // Compute the volume of the exterior region.
  const float volume = exteriorVolumeCuda(grid, relevant, balls, dependencies);

#if 0
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "Elapsed time to compute exterior volume = "
            << elapsedTime << " ms.\n";
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
#endif

  return volume;
}


// Compute the volume of the solvent-excluded cavities and avoid storing any
// level-set function on a grid. Only a patch at a time will be used.
// Solvent probes will be placed by distributing points on a sphere
// (hence the Pos in the name).
// Compute the volume using only the sign of the distance.
inline
float
solventExcludedCavitiesPosCuda(const float targetGridSpacing,
                               const std::vector<geom::Ball<float, 3> >& balls,
                               const float probeRadius)
{
  const std::size_t D = 3;
  typedef GridGeometry<D, PatchExtent, float> Grid;
  typedef Grid::BBox BBox;
  typedef geom::Ball<float, D> Ball;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  //
  // Define the grid geometry for computing the solvent-excluded cavities.
  //
  // Place a bounding box around the balls comprising the molecule.
  BBox targetDomain;
  targetDomain.bound(balls.begin(), balls.end());
  // Define the grid geometry.
  const Grid grid(targetDomain, targetGridSpacing);

  // A ball influences a patch if it is within a distance of 2 * probeRadius.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // Expand the balls.
    std::vector<geom::Ball<float, 3> > offsetBalls(balls);
    const float offset = 2 * probeRadius;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += offset;
    }
    patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

#if 0
  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));
#endif

  // Compute the volume solvent-excluded cavities.
  const float volume = solventExcludedCavitiesPosCuda(grid, balls, probeRadius,
                       dependencies);

#if 0
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  std::cout << "Elapsed time to compute SEC volume = "
            << elapsedTime << " ms.\n";
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
#endif

  return volume;
}


} // namespace levelSet
}
