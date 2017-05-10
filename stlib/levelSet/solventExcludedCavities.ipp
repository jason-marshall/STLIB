// -*- C++ -*-

#if !defined(__levelSet_solventExcludedCavities_ipp__)
#error This file is an implementation detail of solventExcludedCavities.
#endif

namespace stlib
{
namespace levelSet
{


// Determine the patches that are relevant to the solvent-excluded
// cavities.  A patch is irrelevant if it is farther than
// (probeRadius + diagonal) from all balls or if it is farther than
// diagonal inside any ball.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedCavitiesPatches(const GridGeometry<_D, N, _T>& grid,
                               const std::vector<geom::Ball<_T, _D> >& balls,
                               const _T probeRadius,
                               std::vector<bool>* relevant)
{
  typedef GridGeometry<_D, N, _T> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> RangeIterator;

  // The radius of a patch that has been expanded by one grid point in each
  // direction.
  const _T expandedPatchRadius = 0.5 * grid.spacing * (N + 1) *
                                 std::sqrt(_T(_D));
  // Make a regular grid of the patch centers.
  const geom::BBox<_T, _D> patchCenterDomain = {
    grid.getVertexPatchCenter(ext::filled_array<IndexList>(0)),
    grid.getVertexPatchCenter(grid.gridExtents - std::size_t(1))
  };
  const geom::SimpleRegularGrid<_T, _D>
  patchCenters(grid.gridExtents, patchCenterDomain);

  // Allocate memory.
  relevant->resize(ext::product(grid.gridExtents));
  // Start with none being relevant.
  std::fill(relevant->begin(), relevant->end(), false);

  geom::Ball<_T, _D> ball;
  Range range;

  //
  // Mark the ones that are close to a ball as being relevant.
  //
  const _T offset = expandedPatchRadius + probeRadius;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Expand by the probe radius to go from the van der Waals domain
    // to the solvent-excluded domain.
    // Expand by the patch radius to go from intersecting the patch to
    // containing the patch center.
    ball = balls[i];
    ball.radius += offset;
    // Get the patch centers that are inside the bounding box for the ball.
    range = patchCenters.computeRange
      (geom::specificBBox<geom::BBox<_T, _D> >(ball));
    // Loop over these patch centers and determine which ones are actually
    // inside the expanded ball. (This test is equivalent to the bounding
    // ball of the expanded patch intersecting the i_th ball.)
    const RangeIterator pEnd = RangeIterator::end(range);
    for (RangeIterator p = RangeIterator::begin(range); p != pEnd; ++p) {
      if (isInside(ball, grid.getVertexPatchCenter(*p))) {
        (*relevant)[grid.arrayIndex(*p)] = true;
      }
    }
  }

  //
  // Mark the ones that are far inside a ball as being irrelevant.
  //
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Skip the balls that could not contain a patch.
    if (balls[i].radius <= expandedPatchRadius) {
      continue;
    }
    // The ball that contains the centers of patches that are far inside.
    ball = balls[i];
    ball.radius -= expandedPatchRadius;
    // Get the patch centers that are inside the bounding box for the ball.
    range = patchCenters.computeRange
      (geom::specificBBox<geom::BBox<_T, _D> >(ball));
    // Loop over these patch centers and determine which ones are actually
    // inside the shrunken ball. (This test is equivalent to the bounding
    // ball of the patch being far inside the i_th ball.)
    const RangeIterator pEnd = RangeIterator::end(range);
    for (RangeIterator p = RangeIterator::begin(range); p != pEnd; ++p) {
      if (isInside(ball, grid.getVertexPatchCenter(*p))) {
        (*relevant)[grid.arrayIndex(*p)] = false;
      }
    }
  }
}


template<typename _T, std::size_t _D>
inline
void
solventExcludedCavities(GridUniform<_T, _D>* grid,
                        const std::vector<geom::Ball<_T, _D> >& balls,
                        const _T probeRadius, const _T maxDistance)
{
  // Compute the solvent excluded surface.
  solventExcluded(grid, balls, probeRadius);
  // Subtract the balls.
  subtract(grid, grid->domain(), balls, maxDistance);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedCavities(Grid<_T, _D, N>* grid,
                        const std::vector<geom::Ball<_T, _D> >& balls,
                        const _T probeRadius, const _T maxDistance)
{
  typedef container::SimpleMultiArrayRef<_T, _D> MultiArrayRef;
  typedef typename MultiArrayRef::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Determine the patch/ball dependencies for the solvent-excluded surface.
  container::StaticArrayOfArrays<unsigned> solvExcDependencies;
  solventExcludedDependencies(*grid, balls, probeRadius,
                              &solvExcDependencies);

  // Determine the patch/ball dependencies for subtracting the atoms.
  container::StaticArrayOfArrays<unsigned> differenceDependencies;
  {
    std::vector<geom::Ball<_T, _D> > offsetBalls = balls;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += maxDistance;
    }
    patchDependencies(*grid, offsetBalls.begin(), offsetBalls.end(),
                      &differenceDependencies);
  }

  // Use the solvent-excluded dependencies to determine the patches to refine.
  // Refine the appropriate patches and set the rest to have an unknown
  // distance.
  grid->refine(solvExcDependencies);

  // Use a multi-array to wrap the patches.
  const IndexList patchExtents = ext::filled_array<IndexList>(N);
  container::SimpleMultiArrayRef<_T, _D> patch(0, patchExtents);
  std::vector<geom::Ball<_T, _D> > influencingBalls;
  geom::BBox<_T, _D> domain;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid->extents());
  for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
    const std::size_t index = grid->arrayIndex(*i);
    if (!(*grid)[index].isRefined()) {
      continue;
    }
    // Build the parameters for the solvent-excluded surface.
    patch.rebuild((*grid)[index].data(), patchExtents);
    influencingBalls.clear();
    for (std::size_t n = 0; n != solvExcDependencies.size(index);
         ++n) {
      influencingBalls.push_back(balls[solvExcDependencies(index, n)]);
    }
    // Compute the solvent excluded surface.
    domain = grid->getVertexPatchDomain(*i);
    solventExcluded(&patch, domain, influencingBalls, probeRadius);

    // Build the parameters for subtracting the balls.
    influencingBalls.clear();
    for (std::size_t n = 0; n != differenceDependencies.size(index); ++n) {
      influencingBalls.push_back(balls[differenceDependencies(index, n)]);
    }
    // Subtract the balls.
    subtract(&patch, domain, influencingBalls, maxDistance);
  }
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedCavitiesUsingSeeds(Grid<_T, _D, N>* grid,
                                  const std::vector<geom::Ball<_T, _D> >& balls,
                                  const _T probeRadius)
{
  // A little bit more than the diagonal length of a voxel. This constant
  // is also used in solventExcludedUsingSeeds().
  const _T seedThreshold = 1.01 * std::sqrt(double(_D)) * grid->spacing;

  // Compute the solvent-excluded domain.
  solventExcludedUsingSeeds(grid, balls, probeRadius);
  // Subtract the van der Waals domain.
  subtract(grid, balls, seedThreshold);
}


template<typename _T, std::size_t _D>
inline
std::pair<_T, _T>
solventExcludedCavities(const std::vector<geom::Ball<_T, _D> >& balls,
                        const _T probeRadius,
                        const _T targetGridSpacing)
{
  const std::size_t N = 8;

  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::BBox BBox;

  // Place a bounding box around the balls.
  BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // We will need to compute the solvent-accessible surface to determine
  // the seeds. Thus we expand by the probe radius plus the threshold for
  // determining seeds.
  offset(&domain, probeRadius + targetGridSpacing * std::sqrt(_T(_D)));
  // Make the grid.
  Grid grid(domain, targetGridSpacing);

  // Calculate the solvent-excluded cavities.
  solventExcludedCavitiesUsingSeeds(&grid, balls, probeRadius);

  // Compute the content (volume) and boundary (surface area).
  _T content, boundary;
  levelSet::contentAndBoundary(grid, &content, &boundary);
  return std::make_pair(content, boundary);
}


// Compute the content of the solvent-excluded cavities avoid storing any
// level-set function on a grid. Only a patch at a time will be used.
// Compute the content using the content from distance algorithms.
template<typename _T, std::size_t _D>
inline
_T
solventExcludedCavitiesContent
(const _T targetGridSpacing,
 const std::vector<geom::Ball<_T, _D> >& moleculeBalls,
 const _T probeRadius)
{
  const std::size_t N = 8;
  typedef GridGeometry<_D, N, _T> Grid;
  typedef typename Grid::BBox BBox;
  typedef geom::Ball<float, _D> Ball;

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

  // The threshold for accepting a point as a seed is
  // 1.01 * (diagonal length of a voxel)
  // This is also how far we must compute the distance beyond the zero
  // iso-surface of the solvent-accessible surface.
  const float diagonal = 1.01 * std::sqrt(_T(_D)) * grid.spacing;

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
    // CONTINUE REMOVE
    assert(dependencies.getNumberOfArrays() ==
           std::size_t(std::count(relevant.begin(), relevant.end(), true)));
    // CONTINUE: Write a version for the partial dependencies.
#if 0
    // Put the closest balls for each patch first in the list of
    // dependencies.
    putClosestBallsFirst(grid, balls, &dependencies);
#endif
  }

  // Compute the content of the exterior region.
  return exteriorContent(grid, relevant, balls, dependencies);
}


} // namespace levelSet
}
