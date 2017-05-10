// -*- C++ -*-

#if !defined(__levelSet_solventExcluded_ipp__)
#error This file is an implementation detail of solventExcluded.
#endif

namespace stlib
{
namespace levelSet
{


template<typename _T, std::size_t _D>
inline
void
solventExcluded(container::SimpleMultiArrayRef<_T, _D>* grid,
                const geom::BBox<_T, _D>& domain,
                const std::vector<geom::Ball<_T, _D> >& balls,
                const _T probeRadius)
{
  // Compute the distance to the offset balls.
  {
    std::vector<geom::Ball<_T, _D> > offset = balls;
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    negativeDistance(grid, domain, offset);
  }
  // Offset by the probe radius.
  *grid += probeRadius;
}


template<typename _T, std::size_t _D>
inline
void
solventExcluded(GridUniform<_T, _D>* grid,
                const std::vector<geom::Ball<_T, _D> >& balls,
                const _T probeRadius)
{
  solventExcluded(grid, grid->domain(), balls, probeRadius);
}


// Determine the patch/ball dependencies for the solvent-excluded surface.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedDependencies
(const Grid<_T, _D, N>& grid,
 const std::vector<geom::Ball<_T, _D> >& balls,
 const _T probeRadius,
 container::StaticArrayOfArrays<unsigned>* dependencies)
{
  // Calculate the largest radius of the solvent-expanded balls.
  _T maxRadius = 0;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (balls[i].radius > maxRadius) {
      maxRadius = balls[i].radius;
    }
  }
  maxRadius += probeRadius;
  // First expand the balls by the probe radius, then consider
  // their influence. A ball may influence points up to maxRadius
  // beyond its surface. Consider the case that one ball barely
  // intersects the largest ball. Then points all the way up to
  // the center of the largest ball may be closest to the
  // intersection of the two balls.
  const _T offset = probeRadius + maxRadius;
  std::vector<geom::Ball<_T, _D> > offsetBalls = balls;
  for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
    offsetBalls[i].radius += offset;
  }
  patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                    dependencies);
}


template<typename _T, std::size_t _D, std::size_t _PatchExtent>
inline
void
solventExcluded(Grid<_T, _D, _PatchExtent>* grid,
                const std::vector<geom::Ball<_T, _D> >& balls,
                const _T probeRadius)
{
  // Compute the distance to the offset balls.
  {
    std::vector<geom::Ball<_T, _D> > offset = balls;
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    negativeDistance(grid, offset);
  }
  // Offset by the probe radius.
  *grid += probeRadius;
}


// Use linear interpolation to evaluate the value at the center.
template<typename _T, std::size_t _D, std::size_t N>
inline
_T
interpolateCenter(const Patch<_T, _D, N>& patch)
{
  // The patch extent must be a multiple of 2. Then the center point is at
  // the center of a voxel.
  static_assert(N % 2 == 0 && N >= 2, "Bad patch extent.");
  assert(patch.isRefined());

  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;
  typedef typename Iterator::Range Range;
  typedef typename Grid<_T, _D, N>::IndexList IndexList;

  const IndexList Extents = ext::filled_array<IndexList>(2);
  const IndexList Bases = ext::filled_array<IndexList>(N / 2 - 1);
  const Range Voxel = {Extents, Bases};

  _T result = 0;
  // Loop over the voxel at the center of the patch.
  const Iterator iEnd = Iterator::end(Voxel);
  for (Iterator i = Iterator::begin(Voxel); i != iEnd; ++i) {
    result += patch(*i);
  }
  // Divide by 2^_D to get the average.
  return result / (std::size_t(1) << _D);
}


// Compute the seed radius and centers that are used to cover the
// positive region of a patch. These are used in addition to the seeds
// placed along the zero iso-surface. We use as few seeds as possible
// to cover the positive regions.
// The centers are offsets from the lower corner of the patch.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
positiveCoveringSeedParameters
(const Grid<_T, _D, N>& grid, const _T probeRadius, _T* seedRadius,
 std::vector<typename Grid<_T, _D, N>::IndexList>* centerIndices,
 std::vector<typename Grid<_T, _D, N>::Point>* centerOffsets)
{
  typedef typename Grid<_T, _D, N>::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  //
  // Compute the seed radius.
  //
  // Start with a single seed.
  std::size_t indexRadius = (N + 1) / 2;
  *seedRadius = 1.001 * grid.spacing * indexRadius * std::sqrt(_T(_D));
  // Decrease the seed radius as much as necessary.
  while (probeRadius < *seedRadius) {
    // The minimum allowed radius is one grid point. If we have already
    // reached that point, we cannot decrease the radius further.
    assert(indexRadius != 1);
    indexRadius = (indexRadius + 1) / 2;
    *seedRadius = 1.001 * grid.spacing * indexRadius * std::sqrt(_T(_D));
  }
  //
  // Record the patch indices of the seed centers.
  //
  centerIndices->clear();
  centerOffsets->clear();
  // The number of seeds in each dimension.
  const IndexList extents =
    ext::filled_array<IndexList>((N + 2 * indexRadius) /
                                 (2 * indexRadius + 1));
  // Loop over the seed centers.
  const Iterator iEnd = Iterator::end(extents);
  for (Iterator i = Iterator::begin(extents); i != iEnd; ++i) {
    centerIndices->push_back(*i * (2 * indexRadius + 1) + indexRadius);
    centerOffsets->push_back
      (stlib::ext::convert_array<_T>(centerIndices->back()) * grid.spacing);
  }
}


// Add seeds to cover the positive region of the specified patch. These are
// used in addition to the seeds placed along the zero iso-surface.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
positiveCoveringSeeds
(const Grid<_T, _D, N>& grid,
 const typename Grid<_T, _D, N>::IndexList& patchIndex,
 const _T seedRadius,
 const std::vector<typename Grid<_T, _D, N>::IndexList>& centerIndices,
 const std::vector<typename Grid<_T, _D, N>::Point>& centerOffsets,
 std::vector<geom::Ball<_T, _D> >* seeds)
{
  typedef typename Grid<_T, _D, N>::Point Point;

#ifdef STLIB_DEBUG
  assert(grid(patchIndex).isRefined());
#endif

  const Point lowerCorner = grid.getPatchLowerCorner(patchIndex);

  geom::Ball<_T, _D> ball;
  ball.radius = seedRadius;
  for (std::size_t i = 0; i != centerIndices.size(); ++i) {
    if (grid(patchIndex)(centerIndices[i]) > 0) {
      ball.center = lowerCorner + centerOffsets[i];
      seeds->push_back(ball);
    }
  }
}


// Return true if the whole patch is far away (farther than the seed
// threshold) from the solvent-excluded surface. We first check if the
// patch has any positive values. If not, we return false.
// Next we detect the far away condition by placing a ball of radius
// probeRadius + maxDistance - seedThreshold
// at the grid point with maximum distance. If the patch lies within the
// ball, then all of the grid points are more than seedThreshold from
// the solvent-excluded domain.
template<typename _T, std::size_t _D, std::size_t N>
inline
bool
solventExcludedIsFarAway(const Grid<_T, _D, N>& grid,
                         const typename Grid<_T, _D, N>::IndexList& gridIndex,
                         const _T probeRadius,
                         const _T seedThreshold)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef typename Grid::IndexList IndexList;

  // Convenience reference to the patch.
  const VertexPatch& patch = grid(gridIndex);
  assert(patch.isRefined());
  // First find the grid point in the patch with the maximum distance.
  const _T* maxElement = std::max_element(patch.begin(), patch.end());
  // Early exit for patch with no positive values.
  if (*maxElement < 0) {
    return false;
  }
  // Determine the multi-index of the maximum element.
  IndexList patchIndex;
  {
    static_assert(N == 8, "Only extent of 8 supported.");
    // The container index of the maximum element.
    std::size_t containerIndex = maxElement - patch.begin();
    // Convert the container index to a multi-index.
    for (std::size_t i = 0; i != _D; ++i) {
      patchIndex[i] = containerIndex & std::size_t(7);
      containerIndex >>= 3;
    }
  }
  // In order to check if the bounding ball for the patch is inside
  // the seed that is far away from the solvent-excluded domain, we subtract
  // the radius of the bounding ball from the seed ball. Then we check if
  // center of the patch is inside this ball.
  const _T HalfLength = 0.5 * grid.spacing * (N - 1);
  const geom::Ball<_T, _D> seed =
    {grid.getPatchLowerCorner(patchIndex) +
     grid.spacing * stlib::ext::convert_array<_T>(patchIndex),
     _T(probeRadius + *maxElement - seedThreshold -
        HalfLength * std::sqrt(_D))};
  const std::array<_T, _D> center =
    grid.getPatchLowerCorner(patchIndex) + HalfLength;
  return isInside(seed, center);
}


// From the solvent-accessible surface, determine the seeds for computing the
// solvent-excluded domain.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedSeeds(const Grid<_T, _D, N>& grid, const _T probeRadius,
                     const _T seedThreshold,
                     std::vector<geom::Ball<_T, _D> >* seeds,
                     std::vector<bool>* areFarAway)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::Point Point;

  typedef typename Grid::VertexPatch VertexPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

#if 0
  // The length of a vertex patch is grid.spacing * (N - 1).
  const _T HalfLength = 0.5 * grid.spacing * (N - 1);
  // We use the half diagonal length (expanded by a small amount to account
  // truncation errors) for seeds that cover a patch.
  const _T WholePatchSeedRadius = 1.001 * HalfLength * std::sqrt(_D);

  // For this method to work, the probe radius must be at least as large as
  // the radius that we use for covering a whole patch. Otherwise these seeds
  // may intersect the solvent-excluded domain.
  assert(probeRadius >= WholePatchSeedRadius);
#endif

  _T seedRadius = 0;
  std::vector<IndexList> centerIndices;
  std::vector<Point> centerOffsets;
  positiveCoveringSeedParameters(grid, probeRadius, &seedRadius,
                                 &centerIndices, &centerOffsets);

  geom::Ball<_T, _D> seed;
  const Iterator pEnd = Iterator::end(grid.extents());
  for (Iterator p = Iterator::begin(grid.extents()); p != pEnd; ++p) {
    const VertexPatch& patch = grid(*p);
    // We are only concerned with refined patches.
    if (! patch.isRefined()) {
      continue;
    }
    if (solventExcludedIsFarAway(grid, *p, probeRadius, seedThreshold)) {
      (*areFarAway)[grid.arrayIndex(*p)] = true;
    }
    else {
      positiveCoveringSeeds(grid, *p, seedRadius, centerIndices,
                            centerOffsets, seeds);
    }

#if 0
    // CONTINUE: I don't think that I need this now that I am marking
    // far away patches.
    // If the interpolated value at the center is positive, place a
    // seed at that point to cover the grid points in this patch.
    if (interpolateCenter(patch) > 0) {
      seed.center = grid.getPatchLowerCorner(*p);
      seed.center += HalfLength;
      seed.radius = WholePatchSeedRadius;
      seeds->push_back(seed);
    }
#endif
    // Loop over the grid points in the patch.
    const Iterator iEnd = Iterator::end(patch.extents());
    for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
      const _T a = patch(*i);
      // A seed must be near the interface.
      if (a > 0 && a < seedThreshold) {
        seed.radius = probeRadius + a;
        seed.center = grid.indexToLocation(*p, *i);
        seeds->push_back(seed);
      }
    }
  }
}


// Determine the seeds for computing the solvent-excluded domain.
// We will work one patch at a time to avoid having to store the level set
// function for the solvent-accessible domain.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedSeeds(const GridGeometry<_D, N, _T>& grid,
                     const std::vector<geom::Ball<_T, _D> >& balls,
                     const _T probeRadius,
                     std::vector<geom::Ball<_T, _D> >* seeds)
{
  typedef GridGeometry<_D, N, _T> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::Point Point;
  typedef geom::Ball<_T, _D> Ball;
  typedef container::SimpleMultiIndexExtentsIterator<_D> Iterator;

  // The extents of a patch.
  const IndexList PatchExtents = ext::filled_array<IndexList>(N);
  const IndexList CenterIndex = ext::filled_array<IndexList>(N / 2);

  // The threshold for accepting a point as a seed.
  // 1.01 * (diagonal length of a voxel)
  // This is also how far we must compute the distance beyond the zero
  // iso-surface of the solvent-accessible surface.
  const _T seedThreshold = 1.01 * std::sqrt(_T(_D)) * grid.spacing;

  //
  // Determine the patch-ball dependencies for computing the
  // solvent-accessible domain.
  //
  container::StaticArrayOfArrays<unsigned> dependencies;
  std::vector<Ball> offsetBalls(balls);
  {
    // Expand the balls' radii to include the probe radius and the
    // volume of calculated distance.
    const _T offset = probeRadius + seedThreshold;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += offset;
    }
    // Calculate the dependencies.
    patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }
  // For each patch, put the closest balls first. This will improve the
  // performance of the following distance calculation as we stop when
  // the distance becomes negative.
  putClosestBallsFirst(grid, offsetBalls, &dependencies);

  // Change back to the probe radius offset balls.
  for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
    offsetBalls[i].radius -= seedThreshold;
  }

  //
  // Loop over the patches, computing the solvent-accessible domain and
  // determining the seeds. The seeds have distances in the range
  // [0..seedThreshold].
  //
  std::vector<Ball> influencingBalls;
  Ball seed;
  Point x;
  _T minDistance, d;
  const Iterator pEnd = Iterator::end(grid.gridExtents);
  for (Iterator p = Iterator::begin(grid.gridExtents); p != pEnd; ++p) {
    const std::size_t index = grid.arrayIndex(*p);
    // If the patch has no dependent balls, do nothing.
    if (dependencies.empty(index)) {
      continue;
    }
    // Collect the list of balls that influence this patch.
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(index); ++n) {
      influencingBalls.push_back(offsetBalls[dependencies(index, n)]);
    }
    // Loop over the grid points in the patch.
    const Iterator iEnd = Iterator::end(PatchExtents);
    for (Iterator i = Iterator::begin(PatchExtents); i != iEnd; ++i) {
      // The position of the grid point.
      x = grid.indexToLocation(*p, *i);
      minDistance = std::numeric_limits<_T>::infinity();
      // Loop over the influencing balls and compute the distance to their
      // union. (The distance is only guaranteed to be correct for positive
      // distances.)
      for (std::size_t n = 0; n != influencingBalls.size(); ++n) {
        d = distance(influencingBalls[n], x);
        if (d < minDistance) {
          minDistance = d;
        }
        // Early exit. We are not interested in points with negative
        // distance.
        if (minDistance <= 0) {
          break;
        }
      }
      // Check the distance to seed if this is a seed point.
      if (0 < minDistance && minDistance < seedThreshold) {
        seed.center = x;
        seed.radius = probeRadius + minDistance;
        seeds->push_back(seed);
      }
    }

    //
    // Try placing a seed at the center.
    //
    // The position of the grid point.
    x = grid.indexToLocation(*p, CenterIndex);
    minDistance = std::numeric_limits<_T>::infinity();
    // Loop over the influencing balls and compute the distance to their
    // union.
    for (std::size_t n = 0; n != influencingBalls.size(); ++n) {
      d = distance(influencingBalls[n], x);
      if (d < minDistance) {
        minDistance = d;
      }
      // Early exit. We are not interested in points with negative distance.
      if (minDistance <= 0) {
        break;
      }
    }
    // If the distance is positive, we'll put a seed here.
    if (minDistance > 0) {
      seed.center = x;
      seed.radius = probeRadius + minDistance;
      seeds->push_back(seed);
    }
  }
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventExcludedUsingSeeds(Grid<_T, _D, N>* grid,
                          const std::vector<geom::Ball<_T, _D> >& balls,
                          const _T probeRadius)
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // The threshold for accepting a point as a seed.
  // 1.01 * (diagonal length of a voxel)
  // This is also how far we must compute the distance beyond the zero
  // iso-surface of the solvent-accessible surface.
  const _T seedThreshold = 1.01 * std::sqrt(double(_D)) * grid->spacing;

  // Compute the solvent-accessible domain. For the purpose of refining the
  // grid, the balls will be enlarged by probeRadius + seedThreshold. Any
  // patch that intersects these balls will be refined. The level set will
  // be computed on all refined patches. However, only points outside
  // the balls are guaranteed to have the correct Euclidean distance.
  {
    std::vector<geom::Ball<_T, _D> > probeOffsetBalls(balls);
    for (std::size_t i = 0; i != probeOffsetBalls.size(); ++i) {
      probeOffsetBalls[i].radius += probeRadius;
    }
    positiveDistanceAllRefined(grid, probeOffsetBalls, seedThreshold);
  }
#if 0
  // CONTINUE REMOVE
  std::cout << "positiveDistanceOutside\n"
            << "probeRadius = " << probeRadius << '\n';
  printInfo(*grid, std::cout);
  std::cout << "\n";
#endif

  // Determine the seeds used in computing the solvent-excluded domain.
  std::vector<geom::Ball<_T, _D> > seeds;
  std::vector<bool> areFarAway(grid->size(), false);
  solventExcludedSeeds(*grid, probeRadius, seedThreshold, &seeds, &areFarAway);
#if 0
  // CONTINUE REMOVE
  std::cout << "Seeds =\n" << seeds << '\n';
#endif

  // We will use the solvent-accessible grid (with its set of refined patches)
  // to compute the solvent-excluded domain.
  // From this point on, we will work only with the refined grid points.

  // Set all of the refined grid points to infinity.
  std::fill(grid->data(), grid->data() + grid->numVertices(),
            std::numeric_limits<_T>::infinity());
#if 0
  // Unrefined points.
  const Iterator pEnd = Iterator::end(grid->extents());
  for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
    VertexPatch& patch = grid(*p);
    if (! patch.isRefined()) {
      patch.fillValue = std::numeric_limits<_T>::infinity();
    }
  }
#endif

  // Now we work with the seeds to calculate the solvent-excluded domain.
  // Compute the patch dependencies for all patches.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // We need to compute the distance a little past the surface of the seeds.
    // Offset by the diagonal length of a voxel.
    std::vector<geom::Ball<_T, _D> > offsetSeeds(seeds);
    for (std::size_t i = 0; i != offsetSeeds.size(); ++i) {
      offsetSeeds[i].radius += seedThreshold;
    }
    patchDependencies(*grid, offsetSeeds.begin(), offsetSeeds.end(),
                      &dependencies);
  }
#if 0
  // CONTINUE REMOVE
  std::cout << "dependencies =\n" << dependencies << '\n';
#endif

  // CONTINUE: I will need to do this for the CUDA implementation.
# if 0
  // From the dependencies for all patches, calculate the dependencies
  // for only the refined patches.
  container::StaticArrayOfArrays<unsigned> refinedDependencies;
  {
    // Record the number of dependencies for each of the refined patches.
    std::vector<std::size_t> sizes(dependencies.getNumberOfArrays());
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      if ((*grid)[i].isRefined()) {
        sizes[i] = dependencies.size(i);
      }
      else {
        // We ignore dependencies for the unrefined patches.
        sizes[i] = 0;
      }
    }
    // Allocate memory.
    refinedDependencies.rebuild(sizes.begin(), sizes.end());
    // Copy the dependencies for the refined patches.
    for (std::size_t i = 0; i != sizes.size(); ++i) {
      if (! refinedDependencies.empty()) {
        std::copy(dependencies.begin(i), dependencies.end(i),
                  refinedDependencies.begin(i));
      }
    }
  }
#endif

  // In computing the power distance, we only need it to be correct up to
  // seedThreshold away from the surface. Let r be the radius of a ball,
  // d the distance from its center, and t the threshold. Suppose the
  // distance to the ball is negative. If r - d > t, then the point is
  // far inside the ball. Note however, that we are dealing with the
  // power distance; we compute d^2 - r^2. We Manipulate the
  // inequality.
  // r - d > t
  // d - r < -t
  // d^2 - r^2 < -t (d + r)
  // For negative distances d < r, so we use the following test.
  // d^2 - r^2 < -2 r t
  _T maxRadius = 0;
  for (std::size_t i = 0; i != seeds.size(); ++i) {
    if (seeds[i].radius > maxRadius) {
      maxRadius = seeds[i].radius;
    }
  }
  const _T powerDistanceThreshold = 2 * maxRadius * seedThreshold;

  // Compute the power distance to the seeds.
  {
    std::vector<geom::BallSquared<_T, _D> > influencingBalls;
    geom::BallSquared<_T, _D> bs;
    // Loop over the patches.
    const Iterator end = Iterator::end(grid->extents());
    for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
      const std::size_t index = grid->arrayIndex(*i);
      // Ignore the unrefined patches.
      if (!(*grid)[index].isRefined()) {
        continue;
      }
      // If the whole patch is far away from the solvent-excluded domain.
      if (areFarAway[index]) {
        std::fill((*grid)[index].begin(), (*grid)[index].end(),
                  -seedThreshold);
        continue;
      }
      // Build the parameters.
      influencingBalls.clear();
      for (std::size_t n = 0; n != dependencies.size(index); ++n) {
        const geom::Ball<_T, _D>& b = seeds[dependencies(index, n)];
        bs.center = b.center;
        bs.squaredRadius = b.radius * b.radius;
        influencingBalls.push_back(bs);
      }
#if 0
      // CONTINUE REMOVE
      std::cout << "influencingBalls =\n" << influencingBalls << '\n';
#endif
      // Compute the power distance.
      // This function accounts for 90% of the execution time.
      powerDistance(&(*grid)[index], grid->getPatchLowerCorner(*i),
                    grid->spacing, &influencingBalls,
                    powerDistanceThreshold);
    }
  }
#if 0
  // CONTINUE REMOVE
  std::cout << "powerDistance\n";
  printInfo(*grid, std::cout);
  std::cout << "\n";
#endif

  // Reverse the sign of the distance to get the solvent-excluded domain.
  _T* const end = grid->data() + grid->numVertices();
  for (_T* i = grid->data(); i != end; ++i) {
    *i = - *i;
  }
}


} // namespace levelSet
}
