// -*- C++ -*-

#if !defined(__levelSet_solventAccessibleCavities_ipp__)
#error This file is an implementation detail of solventExcluded.
#endif

namespace stlib
{
namespace levelSet
{


// The queue method is deprecated.
template<typename _T, std::size_t _D>
inline
void
solventAccessibleCavities(container::SimpleMultiArrayRef<_T, _D>* grid,
                          const geom::BBox<_T, _D>& domain,
                          const std::vector<geom::Ball<_T, _D> >& balls,
                          const _T probeRadius)
{
  typedef container::SimpleMultiArrayRef<_T, _D> SimpleMultiArray;
  typedef typename SimpleMultiArray::IndexList IndexList;
  typedef typename SimpleMultiArray::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  const _T Inf = std::numeric_limits<_T>::infinity();
  assert(grid->extents()[0] > 1);
  const _T dx = (domain.upper[0] - domain.lower[0]) / (grid->extents()[0] - 1);
  // 1.1 * (diagonal length of a voxel)
  const _T buffer = 1.1 * std::sqrt(double(_D)) * dx;

  // Compute the solvent-center excluded domain.
  {
    // Compute the distance a little bit past the zero iso-surface in order
    // to get an accurate initial condition for the static H-J problem.
    std::vector<geom::Ball<_T, _D> > offset = balls;
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    std::fill(grid->begin(), grid->end(),
              std::numeric_limits<_T>::quiet_NaN());
    positiveDistance(grid, domain, offset, _T(0), buffer);
  }

  // Reverse the sign to get the solvent-center accessible domain.
  for (typename SimpleMultiArray::iterator i = grid->begin(); i != grid->end();
       ++i) {
    *i = -*i;
  }

  // Mark the outside component of the solvent-center-accesible domain as
  // positive far-away.
  std::deque<IndexList> queue;
  IndexList lower, upper;
  // Start with the lower corner.
  IndexList i = ext::filled_array<IndexList>(0);
  assert((*grid)(i) != (*grid)(i) || (*grid)(i) <= 0);
  (*grid)(i) = Inf;
  queue.push_back(i);
  // Set the rest of the points outside to positive far-away.
  while (! queue.empty()) {
    i = queue.front();
    queue.pop_front();
    // Examine all neighboring grid points. There are 3^_D - 1 neighbors.
    // Note that it is not sufficient to just examine the 2*_D adjacent
    // neighbors. Crevices on the surface of the solvent-center-accessible
    // surface can isolate grid points.
    for (std::size_t n = 0; n != _D; ++n) {
      if (i[n] == 0) {
        lower[n] = 0;
      }
      else {
        lower[n] = i[n] - 1;
      }
      upper[n] = std::min(i[n] + 2, grid->extents()[n]);
    }
    const Range range = {upper - lower, lower};
    const Iterator end = Iterator::end(range);
    for (Iterator j = Iterator::begin(range); j != end; ++j) {
      if (*j == i) {
        continue;
      }
      if ((*grid)(*j) != (*grid)(*j) || (*grid)(*j) <= 0) {
        queue.push_back(*j);
        (*grid)(*j) = Inf;
      }
    }
  }

  // The NaN points are inside the solvent-center-accessible cavities.
  // Erase the points with positive distance. The distance from the
  // SCA will be computed with the H-J solver.
  for (typename SimpleMultiArray::iterator i = grid->begin(); i != grid->end();
       ++i) {
    if (*i > 0) {
      // Note that the H-J solver uses max() to denote an unknown point.
      *i = std::numeric_limits<_T>::max();
    }
    else if (*i != *i) {
      *i = -Inf;
    }
  }

  // If there are no non-positive points, there are no solvent-accessible
  // cavities.
  if (! hasNonPositive(grid->begin(), grid->end())) {
    // Fill the grid with positive far-away values and return.
    std::fill(grid->begin(), grid->end(), Inf);
    return;
  }

  // CONTINUE REMOVE
#if 1
  std::cerr << "Before H-J solver.\n";
  printLevelSetInfo(grid->begin(), grid->end(), std::cerr);
#endif
  // Compute the distance from the SCA interface.
  container::MultiArrayRef<_T, _D> gridRef(grid->data(), grid->extents());
  hj::computeUnsignedDistance(gridRef, dx, probeRadius + buffer);

  // Offset to get the solvent accessible surface.
  *grid -= probeRadius;
  // CONTINUE REMOVE
#if 1
  std::cerr << "After H-J solver.\n";
  printLevelSetInfo(grid->begin(), grid->end(), std::cerr);
#endif
}


template<typename _T, std::size_t _D>
inline
void
solventAccessibleCavitiesOld(container::SimpleMultiArrayRef<_T, _D>* grid,
                             const geom::BBox<_T, _D>& domain,
                             const std::vector<geom::Ball<_T, _D> >& balls,
                             const _T probeRadius)
{
  typedef container::SimpleMultiArrayRef<_T, _D> SimpleMultiArray;
  typedef typename SimpleMultiArray::IndexList IndexList;
  typedef typename SimpleMultiArray::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  const _T Inf = std::numeric_limits<_T>::infinity();

  // Compute the solvent-center excluded domain.
  {
    std::vector<geom::Ball<_T, _D> > offset = balls;
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    negativeDistance(grid, domain, offset);
  }

#if 0
  std::cerr << "Solvent-center excluded domain\n";
  printLevelSetInfo(*grid, std::cerr);
#endif

  // Mark the outside component of the solvent-center-accesible domain as
  // positive far-away.
  const IndexList closedUpper = grid->extents() - 1;
  std::deque<IndexList> queue;
  IndexList lower, upper;
  // Start with the lower corner.
  IndexList i = ext::filled_array<IndexList>(0);
  assert((*grid)(i) != (*grid)(i));
  (*grid)(i) = Inf;
  queue.push_back(i);
  // Set the rest of the points outside to positive far-away.
  while (! queue.empty()) {
    i = queue.front();
    queue.pop_front();
    // Examine all neighboring grid points. There are 3^_D - 1 neighbors.
    // Note that it is not sufficient to just examine the 2*_D adjacent
    // neighbors. Crevices on the surface of the solvent-center-accessible
    // surface can isolate grid points.
    for (std::size_t n = 0; n != _D; ++n) {
      if (i[n] == 0) {
        lower[n] = 0;
      }
      else {
        lower[n] = i[n] - 1;
      }
      upper[n] = std::min(i[n] + 2, grid->extents()[n]);
    }
    const Range range = {upper - lower, lower};
    const Iterator end = Iterator::end(range);
    for (Iterator j = Iterator::begin(range); j != end; ++j) {
      if (*j == i) {
        continue;
      }
      if ((*grid)(*j) != (*grid)(*j)) {
        queue.push_back(*j);
        (*grid)(*j) = Inf;
      }
    }
  }

#if 0
  std::cerr << "\nOutside far away:\n";
  printLevelSetInfo(*grid, std::cerr);
#endif

  // If there are no unknown points, there are no solvent-accessible cavities.
  if (! hasUnknown(grid->begin(), grid->end())) {
    // Fill the grid with positive far-away values and return.
    std::fill(grid->begin(), grid->end(), Inf);
    return;
  }

  // Get the boundary of the solvent-center-accessible cavities.
  Iterator end = Iterator::end(grid->extents());
  for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
    if ((*grid)(*i) < 0 && hasUnknownAdjacentNeighbor(*grid, *i)) {
      (*grid)(*i) = -(*grid)(*i);
      queue.push_back(*i);
    }
  }

  // Walk downhill and reverse the sign of the distance.
  while (! queue.empty()) {
    i = queue.front();
    queue.pop_front();
    const _T x = (*grid)(i);
    // Examine each adjacent grid point.
    for (std::size_t n = 0; n != _D; ++n) {
      if (i[n] != closedUpper[n]) {
        ++i[n];
        if ((*grid)(i) < 0 && -(*grid)(i) > x) {
          (*grid)(i) = -(*grid)(i);
          queue.push_back(i);
        }
        --i[n];
      }
      if (i[n] != 0) {
        --i[n];
        if ((*grid)(i) < 0 && -(*grid)(i) > x) {
          (*grid)(i) = -(*grid)(i);
          queue.push_back(i);
        }
        ++i[n];
      }
    }
  }

  // Set the unknown distances to negative far-away. These are the
  // solvent-center-accessible portion of the cavities.
  // Set all of the negative distances to positive far-away. These are not
  // part of the solvent-accessible cavities.
  for (std::size_t i = 0; i != grid->size(); ++i) {
    _T& x = (*grid)[i];
    if (x != x) {
      x = -Inf;
    }
    else if (x < 0) {
      x = Inf;
    }
  }

  // Offset to get the solvent accessible surface.
  *grid -= probeRadius;

#if 0
  std::cerr << "\nSolvent-accessible cavities:\n";
  printLevelSetInfo(*grid, std::cerr);
#endif
}


// This is an old method.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventAccessibleCavitiesWithSeedInterpolation
(Grid<_T, _D, N>* grid,
 const std::vector<geom::Ball<_T, _D> >& balls,
 const _T probeRadius)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::DualIndices DualIndices;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  const _T Inf = std::numeric_limits<_T>::infinity();

  // For this method to work, the probe radius must be at least as large
  // as the diagonal length of a patch. Consider the case that a patch
  // has a single seed in its lower corner. The seed sphere must cover
  // all of the patch grid points in order to ensure that they are
  // correctly marked as being in the solvent-accessible cavity.
  assert(probeRadius > grid->spacing * N * std::sqrt(3));

  // Compute the solvent-center excluded domain.
  {
    std::vector<geom::Ball<_T, _D> > offset(balls);
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    negativePowerDistance(grid, offset);
  }

  // Mark the outside component of the solvent-center-accesible domain as
  // negative far-away.
  std::deque<DualIndices> queue;
  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Start with the lower corner.
  DualIndices di;
  di.first = di.second = ext::filled_array<IndexList>(0);
  assert((*grid)(di) != (*grid)(di) || (*grid)(di) > 0);
  (*grid)(di) = -Inf;
  queue.push_back(di);
  // Set the rest of the points outside to negative far-away.
  while (! queue.empty()) {
    di = queue.front();
    queue.pop_front();
    // Examine the set of all neighboring grid points.
    neighbors.clear();
    grid->allNeighbors(di, insertIterator);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      const DualIndices& n = neighbors[i];
      if ((*grid)(n) != (*grid)(n) || (*grid)(n) > 0) {
        queue.push_back(n);
        (*grid)(n) = -Inf;
      }
    }
  }

  // Record the points at the boundary of the solvent-center-accessible
  // cavities. These are positive points with negative neighbors, and will
  // be used as seed points to compute the solvent-accessible cavities.
  std::vector<geom::Ball<_T, _D> > seeds;
  {
    geom::Ball<_T, _D> seed;
    seed.radius = probeRadius;
    std::vector<DualIndices> neighbors;
    std::back_insert_iterator<std::vector<DualIndices> >
    insertIterator(neighbors);
    const Iterator pEnd = Iterator::end(grid->extents());
    for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
      VertexPatch& patch = (*grid)(*p);
      if (! patch.isRefined()) {
        continue;
      }
      // Loop over the grid points in the patch.
      const Iterator iEnd = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
        if (patch(*i) > 0) {
          const _T a = patch(*i);
          // For each adjacent neighbor.
          for (std::size_t direction = 0; direction != 2 * _D;
               ++direction) {
            neighbors.clear();
            grid->adjacentNeighbors(*p, *i, direction, insertIterator);
            if (neighbors.empty()) {
              continue;
            }
            const _T b = (*grid)(neighbors[0]);
            if (b <= 0) {
              seed.center = grid->indexToLocation(*p, *i);
              // Use linear interpolation to place the seed between
              // this grid point and its adjacent neighbor.
              // a + t (b - a) == 0
              // t = a / (a - b)
              seed.center[direction / 2] += (2 * (direction % 2) - 1) *
                                            a / (a - b) * grid->spacing;
              seeds.push_back(seed);
            }
          }
        }
      }
    }
  }

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
  negativePowerDistance(grid, seeds);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
markOutsideAsNegativeInfQueue(Grid<_T, _D, N>* grid)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::DualIndices DualIndices;

  const _T Inf = std::numeric_limits<_T>::infinity();

  std::deque<DualIndices> queue;
  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Start with the lower corner.
  DualIndices di;
  di.first = di.second = ext::filled_array<IndexList>(0);
  assert((*grid)(di) != (*grid)(di) || (*grid)(di) > 0);
  (*grid)(di) = -Inf;
  queue.push_back(di);
  // Set the rest of the points outside to negative far-away.
  while (! queue.empty()) {
    di = queue.front();
    queue.pop_front();
    // Examine the set of all neighboring grid points.
    neighbors.clear();
    grid->allNeighbors(di, insertIterator);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      const DualIndices& n = neighbors[i];
      if ((*grid)(n) != (*grid)(n) || (*grid)(n) > 0) {
        queue.push_back(n);
        (*grid)(n) = -Inf;
      }
    }
  }
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventAccessibleCavitySeeds(const Grid<_T, _D, N>& grid, const _T probeRadius,
                             std::vector<geom::Ball<_T, _D> >* seeds)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::Point Point;
  typedef typename Grid::DualIndices DualIndices;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // The length of a vertex patch is grid.spacing * (N - 1).
  const _T HalfLength = 0.5 * grid.spacing * (N - 1);
  // We use the half diagonal length (expanded by a small amount to account
  // truncation errors) for seeds that cover a patch.
  const _T WholePatchSeedRadius = 1.001 * HalfLength * std::sqrt(_D);
  // The threshold for accepting a point as a seed.
  const _T seedThreshold = 1.01 * grid.spacing;

  _T seedRadius = 0;
  std::vector<IndexList> centerIndices;
  std::vector<Point> centerOffsets;
  positiveCoveringSeedParameters(grid, probeRadius, &seedRadius,
                                 &centerIndices, &centerOffsets);

  geom::Ball<_T, _D> seed;
  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  DualIndices di;
  const Iterator pEnd = Iterator::end(grid.extents());
  for (Iterator p = Iterator::begin(grid.extents()); p != pEnd; ++p) {
    const VertexPatch& patch = grid(*p);
    di.first = *p;
    if (patch.isRefined()) {
#if 0
      // If the interpolated value at the center is positive, place a
      // seed at that point to cover the grid points in this patch.
      if (interpolateCenter(patch) > 0) {
        seed.center = grid.getPatchLowerCorner(*p);
        seed.center += HalfLength;
        seed.radius = WholePatchSeedRadius;
        seeds->push_back(seed);
      }
#endif
      positiveCoveringSeeds(grid, *p, seedRadius, centerIndices,
                            centerOffsets, seeds);

      // Loop over the grid points in the patch.
      const Iterator iEnd = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
        const _T a = patch(*i);
        // A seed must be near the interface.
        if (a > 0 && a < seedThreshold) {
          // A seed must have a neighbor with positive distance.
          // This requirement excludes cavities that result from the
          // solvent being pinched between two atoms.
          // Examine the set of all neighboring grid points.
          neighbors.clear();
          di.second = *i;
          grid.allNeighbors(di, insertIterator);
          for (std::size_t j = 0; j != neighbors.size(); ++j) {
            const DualIndices& n = neighbors[j];
            if (grid(n) > 0) {
              seed.radius = probeRadius + a;
              seed.center = grid.indexToLocation(*p, *i);
              seeds->push_back(seed);
              break;
            }
          }
        }
      }
    }
    // Unrefined patches.
    else {
      // Check for unrefined patches in the middle of cavities.
      if (patch.fillValue != patch.fillValue) {
        // Add a seed that covers the patch.
        seed.center = grid.getPatchLowerCorner(*p);
        seed.center += HalfLength;
        seed.radius = WholePatchSeedRadius;
        seeds->push_back(seed);
      }
    }
  }
}


// Wrapper for markOutsideAsNegativeInfQueue(). Note that
// markOutsideAsNegativeInf(), defined in outside.h, is only defined for 3-D
// grids.
template<typename _T, std::size_t N>
inline
void
markOutsideAsNegativeInf(Grid<_T, 2, N>* grid)
{
  markOutsideAsNegativeInfQueue(grid);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventAccessibleCavities(Grid<_T, _D, N>* grid,
                          std::vector<geom::Ball<_T, _D> > balls,
                          const _T probeRadius)
{
  const _T Inf = std::numeric_limits<_T>::infinity();

  // Compute the solvent-center excluded domain.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    balls[i].radius += probeRadius;
  }
  positiveDistanceOutside(grid, balls, 2 * grid->spacing);

  // Mark the outside component of the solvent-center-accesible domain as
  // negative far-away.
  markOutsideAsNegativeInf(grid);

  // Record the points at the boundary of the solvent-center-accessible
  // cavities. These are positive points with negative neighbors, and will
  // be used as seed points to compute the solvent-accessible cavities.
  std::vector<geom::Ball<_T, _D> > seeds;
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
  negativePowerDistance(grid, seeds);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
solventAccessibleCavitiesOld(Grid<_T, _D, N>* grid,
                             const std::vector<geom::Ball<_T, _D> >& balls,
                             const _T probeRadius)
{
  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::DualIndices DualIndices;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  const _T Inf = std::numeric_limits<_T>::infinity();

  // Compute the solvent-center excluded domain.
  {
    std::vector<geom::Ball<_T, _D> > offset = balls;
    for (std::size_t i = 0; i != offset.size(); ++i) {
      offset[i].radius += probeRadius;
    }
    negativeDistance(grid, offset);
  }

  // Mark the outside component of the solvent-center-accesible domain as
  // positive far-away.
  std::deque<DualIndices> queue;
  std::vector<DualIndices> neighbors;
  std::back_insert_iterator<std::vector<DualIndices> >
  insertIterator(neighbors);
  // Start with the lower corner.
  DualIndices di;
  di.first = di.second = ext::filled_array<IndexList>(0);
  assert((*grid)(di) != (*grid)(di));
  (*grid)(di) = Inf;
  queue.push_back(di);
  // Set the rest of the points outside to positive far-away.
  while (! queue.empty()) {
    di = queue.front();
    queue.pop_front();
    // Examine the set of all neighboring grid points.
    neighbors.clear();
    grid->allNeighbors(di, insertIterator);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      const DualIndices& n = neighbors[i];
      if ((*grid)(n) != (*grid)(n)) {
        queue.push_back(n);
        (*grid)(n) = Inf;
      }
    }
  }

#if 1
  // CONTINUE: REMOVE
  printLevelSetInfo(*grid, std::cout);
#endif
#if 0
  {
    // CONTINUE: REMOVE
    // Print the unknown points.
    const Iterator pEnd = Iterator::end(grid->extents());
    for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
      VertexPatch& patch = (*grid)(*p);
      if (! patch.isRefined()) {
        continue;
      }
      // Loop over the grid points in the patch.
      const Iterator iEnd = Iterator::end(patch.extents());
      for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
        if (patch(*i) != patch(*i)) {
          std::cout << "Unknown: " << *p << ", " << *i << '\n';
          // Print the neighboring values.
          neighbors.clear();
          grid->allNeighbors(std::make_pair(*p, *i), insertIterator);
          for (std::size_t j = 0; j != neighbors.size(); ++j) {
            std::cout << (*grid)(neighbors[j]) << ' ';
          }
          std::cout << '\n';
        }
      }
    }
  }
#endif

  // If there are no unknown points, there are no solvent-accessible cavities.
  if (! hasUnknown(*grid)) {
    // Fill the grid with positive far-away values and return.
    grid->clear();
    for (std::size_t i = 0; i != grid->size(); ++i) {
      (*grid)[i].fillValue = Inf;
    }
    return;
  }

  // Get the boundary of the solvent-center-accessible cavities.
  // Loop over the refined patches.
  const Iterator pEnd = Iterator::end(grid->extents());
  for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
    VertexPatch& patch = (*grid)(*p);
    if (! patch.isRefined()) {
      continue;
    }
    // Loop over the grid points in the patch.
    const Iterator iEnd = Iterator::end(patch.extents());
    for (Iterator i = Iterator::begin(patch.extents()); i != iEnd; ++i) {
      di.first = *p;
      di.second = *i;
      if (patch(*i) < 0 && hasUnknownAdjacentNeighbor(*grid, di)) {
        patch(*i) = - patch(*i);
        queue.push_back(di);
      }
    }
  }

  // Walk downhill and reverse the sign of the distance.
  while (! queue.empty()) {
    di = queue.front();
    queue.pop_front();
    const _T x = (*grid)(di);
    // Examine the set of adjacent grid points.
    neighbors.clear();
    grid->adjacentNeighbors(di, insertIterator);
    for (std::size_t i = 0; i != neighbors.size(); ++i) {
      _T& y = (*grid)(neighbors[i]);
      if (y < 0 && -y > x) {
        y = - y;
        queue.push_back(neighbors[i]);
      }
    }
  }

  // Set the unknown distances to negative far-away. These are the
  // solvent-center-accessible portion of the cavities.
  // Set all of the negative distances to positive far-away. These are not
  // part of the solvent-accessible cavities.
  for (std::size_t i = 0; i != grid->size(); ++i) {
    VertexPatch& p = (*grid)[i];
    if (p.isRefined()) {
      for (std::size_t j = 0; j != p.size(); ++j) {
        _T& x = p[j];
        if (x != x) {
          x = -Inf;
        }
        else if (x < 0) {
          x = Inf;
        }
      }
    }
    else {
      _T& x = p.fillValue;
      if (x != x) {
        x = -Inf;
      }
      else if (x < 0) {
        x = Inf;
      }
    }
  }

  // Offset to get the solvent accessible surface.
  *grid -= probeRadius;
}


template<typename _T, std::size_t _D>
inline
void
solventAccessibleCavities(const std::vector<geom::Ball<_T, _D> >& balls,
                          const _T probeRadius, const _T targetGridSpacing,
                          std::vector<_T>* content,
                          std::vector<_T>* boundary)
{
  const std::size_t N = 8;

  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::BBox BBox;

  // Place a bounding box around the balls.
  BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // We expand by the probe radius plus the threshold for determining seeds.
  offset(&domain, probeRadius + targetGridSpacing * std::sqrt(_T(_D)));
  // Make the grid.
  Grid grid(domain, targetGridSpacing);

  // Calculate the solvent-accessible cavities.
  solventAccessibleCavities(&grid, balls, probeRadius);

  // Compute the content (volume) and boundary (surface area).
  levelSet::contentAndBoundary(grid, content, boundary);
}


template<typename _T, std::size_t _D>
inline
void
solventAccessibleCavitySeeds(std::vector<geom::Ball<_T, _D> > balls,
                             const _T probeRadius, const _T targetGridSpacing,
                             std::vector<geom::Ball<_T, _D> >* seeds)
{
  const std::size_t N = 8;

  typedef Grid<_T, _D, N> Grid;
  typedef typename Grid::BBox BBox;

  // Place a bounding box around the balls.
  BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // We expand by the probe radius plus the threshold for determining seeds.
  offset(&domain, probeRadius + targetGridSpacing * std::sqrt(_T(_D)));
  // Make the grid.
  Grid grid(domain, targetGridSpacing);

  // Compute the solvent-center excluded domain.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    balls[i].radius += probeRadius;
  }
  positiveDistanceOutside(&grid, balls, 2 * grid.spacing);

  // Mark the outside component of the solvent-center-accesible domain as
  // negative far-away.
  markOutsideAsNegativeInf(&grid);

  // Record the points at the boundary of the solvent-center-accessible
  // cavities. These are positive points with negative neighbors, and will
  // be used as seed points to compute the solvent-accessible cavities.
  solventAccessibleCavitySeeds(grid, probeRadius, seeds);
}


} // namespace levelSet
}
