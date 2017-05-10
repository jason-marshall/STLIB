// -*- C++ -*-

#if !defined(__levelSet_MolecularSurfaces_ipp__)
#error This file is an implementation detail of MolecularSurfaces.
#endif

namespace stlib
{
namespace levelSet
{


//
// Free functions.
//


// Return the maximum radius of the balls.
template<typename _T, std::size_t _D>
inline
_T
maxRadius(const std::vector<geom::Ball<_T, _D> >& balls)
{
  _T result = 0;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (balls[i].radius > result) {
      result = balls[i].radius;
    }
  }
  return result;
}


//
// Member functions.
//


template<typename _T, std::size_t _D, std::size_t N>
template<typename _OutputIterator>
inline
std::pair<_T, _T>
MolecularSurfaces<_T, _D, N>::
solventAccessible(_OutputIterator vertices) const
{
  // Increase the radii of the balls by the probe radius.
  std::vector<geom::Ball<_T, _D> > offset(atoms);
  for (std::size_t i = 0; i != offset.size(); ++i) {
    offset[i].radius += probeRadius;
  }
  return unionOfBalls(offset, vertices);
}


template<typename _T, std::size_t _D, std::size_t N>
template<typename _OutputIterator>
inline
std::pair<_T, _T>
MolecularSurfaces<_T, _D, N>::
solventExcluded(_OutputIterator vertices) const
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Increase the radii by the probe radius.
  std::vector<geom::Ball<_T, _D> > solventOffset = atoms;
  for (std::size_t i = 0; i != solventOffset.size(); ++i) {
    solventOffset[i].radius += probeRadius;
  }

  // Make the virtual grid.
  Grid grid(calculateDomain(solventOffset), targetGridSpacing);

  // Determine the patch/ball dependencies for solvent offset balls.
  container::StaticArrayOfArrays<unsigned> dependencies;
  negativeDistanceDependencies(grid, solventOffset, &dependencies);

  std::vector<geom::Ball<_T, _D> > influencingBalls;
  // Initialize content and boundary.
  _T content = 0;
  _T boundary = 0;
  _T patchContent = 0;
  _T patchBoundary = 0;
  std::pair<_T, _T> cb;
  Point lowerCorner;
  Patch patch;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.extents);
  for (Iterator i = Iterator::begin(grid.extents); i != end; ++i) {
    const Index index = grid.arrayIndex(*i);
    if (dependencies.empty(index)) {
      continue;
    }
    // Build the parameters for the solvent-excluded surface.
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(index); ++n) {
      influencingBalls.push_back(solventOffset[dependencies(index, n)]);
    }
    // Compute the solvent accessible surface with correct negative distance.
    lowerCorner = grid.getPatchLowerCorner(*i);
    negativeDistance(&patch, lowerCorner, grid.spacing, influencingBalls);
    // Offset by the probe radius to get the solvent excluded surface.
    patch += probeRadius;
    // Calculate the content and the boundary.
    contentAndBoundary(patch, lowerCorner, grid.spacing, &patchContent,
                       &patchBoundary, vertices);
    content += patchContent;
    boundary += patchBoundary;
  }
  return std::make_pair(content, boundary);
}


template<typename _T, std::size_t _D, std::size_t N>
template<typename _OutputIterator>
inline
std::pair<_T, _T>
MolecularSurfaces<_T, _D, N>::
solventExcludedCavities(_OutputIterator vertices) const
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Increase the radii by the probe radius.
  std::vector<geom::Ball<_T, _D> > solventOffset = atoms;
  for (std::size_t i = 0; i != solventOffset.size(); ++i) {
    solventOffset[i].radius += probeRadius;
  }

  // Make the virtual grid.
  Grid grid(calculateDomain(solventOffset), targetGridSpacing);

  // Determine the patch/ball dependencies for subtracting the balls.
  container::StaticArrayOfArrays<unsigned> differenceDependencies;
  patchDependencies(grid, atoms.begin(), atoms.end(), &differenceDependencies);
  // Determine the patch/ball dependencies for the solvent-offset balls.
  container::StaticArrayOfArrays<unsigned> offsetDependencies;
  negativeDistanceDependencies(grid, solventOffset, &offsetDependencies);

  std::vector<geom::Ball<_T, _D> > influencingBalls;
  // Initialize content and boundary.
  _T content = 0;
  _T boundary = 0;
  _T patchContent = 0;
  _T patchBoundary = 0;
  std::pair<_T, _T> cb;
  Point lowerCorner;
  Patch patch;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.extents);
  for (Iterator i = Iterator::begin(grid.extents); i != end; ++i) {
    const Index index = grid.arrayIndex(*i);
    if (offsetDependencies.empty(index)) {
      continue;
    }
    // Build the parameters for the solvent-excluded surface.
    influencingBalls.clear();
    for (std::size_t n = 0; n != offsetDependencies.size(index); ++n) {
      influencingBalls.push_back(solventOffset[offsetDependencies(index,
                                 n)]);
    }
    // Compute the solvent accessible surface with correct negative distance.
    lowerCorner = grid.getPatchLowerCorner(*i);
    negativeDistance(&patch, lowerCorner, grid.spacing, influencingBalls);
    // Offset by the probe radius to get the solvent excluded surface.
    patch += probeRadius;

    // Build the parameters for the subtracting the balls.
    influencingBalls.clear();
    for (std::size_t n = 0; n != differenceDependencies.size(index); ++n) {
      influencingBalls.push_back(atoms[differenceDependencies(index, n)]);
    }
    // Subtract the balls.
    subtract(&patch, lowerCorner, grid.spacing, influencingBalls);

    // Calculate the content and the boundary.
    contentAndBoundary(patch, lowerCorner, grid.spacing, &patchContent,
                       &patchBoundary, vertices);
    content += patchContent;
    boundary += patchBoundary;
  }
  return std::make_pair(content, boundary);
}


template<typename _T, std::size_t _D, std::size_t N>
template<typename _OutputIterator>
inline
std::pair<_T, _T>
MolecularSurfaces<_T, _D, N>::
unionOfBalls(const std::vector<geom::Ball<_T, _D> >& balls,
             _OutputIterator vertices) const
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Make the virtual grid.
  Grid grid(calculateDomain(balls), targetGridSpacing);

  // Determine the patch/ball dependencies.
  container::StaticArrayOfArrays<unsigned> dependencies;
  patchDependencies(grid, balls.begin(), balls.end(), &dependencies);

  Patch patch;
  Point lowerCorner;
  std::vector<geom::BallSquared<_T, _D> > influencingBalls;
  geom::BallSquared<_T, _D> bs;
  _T patchContent, patchBoundary;
  // Initialize content and boundary.
  _T content = 0;
  _T boundary = 0;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid.extents);
  for (Iterator i = Iterator::begin(grid.extents); i != end; ++i) {
    const Index index = grid.arrayIndex(*i);
    if (dependencies.empty(index)) {
      continue;
    }
    // Build the parameters.
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(index); ++n) {
      const geom::Ball<_T, _D>& b = balls[dependencies(index, n)];
      bs.center = b.center;
      bs.squaredRadius = b.radius * b.radius;
      influencingBalls.push_back(bs);
    }
    // CONTINUE: Check if the patch is entirely contained within a ball.
    // Compute the power distance.
    lowerCorner = grid.getPatchLowerCorner(*i);
    powerDistance(&patch, lowerCorner, grid.spacing, influencingBalls);
    // Calculate the content and the boundary.
    contentAndBoundary(patch, lowerCorner, grid.spacing, &patchContent,
                       &patchBoundary, vertices);
    content += patchContent;
    boundary += patchBoundary;
  }
  return std::make_pair(content, boundary);
}


template<typename _T, std::size_t _D, std::size_t N>
inline
void
MolecularSurfaces<_T, _D, N>::
negativeDistanceDependencies
(const Grid& grid, const std::vector<geom::Ball<_T, _D> >& balls,
 container::StaticArrayOfArrays<unsigned>* dependencies) const
{
  // A ball may influence points up to maxRadius beyond its surface.
  // Consider the case that one ball barely intersects the largest ball.
  // Then points all the way up to the center of the largest ball may
  // be closest to the intersection of the two balls.
  const _T offset = maxRadius(balls);
  std::vector<geom::Ball<_T, _D> > offsetBalls = balls;
  for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
    offsetBalls[i].radius += offset;
  }
  patchDependencies(grid, offsetBalls.begin(), offsetBalls.end(),
                    dependencies);
}


// Calculate an appropriate domain for the balls.
template<typename _T, std::size_t _D, std::size_t N>
inline
typename MolecularSurfaces<_T, _D, N>::BBox
MolecularSurfaces<_T, _D, N>::
calculateDomain(const std::vector<geom::Ball<_T, _D> >& balls) const
{
  // Place a bounding box around the atoms.
  BBox domain = geom::specificBBox<BBox>(balls.begin(), balls.end());
  // Add the target grid spacing to get one more grid point.
  offset(&domain, targetGridSpacing);
  return domain;
}

} // namespace levelSet
} // namespace stlib
