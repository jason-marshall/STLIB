// -*- C++ -*-

#if !defined(__levelSet_dependencies_ipp__)
#error This file is an implementation detail of dependencies.
#endif

namespace stlib
{
namespace levelSet
{


// For each of the patches, put the ball that is closest to the center first
// in the list of dependencies. This can improve performance when computing
// the distance in a narrow neighborhood of the zero iso-surface.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
putClosestBallsFirst(const GridGeometry<_D, N, _T>& grid,
                     const std::vector<geom::Ball<_T, _D> >& balls,
                     container::StaticArrayOfArrays<unsigned>* dependencies)
{
  typedef typename GridGeometry<_D, N, _T>::Point Point;
  typedef container::SimpleMultiIndexExtentsIterator<_D> Iterator;

  // Loop over the patches.
  Point patchCenter;
  std::size_t minIndex;
  _T minDistance;
  _T d;
  const Iterator pEnd = Iterator::end(grid.gridExtents);
  for (Iterator p = Iterator::begin(grid.gridExtents); p != pEnd; ++p) {
    const std::size_t index = grid.arrayIndex(*p);
    // Ignore patches without any dependencies.
    if (dependencies->empty(index)) {
      continue;
    }
    patchCenter = grid.getVertexPatchCenter(*p);
    minIndex = 0;
    minDistance = ext::squaredDistance(patchCenter,
                                       balls[(*dependencies)(index, 0)].center);
    for (std::size_t i = 1; i != dependencies->size(index); ++i) {
      d = ext::squaredDistance(patchCenter,
                               balls[(*dependencies)(index, i)].center);
      if (d < minDistance) {
        minIndex = i;
        minDistance = d;
      }
    }
    std::swap((*dependencies)(index, 0), (*dependencies)(index, minIndex));
  }
}


// For each of the patches, put the ball that is closest to the center first
// in the list of dependencies. This can improve performance when computing
// the distance in a narrow neighborhood of the zero iso-surface.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
putClosestBallsFirst(const GridGeometry<_D, N, _T>& grid,
                     const std::vector<bool>& isActive,
                     const std::vector<geom::Ball<_T, _D> >& balls,
                     container::StaticArrayOfArrays<unsigned>* dependencies)
{
  typedef typename GridGeometry<_D, N, _T>::Point Point;
  typedef container::SimpleMultiIndexExtentsIterator<_D> Iterator;

  // Loop over the patches.
  Point patchCenter;
  std::size_t minIndex;
  _T minDistance;
  _T d;
  // The index for the dependencies.
  std::size_t n = 0;
  const Iterator pEnd = Iterator::end(grid.gridExtents);
  for (Iterator p = Iterator::begin(grid.gridExtents); p != pEnd; ++p) {
    const std::size_t index = grid.arrayIndex(*p);
    // Ignore inactive patches.
    if (! isActive[index]) {
      continue;
    }
    if (! dependencies->empty(n)) {
      patchCenter = grid.getVertexPatchCenter(*p);
      minIndex = 0;
      minDistance = squaredDistance(patchCenter,
                                    balls[(*dependencies)(n, 0)].center);
      for (std::size_t i = 1; i != dependencies->size(n); ++i) {
        d = squaredDistance(patchCenter,
                            balls[(*dependencies)(n, i)].center);
        if (d < minDistance) {
          minIndex = i;
          minDistance = d;
        }
      }
      std::swap((*dependencies)(n, 0), (*dependencies)(n, minIndex));
    }
    ++n;
  }
  assert(n == dependencies->getNumberOfArrays());
}


} // namespace levelSet
}
