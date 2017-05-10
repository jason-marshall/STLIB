// -*- C++ -*-

#if !defined(__levelSet_subtract_ipp__)
#error This file is an implementation detail of subtract.
#endif

namespace stlib
{
namespace levelSet
{


// Subtract the balls from the level set. Compute the distance up to
// maxDistance past the surface of the balls.
template<typename _T, std::size_t _D>
inline
void
subtract(container::SimpleMultiArrayRef<_T, _D>* grid,
         const geom::BBox<_T, _D>& domain,
         const std::vector<geom::Ball<_T, _D> >& balls,
         const _T maxDistance)
{
  typedef container::SimpleMultiArrayRef<_T, _D> SimpleMultiArray;
  typedef typename SimpleMultiArray::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  geom::SimpleRegularGrid<_T, _D> regularGrid(grid->extents(), domain);
  geom::BBox<_T, _D> window;
  Range range;
  std::array<_T, _D> p;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Build a bounding box around the ball.
    window = geom::specificBBox<geom::BBox<_T, _D> >(balls[i]);
    // Expand by two grid cells.
    offset(&window, maxDistance);
    // Compute the index range for the bounding box.
    range = regularGrid.computeRange(window);
    // For each index in the range.
    const Iterator end = Iterator::end(range);
    for (Iterator index = Iterator::begin(range); index != end; ++index) {
      // Convert the index to a Cartesian location.
      p = regularGrid.indexToLocation(*index);
      // Compute the signed distance to the surface of the ball.
      // Use this to subtract the ball at this point.
      (*grid)(*index) = difference((*grid)(*index), distance(balls[i], p));
    }
  }
}


// Subtract the balls from the level set.
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
inline
void
subtract(container::EquilateralArrayImp<_T, _D, N, _Base>* patch,
         const std::array<_T, _D>& lowerCorner,
         const _T spacing,
         const std::vector<geom::Ball<_T, _D> >& balls)
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  std::array<_T, _D> p;
  // For each grid point.
  const Iterator end = Iterator::end(patch->extents());
  for (Iterator i = Iterator::begin(patch->extents()); i != end; ++i) {
    _T& g = (*patch)(*i);
    p = lowerCorner + spacing * stlib::ext::convert_array<_T>(*i);
    // For each ball.
    for (std::size_t j = 0; j != balls.size(); ++j) {
      // Compute the signed distance to the surface of the ball.
      // Use this to subtract the ball at this point.
      g = difference(g, distance(balls[j], p));
    }
  }
}


// Subtract the balls from the level set. Compute the distance up to
// maxDistance past the surface of the balls.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
subtract(Grid<_T, _D, N>* grid,
         const std::vector<geom::Ball<_T, _D> >& balls,
         const _T maxDistance)
{
  typedef typename Grid<_T, _D, N>::VertexPatch VertexPatch;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Determine the patch/ball dependencies for subtracting the atoms.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    std::vector<geom::Ball<_T, _D> > offsetBalls = balls;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += maxDistance;
    }
    patchDependencies(*grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

  // Loop over the patches.
  std::vector<geom::Ball<_T, _D> > influencingBalls;
  const Iterator end = Iterator::end(grid->extents());
  for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
    const std::size_t index = grid->arrayIndex(*i);
    VertexPatch& patch = (*grid)[index];
    // Ignore the unrefined patches.
    if (! patch.isRefined()) {
      continue;
    }
    // Build the parameters for subtracting the balls.
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(index); ++n) {
      influencingBalls.push_back(balls[dependencies(index, n)]);
    }
    // Subtract the balls.
    subtract(&patch, grid->getPatchLowerCorner(*i), grid->spacing,
             influencingBalls);
  }
}


} // namespace levelSet
}
