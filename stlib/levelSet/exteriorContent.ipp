// -*- C++ -*-

#if !defined(__levelSet_exteriorContent_ipp__)
#error This file is an implementation detail of exteriorContent.
#endif

namespace stlib
{
namespace levelSet
{


// This is used to compute the content from the signed distance.
// Return the function
//        1,             if x <= -1,
// f(x) = 0.5 - 0.5 * x, if -1 < x < 1,
//        0,             if x >= 1.
template<typename _T>
inline
_T
contentFractionFromDistance(const _T x)
{
  if (x <= -1) {
    return 1;
  }
  if (x >= 1) {
    return 0;
  }
  return 0.5 - 0.5 * x;
}


template<typename _T, std::size_t _D, std::size_t N>
inline
_T
exteriorContent
(const GridGeometry<_D, N, _T>& grid,
 const std::vector<bool>& isActive,
 const std::vector<geom::Ball<_T, _D> >& balls,
 const container::StaticArrayOfArrays<unsigned>& dependencies)
{
  typedef GridGeometry<_D, N, _T> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::Point Point;
  typedef geom::Ball<_T, _D> Ball;
  typedef container::SimpleMultiIndexExtentsIterator<_D> Iterator;

  // A little bit more than the diagonal length of a voxel.
  const _T diagonal = 1.01 * std::sqrt(double(_D)) * grid.spacing;
  const IndexList PatchExtents = ext::filled_array<IndexList>(N);

  // CONTINUE REMOVE
  assert(dependencies.getNumberOfArrays() ==
         std::count(isActive.begin(), isActive.end(), true));

  // Constants used for computing content.
  // CONTINUE:
  static_assert(_D == 3, "Only 3D supported.");
  // The content of a voxel.
  const _T voxelContent = grid.spacing * grid.spacing * grid.spacing;
  // The radius of the ball that has the same content as a voxel.
  // 4 * pi r^3 / 3 = dx^3
  const _T inverseVoxelRadius =
    1. / (std::pow(3 / (4 * numerical::Constants<_T>::Pi()), _T(1. / 3)) *
          grid.spacing);

  std::vector<Ball> influencingBalls;
  const Iterator jBegin = Iterator::begin(PatchExtents);
  const Iterator jEnd = Iterator::end(PatchExtents);
  Point x;
  _T f, d;
  _T content = 0;
  std::size_t activeIndex = 0;
  // Loop over the patches.
  const Iterator iEnd = Iterator::end(grid.gridExtents);
  for (Iterator i = Iterator::begin(grid.gridExtents); i != iEnd; ++i) {
    const std::size_t index = grid.arrayIndex(*i);
    // Only compute the content on active patches.
    if (! isActive[index]) {
      continue;
    }
    // Get the influencing balls for this patch.
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(activeIndex); ++n) {
      influencingBalls.push_back(balls[dependencies(activeIndex, n)]);
    }
    ++activeIndex;
    // Loop over the grid points in the patch.
    for (Iterator j = jBegin; j != jEnd; ++j) {
      //
      // Compute the distance to the union of balls.
      //
      x = grid.indexToLocation(*i, *j);
      f = std::numeric_limits<_T>::infinity();
      // Loop over the influencing balls.
      for (std::size_t n = 0; n != influencingBalls.size(); ++n) {
        // Compute the distance to the ball.
        d = distance(influencingBalls[n], x);
        // Update the minimum distance for the union of the balls.
        if (d < f) {
          f = d;
        }
#if 0
        // Early exit for points far inside the union of balls.
        if (f < -diagonal) {
          break;
        }
#else
        // Early exit for points inside the union of balls.
        if (f <= 0) {
          break;
        }
#endif
      }

      // Negate to get the exterior.
      f = - f;

#if 0
      // Compute the content from the distance.
      content += voxelContent *
                 contentFractionFromDistance(f * inverseVoxelRadius);
#else
      if (f < 0) {
        content += voxelContent;
      }
#endif
    }
  }
  assert(activeIndex == dependencies.getNumberOfArrays());

  return content;
}


} // namespace levelSet
}
