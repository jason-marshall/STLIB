// -*- C++ -*-

#if !defined(__levelSet_powerDistance_ipp__)
#error This file is an implementation detail of powerDistance.
#endif

namespace stlib
{
namespace levelSet
{


// Construct a level set for the power distance to a set of balls.
// For each grid point, compute the distance to each of the balls.
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
inline
void
powerDistance(container::EquilateralArrayImp<_T, _D, N, _Base>* patch,
              const std::array<_T, _D>& lowerCorner,
              const _T spacing,
              const std::vector<geom::BallSquared<_T, _D> >& balls)
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  std::array<_T, _D> p;
  _T d;
  // For each grid point.
  const Iterator end = Iterator::end(patch->extents());
  for (Iterator i = Iterator::begin(patch->extents()); i != end; ++i) {
    _T& g = (*patch)(*i);
    g = std::numeric_limits<_T>::infinity();
    p = lowerCorner + spacing * stlib::ext::convert_array<_T>(*i);
    // Find the minimum power distance over the set of balls.
    for (std::size_t j = 0; j != balls.size(); ++j) {
      d = ext::squaredDistance(p, balls[j].center) - balls[j].squaredRadius;
      if (d < g) {
        g = d;
      }
    }
  }
}


// Construct a level set for the power distance to a set of balls.
// For each grid point, compute the distance to each of the balls.
// We pass a pointer to the balls because we will sort them.
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
inline
void
powerDistance(container::EquilateralArrayImp<_T, _D, N, _Base>* patch,
              const std::array<_T, _D>& lowerCorner,
              const _T spacing,
              std::vector<geom::BallSquared<_T, _D> >* balls,
              const _T threshold)
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  assert(threshold > 0);

  // Sort the balls by the squared distance from the center of the patch.
  {
    std::array<_T, _D> patchCenter = lowerCorner +
                                     _T(0.5 * spacing * (N - 1));
    std::vector<_T> distances(balls->size());
    for (std::size_t i = 0; i != distances.size(); ++i) {
      distances[i] = ext::squaredDistance(patchCenter, (*balls)[i].center);
    }
    ads::sortTogether(distances.begin(), distances.end(), balls->begin(),
                      balls->end());
  }

  std::array<_T, _D> p;
  _T d;
  // For each grid point.
  const Iterator end = Iterator::end(patch->extents());
  for (Iterator i = Iterator::begin(patch->extents()); i != end; ++i) {
    _T& g = (*patch)(*i);
    g = std::numeric_limits<_T>::infinity();
    p = lowerCorner + spacing * stlib::ext::convert_array<_T>(*i);
    // Find the minimum power distance over the set of balls.
    for (std::size_t j = 0; j != balls->size(); ++j) {
      d = ext::squaredDistance(p, (*balls)[j].center) -
        (*balls)[j].squaredRadius;
      if (d < g) {
        g = d;
      }
      // If the distance is negative and more than the threshold from
      // the surface, we can stop computing distances for this point.
      if (g < -threshold) {
        break;
      }
    }
  }
}


// Construct a level set for the power distance to a set of balls.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
negativePowerDistance(Grid<_T, _D, N>* grid,
                      const std::vector<geom::Ball<_T, _D> >& balls)
{
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Dispense with the trivial case.
  if (grid->empty()) {
    return;
  }

  // Determine the patch/ball dependencies.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // 1.1 * (diagonal length of a voxel)
    const _T offset = 1.1 * grid->spacing * std::sqrt(_T(_D));
    // Offset the ball radii to include the volume of calculated distance.
    std::vector<geom::Ball<_T, _D> > offsetBalls(balls);
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += offset;
    }
    // Calculate the dependencies.
    patchDependencies(*grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

  // Refine the appropriate patches and set the rest to have an unknown
  // distance.
  grid->refine(dependencies);

  std::vector<geom::BallSquared<_T, _D> > influencingBalls;
  geom::BallSquared<_T, _D> bs;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid->extents());
  for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
    const std::size_t index = grid->arrayIndex(*i);
    if (!(*grid)[index].isRefined()) {
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
    // Compute the power distance.
    powerDistance(&(*grid)[index], grid->getPatchLowerCorner(*i),
                  grid->spacing, influencingBalls);
  }
}


} // namespace levelSet
}
