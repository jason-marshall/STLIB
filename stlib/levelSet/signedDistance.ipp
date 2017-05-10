// -*- C++ -*-

#if !defined(__levelSet_signedDistance_ipp__)
#error This file is an implementation detail of signedDistance.
#endif

namespace stlib
{
namespace levelSet
{


template<typename _T, std::size_t _D>
inline
void
signedDistance(GridUniform<_T, _D>* grid,
               const std::vector<geom::Ball<_T, _D> >& balls,
               const _T maxDistance)
{
  // First compute the negative distance.
  negativeDistance(grid, balls);
  // Add the positive distance.
  positiveDistance(grid, balls, _T(0), maxDistance);
  // Set the positive distance for far away points.
  for (std::size_t i = 0; i != grid->size(); ++i) {
    if ((*grid)[i] != (*grid)[i]) {
      (*grid)[i] = std::numeric_limits<_T>::max();
    }
  }
}


} // namespace levelSet
}
