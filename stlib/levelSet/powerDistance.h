// -*- C++ -*-

#if !defined(__levelSet_powerDistance_h__)
#define __levelSet_powerDistance_h__

#include "stlib/levelSet/Grid.h"

#include "stlib/ads/algorithm/sort.h"

#include "stlib/geom/kernel/Ball.h"
#include "stlib/geom/kernel/BallSquared.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetPowerDistance Power Distance
  These functions calculate the power distance to a set of balls.
*/
//@{

//! Construct a level set for the power distance to a set of balls.
/*! For each grid point, compute the distance to each of the balls. */
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
void
powerDistance(container::EquilateralArrayImp<_T, _D, N, _Base>* patch,
              const std::array<_T, _D>& lowerCorner,
              _T spacing, const std::vector<geom::BallSquared<_T, _D> >& balls);


//! Construct a level set for the power distance to a set of balls.
template<typename _T, std::size_t _D, std::size_t N>
void
negativePowerDistance(Grid<_T, _D, N>* grid,
                      const std::vector<geom::Ball<_T, _D> >& balls);

//@}

} // namespace levelSet
}

#define __levelSet_powerDistance_ipp__
#include "stlib/levelSet/powerDistance.ipp"
#undef __levelSet_powerDistance_ipp__

#endif
