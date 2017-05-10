// -*- C++ -*-

#if !defined(__levelSet_subtract_h__)
#define __levelSet_subtract_h__

#include "stlib/levelSet/Grid.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetSubtract Subtract Balls from a Manifold
*/
//@{

//! Subtract the balls from the level set.
/*! Compute the distance up to \c maxDistance past the surface of the balls. */
template<typename _T, std::size_t _D>
void
subtract(container::SimpleMultiArrayRef<_T, _D>* grid,
         const geom::BBox<_T, _D>& domain,
         const std::vector<geom::Ball<_T, _D> >& balls, _T maxDistance);


//! Subtract the balls from the level set.
template<typename _T, std::size_t _D, std::size_t N, typename _Base>
void
subtract(container::EquilateralArrayImp<_T, _D, N, _Base>* patch,
         const std::array<_T, _D>& lowerCorner, _T spacing,
         const std::vector<geom::Ball<_T, _D> >& balls);


//! Subtract the balls from the level set.
/*! Compute the distance up to \c maxDistance past the surface of the balls. */
template<typename _T, std::size_t _D, std::size_t N>
void
subtract(Grid<_T, _D, N>* grid, const std::vector<geom::Ball<_T, _D> >& balls,
         _T maxDistance);


//@}

} // namespace levelSet
}

#define __levelSet_subtract_ipp__
#include "stlib/levelSet/subtract.ipp"
#undef __levelSet_subtract_ipp__

#endif
