// -*- C++ -*-

#if !defined(__levelSet_geometry_h__)
#define __levelSet_geometry_h__

#include "stlib/geom/kernel/Ball.h"

namespace stlib
{
namespace levelSet
{



//! Return true if the point is inside one of the balls.
template<typename _T, std::size_t _D>
bool
isInside(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::array<_T, _D>& x);


//! Return true if the point is inside one of the active balls.
template<typename _T, std::size_t _D>
bool
isInside(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::vector<std::size_t>& active,
         const std::array<_T, _D>& x);


//! Get the set of balls that intersect the n_th ball.
template<typename _T, std::size_t _D>
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                std::size_t n, std::vector<std::size_t>* intersecting);


//! Get the set of balls that intersect one of the two specified balls.
template<typename _T, std::size_t _D>
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                std::size_t m, std::size_t n,
                std::vector<std::size_t>* intersecting);


//! Get the set of balls that intersect the specified ball.
template<typename _T, std::size_t _D>
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                const geom::Ball<_T, _D>& b,
                std::vector<std::size_t>* intersecting);


//! Return true if the intersection point is on the surface.
template<typename _T>
bool
isOnSurface(const std::vector<geom::Ball<_T, 2> >& balls,
            const std::size_t index1, const std::size_t index2,
            const std::array<_T, 2>& x);


//! Return true if the intersection point is on the surface.
template<typename _T>
bool
isOnSurface(const std::vector<geom::Ball<_T, 3> >& balls,
            std::size_t index1, std::size_t index2, std::size_t index3,
            const std::array<_T, 3>& x);


} // namespace levelSet
}

#define __levelSet_geometry_ipp__
#include "stlib/levelSet/geometry.ipp"
#undef __levelSet_geometry_ipp__

#endif
