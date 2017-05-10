// -*- C++ -*-

/**
  \file
  \brief Implements a class for a hyperplane.
*/

#if !defined(__geom_Hyperplane_h__)
#define __geom_Hyperplane_h__

#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/kernel/simplexTopology.h"

#include <boost/config.hpp>

#include <limits>

#include <cmath>

namespace stlib
{
namespace geom
{

/// A hyperplane in an (n-1)-dimensional flat in n-D space.
/**
  \param _T is the floating-point number type.
  \param _D is the space dimension.

  A hyperplane in n-D is defined by a point on the hyperplane and its
  unit normal. For a point, \f$\vec{p}\f$, and normal, \f$\hat{n}\f$,
  the equation of the hyperplane is \f$\hat{n} \cdot (\vec{x} - \vec{p}) = 0\f$.

  This class is a POD type. Thus, it has no user-defined
  constructors. The hyperplane is defined by the public data members \c
  point and \c normal. Note that in order to be valid, the normal
  must have unit magnitude. Since this class is POD, this is obviously
  not checked upon construction. When implementing functions that
  perform mathematical operations with this class, it is a good idea
  to check the validity of the hyperplane (at least in debugging mode) with
  isValid().

  To construct a hyperplane, you can use brace initialization with the 
  point on the hyperplane and the unit normal. Below are few examples.
  \code
  Hyperplane<float, 3> a = {{{0, 0, 0}}, {{1, 0, 0}}};
  Hyperplane<double, 3> b{{{1, 2, 3}}, {{0.5 * std::sqrt(2.), 0.5 * std::sqrt(2.), 0}}};
  std::array<double, 3> normal = {{1, 2, 3}};
  stlib::ext::normalize(&normal);
  Hyperplane<double, 3> c = {{{0, 0, 0}}, normal};
  \endcode

  The unary + and - operators are defined for hyperplanes. The former has no 
  effect, while the latter reverses the orientation, which is determined by the 
  normal direction. The arithmetic operators += and -= are also defined.
  Adding or subracting a point just moves the \c point data member.
  Finally, equality, inequality, and file I/O operators are defined 
  as well.

  There are two functions, signedDistance(), for computing the signed distance
  to the hyperplane. One of them computes the closest point on the hyperplane
  in addition to returning the distance.
*/
template<typename _T, std::size_t _D>
struct Hyperplane
{
  /// The floating-point number type.
  using Float = _T;
  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;
  /// The representation for a point.
  using Point = std::array<Float, Dimension>;

  /// A point on the hyperplane.
  Point point;
  /// The unit normal to the hyperplane.
  Point normal;
};


//
// Unary Operators
//


/// The positive operator. Return the same hyperplane.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
Hyperplane<_T, _D> const&
operator+(Hyperplane<_T, _D> const& x) noexcept
{
  return x;
}


/// The negative operator. Return the hyperplane with opposite orientation.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
Hyperplane<_T, _D>
operator-(Hyperplane<_T, _D> const& x) noexcept
{
  return Hyperplane<_T, _D>{x.point, - x.normal};
}

//
// Arithmetic operators.
//


/// Translate the hyperplane by +p.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
Hyperplane<_T, _D>&
operator+=(Hyperplane<_T, _D>& x, typename Hyperplane<_T, _D>::Point const& p)
  noexcept
{
  x.point += p;
  return x;
}


/// Translate the hyperplane by -p.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
Hyperplane<_T, _D>&
operator-=(Hyperplane<_T, _D>& x, typename Hyperplane<_T, _D>::Point const& p)
  noexcept
{
  x.point -= p;
  return x;
}


//
// Equality Operators
//


/// Return true if the hyperplanes are equivalent.
/**
   \relates Hyperplane
   Note that the \c point members may differ for equivalent planes.
*/
template<typename _T, std::size_t _D>
inline
bool
operator==(Hyperplane<_T, _D> const& x, Hyperplane<_T, _D> const& y) noexcept
{
  return x.normal == y.normal && ext::dot(x.point - y.point, x.normal) == 0;
}


/// Return true if the hyperplanes are not equal.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
bool
operator!=(Hyperplane<_T, _D> const& x, Hyperplane<_T, _D> const& y) noexcept
{
  return !(x == y);
}


//
// File I/O
//


/// Read the point and normal.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
std::istream&
operator>>(std::istream& in, Hyperplane<_T, _D>& x);


/// Write the point and normal.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
std::ostream&
operator<<(std::ostream& out, Hyperplane<_T, _D> const& x);


/// Return true if the hyperplane is valid.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
bool
isValid(Hyperplane<_T, _D> const& x) noexcept
{
  return x.point == x.point && std::abs(ext::squaredMagnitude(x.normal) - 1) <
    _D * std::numeric_limits<_T>::epsilon();
}


/// Return the distance from p to the hyperplane.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
inline
_T
signedDistance(Hyperplane<_T, _D> const& hyperplane,
               typename Hyperplane<_T, _D>::Point const& p)
{
#ifdef STLIB_DEBUG
  assert(isValid(hyperplane));
#endif
  return ext::dot(p - hyperplane.point, hyperplane.normal);
}


/// Return the distance from the point to the hyperplane. Set the closest point.
/** \relates Hyperplane */
template<typename _T, std::size_t _D>
_T
signedDistance(Hyperplane<_T, _D> const& hyperplane,
               typename Hyperplane<_T, _D>::Point const& p,
               typename Hyperplane<_T, _D>::Point* closestPoint);


/// Return the supporting hyperplane of the specified face of the simplex.
/** 
\relates Hyperplane
\param simplex A simplex in 1-D is an oriented interval.
\param The nth face is that opposite the nth vertex.

\note There is no function for building the supporting hyperplane in 1-D 
from the coordinates of the face vertices. This is because the coordinates
alone do not determine the orientation.
*/
template<typename _T>
Hyperplane<_T, 1>
supportingHyperplane(std::array<std::array<_T, 1>, 2> const& simplex,
                     std::size_t n);


/// Return the supporting hyperplane of the simplex face.
/** 
    \relates Hyperplane
    \param face A simplex face in 2-D is a line segment.
*/
template<typename _T>
Hyperplane<_T, 2>
supportingHyperplane(std::array<std::array<_T, 2>, 2> const& face);


/// Return the supporting hyperplane of the specified face of the simplex.
/** 
    \relates Hyperplane
    \param simplex A simplex in 2-D is an oriented triangle.
    \param The nth face is that opposite the nth vertex.
*/
template<typename _T>
Hyperplane<_T, 2>
supportingHyperplane(std::array<std::array<_T, 2>, 3> const& simplex,
                     std::size_t n);


/// Return the supporting hyperplane of the simplex face.
/** 
    \relates Hyperplane
    \param face A simplex face in 3-D is a triangle segment.
*/
template<typename _T>
Hyperplane<_T, 3>
supportingHyperplane(std::array<std::array<_T, 3>, 3> const& face);


/// Return the supporting hyperplane of the specified face of the simplex.
/** 
    \relates Hyperplane
    \param simplex A simplex in 3-D is an oriented tetrahedron.
    \param The nth face is that opposite the nth vertex.
*/
template<typename _T>
Hyperplane<_T, 3>
supportingHyperplane(std::array<std::array<_T, 3>, 4> const& simplex,
                     std::size_t n);


} // namespace geom
} // namespace stlib

#define __geom_Hyperplane_ipp__
#include "stlib/geom/kernel/Hyperplane.ipp"
#undef __geom_Hyperplane_ipp__

#endif
