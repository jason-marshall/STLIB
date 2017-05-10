// -*- C++ -*-

/*!
  \file DistancePeriodic.h
  \brief Distance in a periodic, axis-aligned, box domain.
*/

#if !defined(__geom_orq_DistancePeriodic_h__)
#define __geom_orq_DistancePeriodic_h__

#include "stlib/geom/kernel/BBox.h"

#include <limits>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace geom
{

//! Distance in a periodic, axis-aligned, box domain.
template<typename _T, std::size_t N>
class DistancePeriodic
{
  //
  // Types.
  //
public:

  //! A Cartesian point.
  typedef std::array<_T, N> Point;

  //
  // Data
  //
private:

  BBox<_T, N> _domain;
  Point _lengths;

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the domain.
  DistancePeriodic(const BBox<_T, N>& domain) :
    _domain(domain),
    _lengths(domain.upper - domain.lower)
  {
    // The box must have positive lengths.
    for (std::size_t i = 0; i != _lengths.size(); ++i) {
      assert(_lengths[i] > 0);
    }
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the domain.
  const BBox<_T, N>&
  domain() const
  {
    return _domain;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Functor.
  // @{
public:

  //! Return the Cartesian distance between the two points.
  /*!
    \pre The points must lie in the domain. This is checked with an assertion.
  */
  _T
  distance(const Point& a, const Point& b) const
  {
    return std::sqrt(squaredDistance(a, b));
  }

  //! Return the squared distance between the two points.
  /*!
    \pre The points must lie in the domain. This is checked with an assertion.
  */
  _T
  squaredDistance(const Point& a, const Point& b) const
  {
    // CONTINUE REMOVE
    if (! (isInside(_domain, a) && isInside(_domain, b))) {
      std::cerr << "_domain = " << _domain << '\n'
                << a << '\n'
                << b << '\n';
    }
    // The points must lie in the domain.
    assert(isInside(_domain, a) && isInside(_domain, b));
    return squaredDistance(a, b, std::integral_constant<std::size_t, N>());
  }

  //! Return the Cartesian distance between the two points.
  /*!
    If the points do not lie in the domain, they will be moved to the
    domain using the periodic property.
  */
  _T
  distanceCorrected(const Point& a, const Point& b) const
  {
    return distance(pointInDomain(a), pointInDomain(b));
  }

  //! Return the squared distance between the two points.
  /*!
    If the points do not lie in the domain, they will be moved to the
    domain using the periodic property.
  */
  _T
  squaredDistanceCorrected(const Point& a, const Point& b) const
  {
    return squaredDistance(pointInDomain(a), pointInDomain(b));
  }

  //! Use the periodic extension to return a point that lies in the domain.
  Point
  pointInDomain(Point a) const
  {
    for (std::size_t i = 0; i != a.size(); ++i) {
      a[i] -= std::floor((a[i] - _domain.lower[i]) / _lengths[i])
              * _lengths[i];
    }
    return a;
  }

protected:

  _T
  squaredDistance(const Point& a, const Point& b,
                  std::integral_constant<std::size_t, 1> /*dimension*/) const
  {
    const Point lower = b - _lengths;
    Point p;
    _T d2 = std::numeric_limits<_T>::infinity();
    _T t;
    std::size_t i;
    for (i = 0, p[0] = lower[0]; i != 3; ++i, p[0] += _lengths[0]) {
      t = ext::squaredDistance(a, p);
      if (t < d2) {
        d2 = t;
      }
    }
    return d2;
  }

  _T
  squaredDistance(const Point& a, const Point& b,
                  std::integral_constant<std::size_t, 2> /*dimension*/) const
  {
    const Point lower = b - _lengths;
    Point p;
    _T d2 = std::numeric_limits<_T>::infinity();
    _T t;
    std::size_t i, j;
    for (j = 0, p[1] = lower[1]; j != 3; ++j, p[1] += _lengths[1]) {
      for (i = 0, p[0] = lower[0]; i != 3; ++i, p[0] += _lengths[0]) {
        t = ext::squaredDistance(a, p);
        if (t < d2) {
          d2 = t;
        }
      }
    }
    return d2;
  }

  _T
  squaredDistance(const Point& a, const Point& b,
                  std::integral_constant<std::size_t, 3> /*dimension*/) const
  {
    const Point lower = b - _lengths;
    Point p;
    _T d2 = std::numeric_limits<_T>::infinity();
    _T t;
    std::size_t i, j, k;
    for (k = 0, p[2] = lower[2]; k != 3; ++k, p[2] += _lengths[2]) {
      for (j = 0, p[1] = lower[1]; j != 3; ++j, p[1] += _lengths[1]) {
        for (i = 0, p[0] = lower[0]; i != 3; ++i, p[0] += _lengths[0]) {
          t = ext::squaredDistance(a, p);
          if (t < d2) {
            d2 = t;
          }
        }
      }
    }
    return d2;
  }

  // @}
};

} // namespace geom
}

#endif
