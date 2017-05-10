// -*- C++ -*-

/*!
  \file levelSet/PatchSquaredDistanceComponents.h
  \brief Patch that is used in computing the squared distance.
*/

#if !defined(__levelSet_PatchSquaredDistanceComponents_h__)
#define __levelSet_PatchSquaredDistanceComponents_h__

#include "stlib/levelSet/PatchGeometry.h"

#ifndef STLIB_NO_SIMD_INTRINSICS
#ifdef __AVX2__
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#else
#error SIMD is not supported.
#endif
#endif

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Patch that is used in computing the squared distance.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8.

  When computing the squared distance from a point to each grid point,
  it is wasteful to compute each distance independently. It is better
  to compute the squared distance for each coordinate, a total of
  3 * 8 = 24 terms. To compute the squared distance for a grid point
  one then adds the appropriate combination of coordinate distances.

  See PatchGeometry for the grid point clustering used in the SIMD versions.
*/
class PatchSquaredDistanceComponents :
  public PatchGeometry
{
  //
  // Types.
  //
private:
  typedef PatchGeometry Base;

  //
  // Member data.
  //
protected:

#ifdef STLIB_NO_SIMD_INTRINSICS

  //! The squared distance in each coordinate.
  std::array<Vector, Extent> _dx, _dy, _dz;

#else

#ifdef __AVX2__
  // The SIMD vector is 2x2x2.
  //! The squared distance in each coordinate.
  std::array < Vector, (Extent / 2) > _dx, _dy, _dz;
#elif defined(__SSE__)
  // The SIMD vector is 2x2x1.
  //! The squared distance in the x, and y coordinates.
  std::array < Vector, (Extent / 2) > _dx, _dy;
  //! The squared distance in the z coordinate.
  std::array<Vector, Extent> _dz;
#else
#error SIMD is not supported.
#endif

#endif

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchSquaredDistanceComponents(const float spacing) :
    Base(spacing),
    _dx(),
    _dy(),
    _dz()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  /*! \name Mathematical functions. */
  // @{
public:

  //! Compute the distance components.
  void
  computeDistanceComponents(Point p)
  {
    // First transform to the offset from the lower corner.
    p -= _lowerCorner;

    // Compute the squared distances in the x, y, and z coordinates.
#ifdef STLIB_NO_SIMD_INTRINSICS

    for (std::size_t i = 0; i != Extent; ++i) {
      const float t = i * _spacing;
      _dx[i] = (t - p[0]) * (t - p[0]);
      _dy[i] = (t - p[1]) * (t - p[1]);
      _dz[i] = (t - p[2]) * (t - p[2]);
    }

#else

    // Note in the following that the *set* intrinsics use little-endian
    // order, so read the arguments from right to left.
    float t, d0, d1;
#ifdef __AVX2__
    for (std::size_t i = 0; i != _dx.size(); ++i) {
      t = 2 * i * _spacing;
      d0 = (t - p[0]) * (t - p[0]);
      d1 = (t - p[0] + _spacing) * (t - p[0] + _spacing);
      _dx[i] = _mm256_set_ps(d1, d0, d1, d0, d1, d0, d1, d0);
      d0 = (t - p[1]) * (t - p[1]);
      d1 = (t - p[1] + _spacing) * (t - p[1] + _spacing);
      _dy[i] = _mm256_set_ps(d1, d1, d0, d0, d1, d1, d0, d0);
      d0 = (t - p[2]) * (t - p[2]);
      d1 = (t - p[2] + _spacing) * (t - p[2] + _spacing);
      _dz[i] = _mm256_set_ps(d1, d1, d1, d1, d0, d0, d0, d0);
    }
#elif defined(__SSE__)
    for (std::size_t i = 0; i != _dx.size(); ++i) {
      t = 2 * i * _spacing;
      d0 = (t - p[0]) * (t - p[0]);
      d1 = (t - p[0] + _spacing) * (t - p[0] + _spacing);
      _dx[i] = _mm_set_ps(d1, d0, d1, d0);
      d0 = (t - p[1]) * (t - p[1]);
      d1 = (t - p[1] + _spacing) * (t - p[1] + _spacing);
      _dy[i] = _mm_set_ps(d1, d1, d0, d0);
    }
    for (std::size_t i = 0; i != _dz.size(); ++i) {
      t = i * _spacing;
      _dz[i] = _mm_set1_ps((t - p[2]) * (t - p[2]));
    }
#else
#error SIMD is not supported.
#endif

#endif

  }

  // @}
};


} // namespace levelSet
}

#endif
