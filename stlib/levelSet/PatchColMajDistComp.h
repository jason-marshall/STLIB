// -*- C++ -*-

/*!
  \file levelSet/PatchColMajDistComp.h
  \brief Patch with column major ordering that is used in computing distance.
*/

#if !defined(__levelSet_PatchColMajDistComp_h__)
#define __levelSet_PatchColMajDistComp_h__

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

//! Patch with column major ordering that is used in computing distance.
/*! The class name stands for Patch with Column Major ordering, Distance
  Components. It is 3-D patch that uses single-precision floating points
  number for calculations. The patch extent in each dimension is 8.

  When computing the squared distance from a point to each grid point,
  it is wasteful to compute each distance independently. It is better
  to compute the squared distance for each coordinate, a total of
  3 * 8 = 24 terms. To compute the squared distance for a grid point
  one then adds the appropriate combination of coordinate distances.
*/
class PatchColMajDistComp :
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
#elif defined(__AVX2__)
  // The SIMD vector is 8x1x1.
  //! The squared distance in the x coordinate.
  Vector _dx;
  //! The squared distance in the y and z coordinates.
  std::array<Vector, Extent> _dy, _dz;
#else // __SSE__
  // The SIMD vector is 4x1x1.
  //! The squared distance in the x coordinate.
  std::array < Vector, Extent / 4 > _dx;
  //! The squared distance in the y and z coordinates.
  std::array<Vector, Extent> _dy, _dz;
#endif

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchColMajDistComp(const float spacing) :
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
    float t;
#ifdef STLIB_NO_SIMD_INTRINSICS
    for (std::size_t i = 0; i != Extent; ++i) {
      t = i * _spacing;
      _dx[i] = (t - p[0]) * (t - p[0]);
      _dy[i] = (t - p[1]) * (t - p[1]);
      _dz[i] = (t - p[2]) * (t - p[2]);
    }
#elif defined(__AVX2__)
    // Note in the following that the *set* intrinsics use little-endian
    // order, so read the arguments from right to left.
    _dx = _mm256_set_ps((7 * _spacing - p[0]) * (7 * _spacing - p[0]),
                        (6 * _spacing - p[0]) * (6 * _spacing - p[0]),
                        (5 * _spacing - p[0]) * (5 * _spacing - p[0]),
                        (4 * _spacing - p[0]) * (4 * _spacing - p[0]),
                        (3 * _spacing - p[0]) * (3 * _spacing - p[0]),
                        (2 * _spacing - p[0]) * (2 * _spacing - p[0]),
                        (_spacing - p[0]) * (_spacing - p[0]),
                        p[0] * p[0]);

    for (std::size_t i = 0; i != _dy.size(); ++i) {
      t = i * _spacing;
      _dy[i] = _mm256_set1_ps((t - p[1]) * (t - p[1]));
      _dz[i] = _mm256_set1_ps((t - p[2]) * (t - p[2]));
    }
#else // __SSE__
    _dx[0] = _mm_set_ps((3 * _spacing - p[0]) * (3 * _spacing - p[0]),
                        (2 * _spacing - p[0]) * (2 * _spacing - p[0]),
                        (_spacing - p[0]) * (_spacing - p[0]),
                        p[0] * p[0]);
    _dx[1] = _mm_set_ps((7 * _spacing - p[0]) * (7 * _spacing - p[0]),
                        (6 * _spacing - p[0]) * (6 * _spacing - p[0]),
                        (5 * _spacing - p[0]) * (5 * _spacing - p[0]),
                        (4 * _spacing - p[0]) * (4 * _spacing - p[0]));

    for (std::size_t i = 0; i != _dy.size(); ++i) {
      t = i * _spacing;
      _dy[i] = _mm_set1_ps((t - p[1]) * (t - p[1]));
      _dz[i] = _mm_set1_ps((t - p[2]) * (t - p[2]));
    }
#endif

  }

  // @}
};


} // namespace levelSet
}

#endif
