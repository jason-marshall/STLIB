// -*- C++ -*-

/*!
  \file levelSet/PatchGeometry.h
  \brief Describe the geometry of a patch.
*/

#if !defined(__levelSet_PatchGeometry_h__)
#define __levelSet_PatchGeometry_h__

#include "stlib/ext/array.h"
#include "stlib/simd/constants.h"

namespace stlib
{
namespace levelSet
{


//! Describe the geometry of a patch.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8.

  For the AVX2 version, the grid points are clustered into groups of eight.
  The SIMD vector forms a 2 x 2 x 2 voxel.
  For the SSE version, the grid points are clustered into groups of four.
  The shape of the SIMD vectors is 2 x 2 x 1.
*/
class PatchGeometry
{
  //
  // Constants.
  //
public:
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t D = 3;
  //! The patch extent in each dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Extent = 8;
  //! The number of grid points.
  BOOST_STATIC_CONSTEXPR std::size_t NumPoints = Extent* Extent* Extent;
  //! The size of the SIMD vector.
#ifdef STLIB_NO_SIMD_INTRINSICS
  BOOST_STATIC_CONSTEXPR std::size_t VectorSize = 1;
#elif defined(__AVX2__)
  BOOST_STATIC_CONSTEXPR std::size_t VectorSize = 8;
#else
  BOOST_STATIC_CONSTEXPR std::size_t VectorSize = 4;
#endif
  //! The number of SIMD vectors.
  BOOST_STATIC_CONSTEXPR std::size_t NumVectors = NumPoints / VectorSize;

  //
  // Types.
  //
public:
  //! A Cartesian point.
  typedef std::array<float, D> Point;
  //! The SIMD vector type.
#ifdef STLIB_NO_SIMD_INTRINSICS
  typedef float Vector;
#elif defined(__AVX2__)
  typedef __m256 Vector;
#else
  typedef __m128 Vector;
#endif

  //
  // Member data.
  //
protected:

  //! The grid spacing.
  float _spacing;
  //! The location of the lower corner of the patch.
  Point _lowerCorner;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchGeometry(const float spacing) :
    // Set the grid spacing.
    _spacing(spacing),
    // Set the lower corner to an invalid value. It must then be set
    // before the patch is used.
    _lowerCorner(ext::filled_array<Point>(std::numeric_limits<float>::
                                          quiet_NaN()))
  {
  }

  // @}
};


} // namespace levelSet
}

#endif
