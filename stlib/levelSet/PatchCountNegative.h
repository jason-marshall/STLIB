// -*- C++ -*-

/*!
  \file levelSet/PatchCountNegative.h
  \brief Patch for counting points with negative distance.
*/

#if !defined(__levelSet_PatchCountNegative_h__)
#define __levelSet_PatchCountNegative_h__

#include "stlib/levelSet/PatchSquaredDistanceComponents.h"
#include "stlib/geom/kernel/Ball.h"

namespace stlib
{
namespace levelSet
{


//! Patch for counting points with negative distance.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8. */
class PatchCountNegative :
  public PatchSquaredDistanceComponents
{
  //
  // Types.
  //
private:
  typedef PatchSquaredDistanceComponents Base;

public:
  //! A ball.
  typedef geom::Ball<float, D> Ball;

  //
  // Member data.
  //
private:

  //! The number of active grid points (or active SIMD vectors for SIMD code).
  std::size_t _numActive;
#ifdef STLIB_NO_SIMD_INTRINSICS
  //! The active indices.
  /*! The grid points have indices from 0 to 511. Note: Using an array of
   indices would yield similar performance. */
  std::array<unsigned, NumVectors> _activeIndices;
#else
  //! The active indices.
  /*! For AVX2, the SIMD vectors have indices from 0 to 63. The extents are
    4 x 4 x 4. For SSE, the SIMD vectors have indices from 0 to 127. The
    extents are 4 x 4 x 8. Note that, either way, we can store the index
    type in an 8-bit unsigned char instead of the larger type that is
    needed for the scalar code. */
  std::array<unsigned char, NumVectors> _activeIndices;
  //! Bit masks that record which of the grid points in the SIMD vectors are active.
  std::array<unsigned char, NumVectors> _activeMasks;
#endif


  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchCountNegative(const float spacing) :
    Base(spacing),
    _numActive(0),
    _activeIndices()
#ifndef STLIB_NO_SIMD_INTRINSICS
    , _activeMasks()
#endif
  {
  }

  // @}
  //--------------------------------------------------------------------------
  /*! \name Mathematical functions. */
  // @{
public:

  //! Set the lower corner. Make all points active.
  void
  initialize(const Point& lowerCorner);

  //! Clip using the given ball.
  void
  clip(const Ball& ball);

  //! Return the number of grid points with negative distance.
  std::size_t
  numNegative() const;

  // @}
};


} // namespace levelSet
}

#define __levelSet_PatchCountNegative_ipp__
#include "stlib/levelSet/PatchCountNegative.ipp"
#undef __levelSet_PatchCountNegative_ipp__

#endif
