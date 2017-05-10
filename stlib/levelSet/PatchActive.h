// -*- C++ -*-

/*!
  \file levelSet/PatchActive.h
  \brief Patch that keeps track of active grid points.
*/

#if !defined(__levelSet_PatchActive_h__)
#define __levelSet_PatchActive_h__

#include "stlib/levelSet/PatchSquaredDistanceComponents.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/numerical/integer/bits.h"
#include "stlib/simd/operators.h"

#include <cstring>

namespace stlib
{
namespace levelSet
{


//! Patch that keeps track of active grid points.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8. */
class PatchActive :
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
public:

  //! The number of active grid points (or active SIMD vectors for SIMD code).
  std::size_t numActive;
#ifdef STLIB_NO_SIMD_INTRINSICS
  //! The active indices.
  /*! The grid points have indices from 0 to 511. Note: Using an array of
   indices would yield similar performance. */
  std::array<unsigned, NumVectors> activeIndices;
#else
  //! The active indices.
  /*! For AVX2, the SIMD vectors have indices from 0 to 63. The extents are
    4 x 4 x 4. For SSE, the SIMD vectors have indices from 0 to 127. The
    extents are 4 x 4 x 8. Note that, either way, we can store the index
    type in an 8-bit unsigned char instead of the larger type that is
    needed for the scalar code. */
  std::array<unsigned char, NumVectors> activeIndices;
#endif
  //! Bit masks that record which of the grid points in the SIMD vectors are active.
  std::array<unsigned char, NumVectors> activeMasks;


  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchActive(const float spacing) :
    Base(spacing),
    numActive(0),
    activeIndices(),
    activeMasks()
  {
#ifdef STLIB_NO_SIMD_INTRINSICS
    // For the scalar version, the active masks are constant.
    std::fill(activeMasks.begin(), activeMasks.end(), 1);
#endif
  }

  // @}
  //--------------------------------------------------------------------------
  /*! \name Mathematical functions. */
  // @{
public:

  //! Set the lower corner. Make all points active or inactive.
  void
  initialize(const Point& lowerCorner, bool areActive = true);

  //! Set the lower corner. Set the positive grid points to be active.
  void
  initializePositive(const Point& lowerCorner,
                     const std::array<Vector, NumVectors>& grid);

  //! Clip using the given ball.
  void
  clip(const Ball& ball);

  //! Return the number of active grid points.
  std::size_t
  numActivePoints();

  //! Return true if there are any active grid points.
  /*! Note that this may be more efficient than checking if
    numActivePoints() == 0. */
  bool
  hasActive()
  {
    return numActive != 0;
  }

private:

  //! Move the inactive points out of the active region.
  void
  moveInactive();

  // @}
};


} // namespace levelSet
}

#define __levelSet_PatchActive_ipp__
#include "stlib/levelSet/PatchActive.ipp"
#undef __levelSet_PatchActive_ipp__

#endif
