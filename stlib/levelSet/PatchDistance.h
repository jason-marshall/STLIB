// -*- C++ -*-

/*!
  \file levelSet/PatchDistance.h
  \brief Patch that stores the distance grid.
*/

#if !defined(__levelSet_PatchDistance_h__)
#define __levelSet_PatchDistance_h__

#include "stlib/levelSet/PatchSquaredDistanceComponents.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/simd/functions.h"

#if !defined(STLIB_NO_SIMD_INTRINSICS) && !defined(__SSE__) && !defined(__AVX2__)
#error SIMD is not supported.
#endif

namespace stlib
{
namespace levelSet
{


//! Patch that stores the distance grid.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8. */
class PatchDistance :
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

  //! The grid of distances.
  /*! We use a 1-D array to represent the 3-D array. For AVX2, the SIMD
    vectors are 2x2x2. For SSE, they are 2x2x1. */
  std::array<Vector, NumVectors> grid;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchDistance(const float spacing) :
    Base(spacing),
    grid()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  /*! \name Mathematical functions. */
  // @{
public:

  //! Set the lower corner. Initialize the grid values.
  void
  initialize(const Point& lowerCorner,
             float value = std::numeric_limits<float>::infinity());

  //! Add the ball with a union operation.
  void
  unionEuclidean(const Ball& ball);

  //! Access the specified SIMD vector.
  /*! Use 1-D indexing for the 3-D grid. */
  Vector
  operator[](std::size_t i) const
  {
    return grid[i];
  }

  //! For all grid points that are greater than or equal to the threshold, set to the specified value.
  void
  conditionalSetValueGe(float threshold, float value);

  // @}
};


} // namespace levelSet
}

#define __levelSet_PatchDistance_ipp__
#include "stlib/levelSet/PatchDistance.ipp"
#undef __levelSet_PatchDistance_ipp__

#endif
