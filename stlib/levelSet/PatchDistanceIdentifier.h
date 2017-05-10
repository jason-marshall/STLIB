// -*- C++ -*-

/*!
  \file levelSet/PatchDistanceIdentifier.h
  \brief Patch that stores distance and identifier grids.
*/

#if !defined(__levelSet_PatchDistanceIdentifier_h__)
#define __levelSet_PatchDistanceIdentifier_h__

#include "stlib/levelSet/PatchDistance.h"

#if !defined(STLIB_NO_SIMD_INTRINSICS) && !defined(__SSE__) && !defined(__AVX2__)
#error SIMD is not supported.
#endif

namespace stlib
{
namespace levelSet
{


//! Patch that stores distance and identifier grids.
/*! 3-D patch that uses single-precision floating points number for
  calculations. The patch extent in each dimension is 8. */
class PatchDistanceIdentifier :
  public PatchDistance
{
  //
  // Types.
  //
private:
  typedef PatchDistance Base;

  //
  // Member data.
  //
public:

  //! The grid of identifiers.
  /*! We use a 1-D array to represent the 3-D array. For AVX2, the SIMD
    vectors are 2x2x2. For SSE, they are 2x2x1.

    The value \c std::numeric_limits<unsigned>::max() to represent an
    invalid identifier.
  */
#ifdef STLIB_NO_SIMD_INTRINSICS
  std::array<unsigned, NumVectors> identifiers;
#elif defined(__AVX2__)
  std::array<__m256i, NumVectors> identifiers;
#else
  std::array<__m128i, NumVectors> identifiers;
#endif

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the synthesized copy constructor, assignment operator, and
    destructor.
  */
  // @{
public:

  //! Construct from the grid spacing.
  PatchDistanceIdentifier(const float spacing) :
    Base(spacing),
    identifiers()
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
  unionEuclidean(const Ball& ball, unsigned id);

  //! Access the specified SIMD vector.
  /*! Use 1-D indexing for the 3-D grid. */
  Vector
  operator[](std::size_t i) const
  {
    return grid[i];
  }

  //! For all grid points that are greater than or equal to the threshold, set to the specified value.
  void
  conditionalSetValueGe(float threshold, float value,
                        unsigned id = std::numeric_limits<unsigned>::max());

  // @}
};


} // namespace levelSet
}

#define __levelSet_PatchDistanceIdentifier_ipp__
#include "stlib/levelSet/PatchDistanceIdentifier.ipp"
#undef __levelSet_PatchDistanceIdentifier_ipp__

#endif
