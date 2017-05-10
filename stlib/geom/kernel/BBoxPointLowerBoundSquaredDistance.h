// -*- C++ -*-

/**
  \file
  \brief Calculate a lower bound on the squared distance from a point to a bounding box.
*/

#if !defined(__geom_BBoxDistanceSimd_h__)
#define __geom_BBoxDistanceSimd_h__

#include "stlib/geom/kernel/BBox.h"

//#include "stlib/simd/allocator.h"
#include "stlib/simd/array.h"
//#include "stlib/simd/constants.h"
//#include "stlib/simd/functions.h"

namespace stlib
{
namespace geom
{


// Calculate a lower bound on the squared distance from a point to a bounding box.
template<typename _Float, std::size_t _D>
class BBoxPointLowerBoundSquaredDistance
{
  //
  // Constants and types.
  //
public:

  /// A Cartesian point with aligned memory.
  typedef stlib::simd::array<_Float, _D> AlignedPoint;
  /// An axis-aligned bounding box.
  typedef stlib::geom::BBox<_Float, _D> BBox;

  //
  // Data
  //
private:

  /// Coordinates for the lower corner.
  AlignedPoint _lower;
  /// Coordinates for the upper corner.
  AlignedPoint _upper;

public:

  /// Default constructor.
  BBoxPointLowerBoundSquaredDistance() :
    _lower(),
    _upper()
  {
  }

  /// Construct from a bounding box.
  BBoxPointLowerBoundSquaredDistance(BBox const& box);

  /// Return a lower bound on the squared distance from the point to the bounding box.
  _Float
  operator()(AlignedPoint const& x) const;
};


} // namespace geom
} // namespace stlib

#define __geom_BBoxPointLowerBoundSquaredDistance_ipp__
#include "stlib/geom/kernel/BBoxPointLowerBoundSquaredDistance.ipp"
#undef __geom_BBoxPointLowerBoundSquaredDistance_ipp__

#endif
