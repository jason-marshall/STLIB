// -*- C++ -*-

/*!
  \file
  \brief Distance calculations for axis-oriented bounding boxes.
*/

#if !defined(__geom_BBoxDistance_h__)
#define __geom_BBoxDistance_h__

#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace geom
{

/*
[seanm@xenon kernel]$ ./BBoxDistanceScalarFloat 
Total distance = 7.10329e+06
minMinDist2 = 10.2911 nanoseconds.
maxMaxDist2 = 4.002 nanoseconds.
maxMinDist = 18.6698 nanoseconds.
nxnDist2 = 25.8313 nanoseconds.
upperBoundDist2 = 12.8203 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceScalarFloat 
Total distance = 7.10329e+06
minMinDist2 = 10.4645 nanoseconds.
maxMaxDist2 = 3.99733 nanoseconds.
maxMinDist = 18.6688 nanoseconds.
nxnDist2 = 25.0239 nanoseconds.
upperBoundDist2 = 10.9259 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceScalarFloat 
Total distance = 7.10329e+06
minMinDist2 = 10.564 nanoseconds.
maxMaxDist2 = 3.99747 nanoseconds.
maxMinDist = 18.6646 nanoseconds.
nxnDist2 = 25.8192 nanoseconds.
upperBoundDist2 = 12.8273 nanoseconds.

[seanm@xenon kernel]$ ./BBoxDistanceScalarDouble 
Total distance = 7.09589e+06
minMinDist2 = 10.3792 nanoseconds.
maxMaxDist2 = 3.99966 nanoseconds.
maxMinDist = 18.6567 nanoseconds.
nxnDist2 = 22.3583 nanoseconds.
upperBoundDist2 = 10.9472 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceScalarDouble 
Total distance = 7.09589e+06
minMinDist2 = 10.4263 nanoseconds.
maxMaxDist2 = 3.99866 nanoseconds.
maxMinDist = 18.6305 nanoseconds.
nxnDist2 = 22.3366 nanoseconds.
upperBoundDist2 = 11.0169 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceScalarDouble 
Total distance = 7.09589e+06
minMinDist2 = 10.4324 nanoseconds.
maxMaxDist2 = 4.00346 nanoseconds.
maxMinDist = 17.8419 nanoseconds.
nxnDist2 = 22.2904 nanoseconds.
upperBoundDist2 = 10.9447 nanoseconds.

[seanm@xenon kernel]$ ./BBoxDistanceSimdFloat 
Meaningless result = 1770.72
lowerBound2Time = 0.787887 nanoseconds.
upperBound2Time = 1.68363 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceSimdFloat 
Meaningless result = 1770.72
lowerBound2Time = 1.37718 nanoseconds.
upperBound2Time = 1.69017 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceSimdFloat 
Meaningless result = 1770.72
lowerBound2Time = 0.786422 nanoseconds.
upperBound2Time = 1.68623 nanoseconds.

[seanm@xenon kernel]$ ./BBoxDistanceSimdDouble 
Meaningless result = 1770.72
lowerBound2Time = 1.56572 nanoseconds.
upperBound2Time = 3.50575 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceSimdDouble 
Meaningless result = 1770.72
lowerBound2Time = 1.58195 nanoseconds.
upperBound2Time = 3.51145 nanoseconds.
[seanm@xenon kernel]$ ./BBoxDistanceSimdDouble 
Meaningless result = 1770.72
lowerBound2Time = 1.57441 nanoseconds.
upperBound2Time = 3.51141 nanoseconds.
 */

//! Return the minimum squared distance between the point and the bounding box.
/*! This is a lower bound on the squared distance between the point and the
  object bounded by the box. */
template<typename _Float, std::size_t _D>
_Float
minDist2(const BBox<_Float, _D>& a, const std::array<_Float, _D>& p);


//! Return the minimum squared distance between the bounding boxes.
/*! This is a lower bound on the distance between any point in the first
  box and the object bounded by the second box. */
template<typename _Float, std::size_t _D>
_Float
minMinDist2(const BBox<_Float, _D>& a, const BBox<_Float, _D>& b);


//! Return an upper bound on the distance from points in the first box to the object bounded by the second.
/*! Use the fact that an object touches all faces of its tight bounding box. */
template<typename _Float, std::size_t _D>
_Float
nxnDist2(const BBox<_Float, _D>& a, const BBox<_Float, _D>& b);


//! Return the minimum of the maximum distance between points in the bounding box and the specified points.
/*! Given that the supplied points lie on an object. This provides an upper
  bound on the distance between points in the box and the object. */
template<typename _Float, std::size_t _D, std::size_t N>
_Float
upperBoundDist2(const BBox<_Float, _D>& a,
                const std::array<std::array<_Float, _D>, N>& points);


} // namespace geom
}

#define __geom_BBoxDistance_ipp__
#include "stlib/geom/kernel/BBoxDistance.ipp"
#undef __geom_BBoxDistance_ipp__

#endif
