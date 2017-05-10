// -*- C++ -*-

/*!
  \file
  \brief Includes the classes in the kernel sub-package.
*/

namespace stlib {
namespace geom {
/*!
  \page geom_kernel Kernel Package

  \par
  The kernel sub-package has the following functionality:
  - Point.h has \ref point "functions" for using std:array's as points
  and vectors.
  - content.h defines \ref content "functions" to compute content
  (length, area, volume, etc.).
  - Ball is a ball in N-D.
  - BBox is an axis-aligned bounding box in N-D, which is a closed interval.
  - Circle3 is a circle in 3-D.
  - CircularArc3 is a circular arc in 3-D.
  - Hyperplane is an (n-1)-D flat in n-D space.
  - Line_2 is a line in 2-D.
  - orientation.h defines \ref geom_kernel_orientation "orientation" functions.
  - ParametrizedLine is a parametrized line in N-D.
  - ParametrizedPlane is a parametrized plane in N-D.
  - SegmentMath is a line segment in N-D that supports some
    mathematical operations.
  - There are functions and classes for calculating
  \ref geom_kernel_bboxDist "lower and upper bounds on distance"
  with bounding boxes and extreme points.
  - \ref simplex_topology

  Use this sub-package by including the file kernel.h.
*/

/*!
  \page geom_kernel_bboxDist Bounding Box Distances

  \par Metrics using tight bounding boxes.
  A number of metrics may be used to bound the distance between two groups
  of objects. We use the squared distance, which is more efficient to compute.
  (We will omit the "squared" qualification in the rest of this section.)
  The most common metrics use tight bounding boxes.
  The <i>min-min distance</i> is the minimum distance between any points in the
  two bounding boxes. This is a lower bound on the distance between any two 
  objects in the bounding boxes. The max-max distance is the maximum distance 
  beteen any two points in the bounding boxes. This is a loose upper bound 
  on the distance between any two objects. A tighter upper bound is provided
  by the NXN distance. NXN stands for min-max-min and was introduced in
  "Efficient Evaluation of All-Nearest-Neighbor Queries" by Chen and Patel.
  It uses the fact that for a tight bounding box for a group of objects, an 
  object touches every face.

  \par Metrics using extreme points.
  We can get an even tighter upper bound on the
  distance by using extreme points instead of a bounding box for the second 
  group of objects. (Extreme points are implemented in the geom::ExtremePoints
  class.) We store the points on the objects with minimum
  and maximum coordinates in each dimension. 
  Thus, we store <i>2N</i> points for objects in N-D space. Extreme points
  may be manipulated in much the same way as for tight bounding boxes; 
  merging and adding points is easy. Of course, one may calculate a bounding
  box from the extreme points representation.

  \par
  When bounding the distance using a tight bounding
  box for the source objects and extreme points for the target objects, we 
  typically use only half of the extreme points. By comparing the centroids 
  of the two bounding boxes, we determine directions in each dimension.
  We use the directions to pick the point that is likely closer in each 
  dimension. We do not measure the cost of computing the directions. In 
  applications using trees, this cost is usually amortized over a number of
  object groups anyway.

  \par Performance.
  In the table below we show the time per distance calculation in nanoseconds.
  We present results for both single precision (\c float) and
  double precision. The tests were run on an Intel(R) Xeon(R) CPU
  E5-2687W v2 @ 3.4 GHz. The executables were compiled with GCC 4.4.7
  with the AVX compiler flag enabled.

  \par
  <table>
  <tr>
  <th>
  <th> float
  <th> double
  <tr>
  <th> Min-min distance
  <td> 10.3 ns
  <td> 10.4 ns
  <tr>
  <th> Max-max distance
  <td> 4.0 ns
  <td> 4.0 ns
  <tr>
  <th> NXN distance
  <td> 25.0 ns
  <td> 22.3 ns
  <tr>
  <th> Extreme points upper bound
  <td> 10.9 ns
  <td> 10.9 ns
  </table>

  \par
  First note that there is little difference in performance between using
  single and double precision. This is expected as the algorithms don't utilize
  SIMD operations. Getting a lower bound on the distance using the 
  min-min metric takes about 10 nanoseconds. Getting an upper bound with 
  the max-max distance is cheap, but the bound is loose. The NXN distance
  is significantly more expensive to calculate, but the tighter bound 
  would more than offset the cost in most applications. Using extreme 
  points to bound the distance is about half as expensive as the NXN bound.
  As noted before, the extreme points bound is also tighter, however, 
  the extreme points require more storage than a bounding box.

  \par SIMD.
  The geom::BBoxDistance class calculates lower bounds with the min-min 
  distance and upper bounds using extreme points. It utilizes SIMD intrinsics to
  accelerate the computations. In the table below we provide timing results
  for both single and double precision. First consider the former.
  Using SIMD results in much faster calculations. Calculating the lower bound
  is more than 12 times faster, while calculating the upper bound is about 
  6 times faster than the scalar code. Using double-precision
  floating-point numbers roughly doubles the cost.

  \par
  <table>
  <tr>
  <th>
  <th> float
  <th> double
  <tr>
  <th> Lower bound
  <td> 0.8 ns
  <td> 1.6 ns
  <tr>
  <th> Upper bound
  <td> 1.7 ns
  <td> 3.5 ns
  </table>

  \par Conclusions.
  For getting a lower bound on the distance, one typically uses the min-min 
  distance. For the upper bound, there are a number of choices. The max-max
  distance is inexpensive to calculate, but the bound is loose. Recent work
  has indicated that the NXN distance, although more expensive to 
  calculate, results in faster search methods due to its 
  tighter bound. Using extreme points is both faster and more accurate, but
  requires more storage. However, for most applications, the required storage
  for the objects would be significantly greater than that for the 
  extreme points, so this storage overhead is likely not a significant issue.
  Using SIMD to calculate the lower and upper bounds provides a major 
  performance boost. For search algorithms that can be adapted to batch
  their bounding operations, using SIMD should significantly accelerate 
  the method.
*/

} // namespace geom
} // namespace stlib

#if !defined(__geom_kernel_h__)
#define __geom_kernel_h__

#include "stlib/geom/kernel/BBox.h"
#include "stlib/geom/kernel/Ball.h"
#include "stlib/geom/kernel/BallSquared.h"
#include "stlib/geom/kernel/Circle3.h"
#include "stlib/geom/kernel/CircularArc3.h"
#include "stlib/geom/kernel/Hyperplane.h"
#include "stlib/geom/kernel/Line_2.h"
#include "stlib/geom/kernel/MinBall3.h"
#include "stlib/geom/kernel/ParametrizedLine.h"
#include "stlib/geom/kernel/ParametrizedPlane.h"
#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/kernel/SegmentMath.h"
#include "stlib/geom/kernel/SmallestEnclosingBall.h"
#include "stlib/geom/kernel/content.h"
#include "stlib/geom/kernel/distance.h"
#include "stlib/geom/kernel/orientation.h"
#include "stlib/geom/kernel/simplexTopology.h"

#endif
