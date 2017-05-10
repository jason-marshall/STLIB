// -*- C++ -*-

/**
  \file
  \brief Functions for determining the distance to a simplex.
*/

#if !defined(__geom_simplex_distance_h__)
#define __geom_simplex_distance_h__

#include "stlib/geom/kernel/simplexTopology.h"

#include "stlib/geom/kernel/Line_2.h"
#include "stlib/geom/kernel/Hyperplane.h"

namespace stlib
{
namespace geom
{

/// Compute the distance to a simplex.
/** 
  If the simplex dimension is equal to the space dimension, then the
  distance is signed. For example, the distance from a 3-D point to a 
  tetrahedron is negative if the point is inside and positive if the point
  is outside. If the simplex dimension is less than the space dimension,
  then the distance is unsigned. For example, the distance from a 3-D point
  to a triangle face is non-negative.
*/
template<std::size_t SpaceD, std::size_t _M = SpaceD, typename _T = double>
class SimplexDistance;

//-----------------------------------------------------------------------------
/** \defgroup simplex_distance Distance to a Simplex
 */
//@{

//---------------------------------------------------------------------------
// Inside tests.
//---------------------------------------------------------------------------

/// Return true if the 1-D point is inside the 1-D simplex.
/**
  \param s is a 1-D simplex (an interval) in 1-D space.
  \param x is a point in 1-D Cartesian space.

  \return true if the point is inside the simplex.
*/
template<typename T>
bool
isIn(const std::array < std::array<T, 1>, 1 + 1 > & s,
     const std::array<T, 1>& x);

/// Return true if the 2-D point is inside the 2-D simplex.
/**
  \param s is a 2-D simplex (a triangle) in 2-D space with Cartesian points
  as vertices.
  \param x is a point in 2-D Cartesian space.

  Consider the supporting lines of the three faces.  The point is in
  the simplex if and only if it has negative distance to each of the
  lines.

  \return true if the point is inside the simplex.
*/
template<typename T>
bool
isIn(const std::array < std::array<T, 2>, 2 + 1 > & s,
     const std::array<T, 2>& x);

/// Return true if the 3-D point is inside the 3-D simplex.
/**
  \param s is a 3-D simplex (a tetrahedran) in 3-D space with Cartesian points
  as vertices.
  \param x is a point in 3-D Cartesian space.

  Consider the supporting planes of the four faces.  The point is in
  the simplex if and only if it has negative distance to each of the
  planes.

  \return true if the point is inside the simplex.
*/
template<typename T>
bool
isIn(const std::array < std::array<T, 3>, 3 + 1 > & s,
     const std::array<T, 3>& x);


//---------------------------------------------------------------------------
// Interior distance.
//---------------------------------------------------------------------------


/// Compute the distance from the 1-D, interior point to the 1-D simplex.
/**
  \param s is a 1-D simplex (an interval) in 1-D space.
  \param x is a point that lies inside the simplex.

  The point must be inside the simplex.  This means that the distance
  will be non-positive.

  \return the distance from the point to the simplex.
*/
template<typename T>
T
computeDistanceInterior(const std::array < std::array<T, 1>, 1 + 1 > & s,
                        const std::array<T, 1>& x);


/// Compute the distance from the 2-D, interior point to the 2-D simplex.
/**
  \param s is a 2-D simplex (a triangle) in 2-D space with Cartesian points
  as vertices.
  \param x is a point that lies inside the simplex.

  The point must be inside the simplex.  This means that the distance
  will be non-positive.

  \return the distance from the point to the simplex.
*/
template<typename T>
T
computeDistanceInterior(const std::array < std::array<T, 2>, 2 + 1 > & s,
                        const std::array<T, 2>& x);


/// Compute the distance from the 3-D, interior point to the 3-D simplex.
/**
  \param s is a 3-D simplex (a tetrahedron) in 3-D space with Cartesian points
  as vertices.
  \param x is a point that lies inside the simplex.

  The point must be inside the simplex.  This means that the distance
  will be non-positive.

  \return the distance from the point to the simplex.
*/
template<typename T>
T
computeDistanceInterior(const std::array < std::array<T, 3>, 3 + 1 > & s,
                        const std::array<T, 3>& x);


//---------------------------------------------------------------------------
// Distance.
//---------------------------------------------------------------------------

/// Compute the signed distance from the 1-D point to the 1-D simplex.
/**
  \param s is a 1-D simplex (an interval) in 1-D space.
  \param x is a Cartesian point.

  \return the signed distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 1>, 1 + 1 > & s,
                const std::array<T, 1>& x);

/// Compute the unsigned distance from the 2-D point to the 1-D simplex.
/**
  \param s is a 1-D simplex (a line segment) in 2-D space.
  \param x is a Cartesian point in 2-D space.

  \return the unsigned distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 2>, 1 + 1 > & s,
                const std::array<T, 2>& x);

/// Compute the unsigned distance from the 3-D point to the 1-simplex.
/**
  \param s is a 1-simplex (a line segment) in 3-D space.
  \param x is a Cartesian point in 3-D space.

  \return the unsigned distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 3>, 1 + 1 > & s,
                const std::array<T, 3>& x);

/// Compute the signed distance from the 2-D point to the 2-D simplex.
/**
  \param s is a 2-D simplex (a triangle) in 2-D space with Cartesian points
  as vertices.
  \param x is a Cartesian point.

  \return the signed distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 2>, 2 + 1 > & s,
                const std::array<T, 2>& x);

/// Compute the unsigned distance from the 3-D point to the 2-D simplex.
/**
  \param s is a 2-simplex (a triangle) in 3-D space with 3-D Cartesian points
  as vertices.
  \param x is a 3-D Cartesian point.

  \return the unsigned distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 3>, 2 + 1 > & s,
                const std::array<T, 3>& x);

/// Compute the signed distance from the 3-D point to the 3-D simplex.
/**
  \param s is a 3-D simplex (a tetrahedron) in 3-D space with Cartesian points
  as vertices.
  \param x is a Cartesian point.

  \return the signed distance from the point to the simplex.
*/
template<typename T>
T
computeDistance(const std::array < std::array<T, 3>, 3 + 1 > & s,
                const std::array<T, 3>& x);


//---------------------------------------------------------------------------
// Unsigned distance.
//---------------------------------------------------------------------------


/// Compute the unsigned distance from the 1-D point to the 1-D simplex.
/**
  \param s is a 1-D simplex (an interval) in 1-D space.
  \param x is a Cartesian point.

  \return the unsigned distance from the point to the simplex.
  The distance is zero inside the simplex and positive outside.
*/
template<typename T>
inline
T
computeUnsignedDistance(const std::array < std::array<T, 1>, 1 + 1 > & s,
                        const std::array<T, 1>& x) {
  return std::max(T(0), computeDistance(s, x));
}


/// Compute the unsigned distance from the 2-D point to the 2-D simplex.
/**
  \param s is a 2-D simplex (a triangle) in 2-D space.
  \param x is a 2-D Cartesian point.

  \return the unsigned distance from the point to the simplex.
  The distance is zero inside the simplex and positive outside.
*/
template<typename T>
inline
T
computeUnsignedDistance(const std::array < std::array<T, 2>, 2 + 1 > & s,
                        const std::array<T, 2>& x) {
  return std::max(T(0), computeDistance(s, x));
}


/// Compute the unsigned distance from the 3-D point to the 3-D simplex.
/**
  \param s is a 3-D simplex (a tetrahedron) in 3-D space.
  \param x is a 3-D Cartesian point.

  \return the unsigned distance from the point to the simplex.
  The distance is zero inside the simplex and positive outside.
*/
template<typename T>
inline
T
computeUnsignedDistance(const std::array < std::array<T, 3>, 3 + 1 > & s,
                        const std::array<T, 3>& x) {
  return std::max(0., computeDistance(s, x));
}


//---------------------------------------------------------------------------
// Signed distance.
//---------------------------------------------------------------------------


/// Compute the signed distance from the N-D point with normal to the N-D point.
/**
  \param p is an N-D Cartesian point.
  \param n is a normal direction for the point.
  \param x is an N-D Cartesian point.

  \return the signed distance.  If \f$(x - p) \cdot n > 0\f$, then the
  distance has positive sign.
*/
template<std::size_t N, typename T>
T
computeSignedDistance(const std::array<T, N>& p,
                      const std::array<T, N>& n,
                      const std::array<T, N>& x);


/// Compute the signed distance from the 2-D point to the 1-simplex.
/**
  \param s is a 1-D simplex (a line segment) in 2-D space.
  \param x is a Cartesian point in 2-D space.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below, but not to the side of the simplex.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 2>, 1 + 1 > & s,
                      const std::array<T, 2>& x);


/// Compute the signed distance and closest point from the 2-D point to the 1-simplex.
/**
  \param s is a 1-D simplex (a line segment) in 2-D space.
  \param x is a Cartesian point in 2-D space.
  \param closestPoint is the closest point on the simplex.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below, but not to the side of the simplex.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 2>, 1 + 1 > & s,
                      const std::array<T, 2>& x,
                      std::array<T, 2>* closestPoint);


/// Compute the signed distance from the 3-D point to the 2-simplex.
/**
  \param s is a 2-simplex (a triangle) in 3-D space.
  \param x is a Cartesian point in 3-D space.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below, but not to the side of the simplex.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 3>, 2 + 1 > & s,
                      const std::array<T, 3>& x);


/// Compute the signed distance and closest point from the 3-D point to the 2-simplex.
/**
  \param s is a 2-simplex (a triangle) in 3-D space.
  \param n is the unit normal to the triangle face.
  \param x is a Cartesian point in 3-D space.
  \param closestPoint is the closest point on the simplex.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below, but not to the side of the simplex.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 3>, 2 + 1 > & s,
                      const std::array<T, 3>& n,
                      const std::array<T, 3>& x,
                      std::array<T, 3>* closestPoint);


/// Compute the signed distance from the 3-D point to the 1-simplex with normal.
/**
  \param s is a 1-simplex (a line segment) in 3-D space.
  \param n is a normal direction for line segment.  The normal direction
  determines the sign of the distance.
  \param x is a Cartesian point in 3-D space.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below the line segment, but not to the side.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 3>, 1 + 1 > & s,
                      const std::array<T, 3>& n,
                      const std::array<T, 3>& x);


/// Compute the signed distance and closest point from the 3-D point to the 1-simplex with normal.
/**
  \param s is a 1-simplex (a line segment) in 3-D space.
  \param n is a normal direction for line segment.  The normal direction
  determines the sign of the distance.
  \param x is a Cartesian point in 3-D space.
  \param closestPoint is the closest point on the simplex.

  \return the signed distance from the point to the simplex.  The signed
  distance is defined above and below the line segment, but not to the side.
  If the point is not above or below the simplex, return
  \c std::numeric_limits<T>::max().
*/
template<typename T>
T
computeSignedDistance(const std::array < std::array<T, 3>, 1 + 1 > & s,
                      const std::array<T, 3>& n,
                      const std::array<T, 3>& x,
                      std::array<T, 3>* closestPoint);


//---------------------------------------------------------------------------
// Project to a lower dimension.
//---------------------------------------------------------------------------


/// Project the simplex and the point in 2-D to 1-D.
/**
  The first point in the 2-D simplex will be mapped to the origin; the second
  point will be mapped to the positive x axis.  The 2-D point will be mapped
  to the x-axis.

  \param s2 is the 1-simplex in 2-D.
  \param x2 is the 2-D point.
  \param s1 is the mapped simplex, a 1-simplex in 1-D.
  \param x1 is the mapped point, a 1-D point.
*/
template<typename T>
void
project(const std::array < std::array<T, 2>, 1 + 1 > & s2,
        const std::array<T, 2>& x2,
        std::array < std::array<T, 1>, 1 + 1 > * s1,
        std::array<T, 1>* x1);

/// Project the simplex and the point in 2-D to 1-D.
/**
  The first point in the 2-D simplex will be mapped to the origin; the second
  point will be mapped to the positive x axis.  The 2-D point will be mapped
  to the x-axis.

  \param s2 is the 1-simplex in 2-D.
  \param x2 is the 2-D point.
  \param s1 is the mapped simplex, a 1-simplex in 1-D.
  \param x1 is the mapped point, a 1-D point.
  \param y1 is the normal offset of x2 from s2.
*/
template<typename T>
void
project(const std::array < std::array<T, 2>, 1 + 1 > & s2,
        const std::array<T, 2>& x2,
        std::array < std::array<T, 1>, 1 + 1 > * s1,
        std::array<T, 1>* x1,
        std::array<T, 1>* y1);

/// Project the simplex and the point in 3-D to 2-D.
/**
  The first point in the 3-D simplex will be mapped to the origin; the second
  point will be mapped to the positive x axis.  The triangle will have the
  positive orientation in the 2-D plane.  The 3-D point will be mapped
  to the xy-plane.

  \param s3 is the 2-simplex in 3-D.
  \param x3 is the 3-D point.
  \param s2 is the mapped simplex, a 2-simplex in 2-D.
  \param x2 is the mapped point, a 2-D point.
*/
template<typename T>
void
project(const std::array < std::array<T, 3>, 2 + 1 > & s3,
        const std::array<T, 3>& x3,
        std::array < std::array<T, 2>, 2 + 1 > * s2,
        std::array<T, 2>* x2);

/// Project the simplex and the point in 3-D to 2-D.
/**
  The first point in the 3-D simplex will be mapped to the origin; the second
  point will be mapped to the positive x axis.  The triangle will have the
  positive orientation in the 2-D plane.  The 3-D point will be mapped
  to the xy-plane.

  \param s3 is the 2-simplex in 3-D.
  \param x3 is the 3-D point.
  \param s2 is the mapped simplex, a 2-simplex in 2-D.
  \param x2 is the mapped point, a 2-D point.
  \param z1 is the normal offset of x3 from s3.
*/
template<typename T>
void
project(const std::array < std::array<T, 3>, 2 + 1 > & s3,
        const std::array<T, 3>& x3,
        std::array < std::array<T, 2>, 2 + 1 > * s2,
        std::array<T, 2>* x2,
        std::array<T, 1>* z1);


//---------------------------------------------------------------------------
// Closest Point.
//---------------------------------------------------------------------------

/// Return the unsigned distance from the 2-D point to the 1-D simplex and compute the closest point.
/**
  \param simplex is a 1-D simplex (a line segment) in 2-D space.
  \param point is a Cartesian point in 2-D space.
  \param closestPoint is the closest point on the simplex.

  \return the unsigned distance from the point to the simplex.
*/
template<typename T>
T
computeClosestPoint(const std::array < std::array<T, 2>, 1 + 1 > & simplex,
                    const std::array<T, 2>& point,
                    std::array<T, 2>* closestPoint);

//@}

} // namespace geom
} // namespace stlib

#define __geom_simplex_distance_ipp__
#include "stlib/geom/mesh/simplex/simplex_distance.ipp"
#undef __geom_simplex_distance_ipp__

#endif
