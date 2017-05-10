// -*- C++ -*-

/*!
  \file IntersectionQuerySimplexBalls.h
  \brief Determine which balls intersect a simplex.
*/

#if !defined(__geom_IntersectionQuerySimplexBalls_h__)
#define __geom_IntersectionQuerySimplexBalls_h__

#include "stlib/geom/orq/BallQueryBalls.h"
#include "stlib/geom/mesh/simplex/simplex_distance.h"

namespace stlib
{
namespace geom
{

//! Determine which balls intersect a simplex.
/*!
  CONTINUE
*/
template<std::size_t N, typename _T,
         template<std::size_t, typename> class _Orq>
class IntersectionQuerySimplexBalls
{
  //
  // Types.
  //
public:

  //! A Cartesian point.
  typedef std::array<_T, N> Point;
  //! A ball.
  typedef geom::Ball<_T, N> Ball;
  //! A simplex of points.
  typedef std::array < Point, N + 1 > Simplex;

  //
  // Data
  //
private:

  //! The ball query data structure.
  const BallQueryBalls<N, _T, _Orq> _ballQuery;

  //
  // Not implemented.
  //
private:

  IntersectionQuerySimplexBalls&
  operator=(const IntersectionQuerySimplexBalls&);

  //--------------------------------------------------------------------------
  /*! \name Constructors.
    Use the synthesized copy constructor and destructor.
  */
  // @{
public:

  //! Construct from the sequence of balls.
  template<typename _InputIterator>
  IntersectionQuerySimplexBalls(_InputIterator begin, _InputIterator end) :
    // Build the ball query data structure.
    _ballQuery(begin, end)
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Window Queries.
  // @{
public:

  //! Get the indices of the balls that intersect the simplex.
  /*!
    \return The number of balls that do intersect.
  */
  template<typename _OutputIterator>
  std::size_t
  query(_OutputIterator iter, const Simplex& simplex) const
  {
    // Make a bounding ball that encloses the simplex.
    // First compute the center.
    Point center = simplex[0];
    for (std::size_t i = 1; i != simplex.size(); ++i) {
      center += simplex[i];
    }
    center /= _T(simplex.size());
    // Then the radius.
    _T squaredRadius = 0;
    for (std::size_t i = 0; i != simplex.size(); ++i) {
      const _T d2 = ext::squaredDistance(center, simplex[i]);
      if (d2 > squaredRadius) {
        squaredRadius = d2;
      }
    }
    const Ball boundingBall = {center, std::sqrt(squaredRadius)};

    // Perform a ball query to get the candidate balls.
    std::vector<std::size_t> candidates;
    _ballQuery.query(std::back_inserter(candidates), boundingBall);

    // Determine which of the candidates intersect the simplex.
    // The functor for computing distance to a simplex.
    SimplexDistance<N, N, _T> simplexDistance(simplex);
    std::size_t count = 0;
    for (std::size_t i = 0; i != candidates.size(); ++i) {
      const Ball ball = _ballQuery.balls[candidates[i]];
      // If the candidate ball intersects the simplex.
      if (simplexDistance(ball.center) <= ball.radius) {
        // Record the index.
        *iter++ = candidates[i];
        ++count;
      }
    }
    return count;
  }

  // @}
};

} // namespace geom
}

#endif
