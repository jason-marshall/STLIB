// -*- C++ -*-

/*!
  \file BallQueryBalls.h
  \brief Ball queries for a set of balls.
*/

#if !defined(__geom_BallQueryBalls_h__)
#define __geom_BallQueryBalls_h__

#include "stlib/geom/kernel/Ball.h"

#include <iterator>
#include <functional>

namespace stlib
{
namespace geom
{


//! Functor for accessing the center from an iterator to a ball.
template<typename _BallIterator>
struct GetBallIteratorCenter :
    public std::unary_function<_BallIterator,
    typename std::iterator_traits<_BallIterator>::value_type::Point> {
  typedef std::unary_function
  <_BallIterator,
  typename std::iterator_traits<_BallIterator>::value_type::Point> Base;

  const typename Base::result_type&
  operator()(typename Base::argument_type i) const
  {
    return i->center;
  }
};


//! Ball queries for a set of balls.
/*!
  CONTINUE
*/
template<std::size_t N, typename _T,
         template<std::size_t, typename> class _Orq>
class BallQueryBalls
{
  //
  // Types.
  //
private:

  typedef typename std::vector<Ball<_T, N> >::const_iterator BallIterator;
  typedef _Orq<N, GetBallIteratorCenter<BallIterator> > Orq;

  //
  // Data
  //
public:

  //! The sequence of balls.
  const std::vector<Ball<_T, N> > balls;
  //! The maximum radius of the balls.
  const _T maxRadius;

private:

  //! The ORQ data structure.
  const Orq _orq;

  //
  // Not implemented.
  //
private:

  BallQueryBalls&
  operator=(const BallQueryBalls&);

  //--------------------------------------------------------------------------
  /*! \name Constructors.
    Use the synthesized copy constructor and destructor.
  */
  // @{
public:

  //! Construct from the sequence of balls.
  template<typename _InputIterator>
  BallQueryBalls(_InputIterator begin, _InputIterator end) :
    // Copy the sequence of balls.
    balls(begin, end),
    // Calculate the maximum radius.
    maxRadius(computeMaxRadius(balls)),
    // Build the ORQ data structure.
    _orq(balls.begin(), balls.end())
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Window Queries.
  // @{
public:

  //! Get the indices of the balls that intersect the query ball.
  /*!
    \return The number of balls that do intersect.
  */
  template<typename _OutputIterator>
  std::size_t
  query(_OutputIterator iter, const Ball<_T, N>& ball) const
  {
    // Bound the ball and enlarge by the max radius.
    BBox<_T, N> enlarged = specificBBox<BBox<_T, N> >(ball);
    offset(&enlarged, maxRadius);
    // Perform an ORQ to get the candidates.
    std::vector<BallIterator> candidates;
    _orq.computeWindowQuery(std::back_inserter(candidates), enlarged);
    // Determine which of the candidates intersect the query ball.
    std::size_t count = 0;
    for (std::size_t i = 0; i != candidates.size(); ++i) {
      // If the candidate ball intersects the query ball.
      if (doIntersect(*candidates[i], ball)) {
        // Record the index.
        *iter++ = std::distance(balls.begin(), candidates[i]);
        ++count;
      }
    }
    return count;
  }

  // @}
private:

  static
  _T
  computeMaxRadius(const std::vector<Ball<_T, N> > balls)
  {
    _T r = 0;
    for (std::size_t i = 0; i != balls.size(); ++i) {
      if (balls[i].radius > r) {
        r = balls[i].radius;
      }
    }
    return r;
  }
};

} // namespace geom
}

#endif
