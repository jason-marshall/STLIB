// -*- C++ -*-

/*!
  \file
  \brief Compute smallest enclosing ball radii.
*/

#if !defined(__geom_SmallestEnclosingBall_h__)
#define __geom_SmallestEnclosingBall_h__

#include "stlib/geom/kernel/content.h"

#include <list>

namespace stlib
{
namespace geom
{


//! Return the squared radius of the smallest enclosing ball.
/*!
  The smallest enclosing ball is either the circumscribed ball or a ball 
  centered on the midpoint of the longest edge with a radius of half the 
  edge length. The circumscribed ball is the smallest enclosing ball if 
  the center is not exterior to the triangle. The midpoint centered ball
  is an enclosing ball if contains the opposite vertex. In this algorithm,
  we compute the radii for both balls. We check if the midpoint ball 
  is valid, and choose the smaller as the smallest enclosing ball.
*/
template<typename _Float, std::size_t _Dimension>
inline
_Float
computeSmallestEnclosingBallSquaredRadius
(std::array<_Float, _Dimension> const& v0,
 std::array<_Float, _Dimension> const& v1,
 std::array<_Float, _Dimension> const& v2)
{
  typedef std::array<_Float, _Dimension> Point;

  // First compute the squared radius of the circumscribed ball.
  _Float radius2 = std::numeric_limits<_Float>::infinity();
  _Float const l01 = ext::squaredDistance(v0, v1);
  _Float const l12 = ext::squaredDistance(v1, v2);
  _Float const l20 = ext::squaredDistance(v2, v0);
  _Float const area2 = computeSquaredArea(v0, v1, v2);
  if (area2 > 0) {
    radius2 = _Float(1./16) * l01 * l12 * l20 / area2;
  }
  // If the smallest enclosing ball for the longest edge also contains the
  // opposite point, check if this radius is smaller. First, identify the
  // longest edge, a-b, and the opposite point, c.
  _Float longestEdge = l01;
  Point const* a = &v0;
  Point const* b = &v1;
  Point const* c = &v2;
  if (l12 >= l01 && l12 >= l20) {
    // v1-v2 is the longest edge.
    longestEdge = l12;
    a = &v1;
    b = &v2;
    c = &v0;
  }
  else if (l20 >= l01 && l20 >= l12) {
    // v2-v0 is the longest edge.
    longestEdge = l20;
    a = &v2;
    b = &v0;
    c = &v1;
  }
  _Float const r2 = 0.25 * longestEdge;
  // If this radius is smaller and the ball encloses the opposite point,
  // use this radius.
  if (r2 < radius2 && ext::squaredDistance(_Float(0.5) * (*a + *b), *c) <= r2) {
    radius2 = r2;
  }
  return radius2;
}


//! The smallest ball with a set of n <= _Dimension+1 points on the boundary.
template<std::size_t _Dimension, typename _T = double>
class SmallestEnclosingBallBoundary
{
  //
  // Types
  //
public:

  //! A Cartesian point.
  typedef std::array<_T, _Dimension> Point;

  //
  // Member data.
  //
private:

  // Size.
  std::size_t m;
  // Number of support points.
  std::size_t s;
  Point q0;

  _T z[_Dimension + 1];
  _T f[_Dimension + 1];
  Point v[_Dimension + 1];
  Point a[_Dimension + 1];

  Point c[_Dimension + 1];
  _T squaredRadii[_Dimension + 1];

  // refers to some c[j]
  Point* currentCenter;
  _T currentSquaredRadius;

public:
  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default copy constructor, assignment operator, and destructor.
  */
  // @{

  SmallestEnclosingBallBoundary()
  {
    reset();
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Accesors.
  // @{

  const Point&
  center() const
  {
    return *currentCenter;
  }

  _T
  squaredRadius() const
  {
    return currentSquaredRadius;
  }

  std::size_t
  size() const
  {
    return m;
  }

  std::size_t
  supportSize() const
  {
    return s;
  }

  _T
  excess(const Point& p) const
  {
    return ext::squaredDistance(p, *currentCenter) - currentSquaredRadius;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  // @{

  //! Generates an empty sphere with zero size and no support points.
  void
  reset();

  bool
  push(const Point& p);

  void
  pop()
  {
    --m;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Validation.
  // @{

  _T
  slack() const;

  // @}
};

template<std::size_t _Dimension, typename _T>
void
SmallestEnclosingBallBoundary<_Dimension, _T>::reset()
{
  m = s = 0;
  // we misuse c[0] for the center of the empty sphere
  std::fill(c[0].begin(), c[0].end(), 0);
  currentCenter = &c[0];
  currentSquaredRadius = -1;
}




template<std::size_t _Dimension, typename _T>
bool
SmallestEnclosingBallBoundary<_Dimension, _T>::push(const Point& p)
{
  std::size_t i, j;
  _T eps = 1e-32;
  if (m == 0) {
    q0 = p;
    c[0] = q0;
    squaredRadii[0] = 0;
  }
  else {
    // set v_m to Q_m
    v[m] = p - q0;

    // compute the a_{m,i}, i < m
    for (i = 1; i != m; ++i) {
      a[m][i] = 2. * ext::dot(v[i], v[m]) / z[i];
    }

    // update v_m to Q_m-\bar{Q}_m
    for (i = 1; i != m; ++i) {
      for (j = 0; j != _Dimension; ++j) {
        v[m][j] -= a[m][i] * v[i][j];
      }
    }

    // compute z_m
    z[m] = 2. * ext::squaredMagnitude(v[m]);

    // reject push if z_m too small
    if (z[m] < eps * currentSquaredRadius) {
      return false;
    }

    // update c, sqr_r
    _T e = ext::squaredDistance(p, c[m - 1]) - squaredRadii[m - 1];
    f[m] = e / z[m];

    for (i = 0; i != _Dimension; ++i) {
      c[m][i] = c[m - 1][i] + f[m] * v[m][i];
    }
    squaredRadii[m] = squaredRadii[m - 1] + 0.5 * e * f[m];
  }
  currentCenter = &c[m];
  currentSquaredRadius = squaredRadii[m];
  s = ++m;
  return true;
}

template<std::size_t _Dimension, typename _T>
_T
SmallestEnclosingBallBoundary<_Dimension, _T>::slack() const
{
  _T l[_Dimension + 1];
  _T minL = 0;
  l[0] = 1;
  for (std::size_t i = s - 1; i > 0; --i) {
    l[i] = f[i];
    for (std::size_t k = s - 1; k > i; --k) {
      l[i] -= a[k][i] * l[k];
    }
    if (l[i] < minL) {
      minL = l[i];
    }
    l[0] -= l[i];
  }
  if (l[0] < minL) {
    minL = l[0];
  }
  return (minL < 0) ? -minL : 0;
}


//! Smallest enclosing ball of a set of points.
template<std::size_t _Dimension, typename _T = double>
class SmallestEnclosingBall
{
  //
  // Types.
  //
public:
  //! A Cartesian point.
  typedef std::array<_T, _Dimension> Point;
  typedef typename std::list<Point>::iterator iterator;
  typedef typename std::list<Point>::const_iterator const_iterator;

  //
  // Member data.
  //
private:
  //! Internal point set.
  std::list<Point> L;
  //! The current ball.
  SmallestEnclosingBallBoundary<_Dimension, _T> B;
  // The end of the support set.
  iterator supportEnd;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    We use the default destructor. The copy constructor and assignment
    operator are private.
  */
  // @{
public:

  // Create an empty ball.
  SmallestEnclosingBall()
  {
  }

  // Copy p to the internal point set.
  void
  checkIn(const Point& p)
  {
    L.push_back(p);
  }

  // Build the smallest enclosing ball of the internal point set.
  void
  build()
  {
    B.reset();
    supportEnd = L.begin();
    pivot_mb(L.end());
  }

  // Build the smallest enclosing ball with the specified internal point set.
  template<typename _InputIterator>
  void
  build(_InputIterator first, _InputIterator last)
  {
    L.clear();
    L.insert(L.begin(), first, last);
    build();
  }

private:

  // Not implemented.
  SmallestEnclosingBall(const SmallestEnclosingBall&);

  // Not implemented.
  SmallestEnclosingBall&
  operator=(const SmallestEnclosingBall&);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the center of the ball (undefined if ball is empty).
  const Point&
  center() const
  {
    return B.center();
  }

  //! Return the squared radius of the ball (-1 if ball is empty).
  _T
  squaredRadius() const
  {
    return B.squaredRadius();
  }

  //! Return the size of the internal point set.
  std::size_t
  numInternalPoints() const
  {
    return L.size();
  }

  //! Return the beginning of the internal point set.
  const_iterator
  pointsBegin() const
  {
    return L.begin();
  }

  //! Return the end of the internal point set.
  const_iterator
  pointsEnd() const
  {
    return L.end();
  }

  //! Return the size of the support point set.
  /*! The point set has the following properties:
    - There are at most _Dimension + 1 support points.
    - All support points are on the boundary of the computed ball.
    - The smallest enclosing ball of the support point set equals the
    smallest enclosing ball of the internal point set. */
  std::size_t
  numSupportPoints() const
  {
    return B.supportSize();
  }

  //! Return the beginning of the internal point set.
  const_iterator
  supportPointsBegin() const
  {
    return L.begin();
  }

  //! Return the end of the internal point set.
  const_iterator
  supportPointsEnd() const
  {
    return supportEnd;
  }

  //! Assess the quality of the computed ball.
  /*! The return value is the maximum squared distance of any support
   point or point outside the ball to the boundary of the ball,
   divided by the squared radius of the ball. If everything went
   fine, this will be less than e-15 and says that the computed ball
   approximately contains all the internal points and has all the
   support points on the boundary.

   The slack parameter that is set by the method says something about
   whether the computed ball is really the *smallest* enclosing ball
   of the support points; if everything went fine, this value will be
   0; a positive value may indicate that the ball is not smallest
   possible, with the deviation from optimality growing with the
   slack. */
  _T
  accuracy(_T* slack) const;

  // returns true if the accuracy is below the given tolerance and the
  // slack is 0
  bool
  isValid(const _T tolerance = std::numeric_limits<_T>::epsilon())
  const
  {
    _T slack;
    return ((accuracy(&slack) < tolerance) && (slack == 0));
  }

  // @}
  //
  // Private methods.
  //
private:

  void
  mtf_mb(iterator k);

  void
  pivot_mb(iterator k);

  void
  moveToFront(iterator j);

  _T
  maxExcess(iterator t, iterator i, iterator* pivot) const;
};


template<std::size_t _Dimension, typename _T>
void
SmallestEnclosingBall<_Dimension, _T>::
mtf_mb(iterator i)
{
  supportEnd = L.begin();
  if (B.size() == _Dimension + 1) {
    return;
  }
  for (iterator k = L.begin(); k != i;) {
    iterator j = k++;
    if (B.excess(*j) > 0) {
      if (B.push(*j)) {
        mtf_mb(j);
        B.pop();
        moveToFront(j);
      }
    }
  }
}

template<std::size_t _Dimension, typename _T>
void
SmallestEnclosingBall<_Dimension, _T>::
moveToFront(iterator j)
{
  if (supportEnd == j) {
    supportEnd++;
  }
  L.splice(L.begin(), L, j);
}


template<std::size_t _Dimension, typename _T>
void
SmallestEnclosingBall<_Dimension, _T>::
pivot_mb(iterator i)
{
  iterator t = ++L.begin();
  mtf_mb(t);
  _T maxE;
  _T oldSquaredRadius = -1;
  do {
    iterator pivot;
    maxE = maxExcess(t, i, &pivot);
    if (maxE > 0) {
      t = supportEnd;
      if (t == pivot) {
        ++t;
      }
      oldSquaredRadius = B.squaredRadius();
      B.push(*pivot);
      mtf_mb(supportEnd);
      B.pop();
      moveToFront(pivot);
    }
  }
  while (maxE > 0 && B.squaredRadius() > oldSquaredRadius);
}


template<std::size_t _Dimension, typename _T>
_T
SmallestEnclosingBall<_Dimension, _T>::
maxExcess(iterator t, iterator i, iterator* pivot) const
{
  const Point& c = B.center();
  const _T sqr_r = B.squaredRadius();
  _T e, maxE = 0;
  for (iterator k = t; k != i; ++k) {
    const Point& p = *k;
    e = ext::squaredDistance(p, c) - sqr_r;
    if (e > maxE) {
      maxE = e;
      *pivot = k;
    }
  }
  return maxE;
}


template<std::size_t _Dimension, typename _T>
_T
SmallestEnclosingBall<_Dimension, _T>::
accuracy(_T* slack) const
{
  _T e, maxE = 0;
  std::size_t numSupp = 0;
  const_iterator i;
  for (i = L.begin(); i != supportEnd; ++i, ++numSupp) {
    if ((e = std::abs(B.excess(*i))) > maxE) {
      maxE = e;
    }
  }

  // you've found a non-numerical problem if the following ever fails
  assert(numSupp == numSupportPoints());

  for (i = supportEnd; i != L.end(); ++i) {
    if ((e = B.excess(*i)) > maxE) {
      maxE = e;
    }
  }

  *slack = B.slack();
  return maxE / squaredRadius();
}

} // namespace geom
} // namespace stlib

#endif
