// -*- C++ -*-

#if !defined(__levelSet_geometry_ipp__)
#error This file is an implementation detail of geometry.
#endif

namespace stlib
{
namespace levelSet
{


template<typename _T, std::size_t _D>
inline
bool
isInside(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::array<_T, _D>& x)
{
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (isInside(balls[i], x)) {
      return true;
    }
  }
  return false;
}


template<typename _T, std::size_t _D>
inline
bool
isInside(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::size_t n,
         const std::array<_T, _D>& x)
{
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i != n && isInside(balls[i], x)) {
      return true;
    }
  }
  return false;
}


// Return true if the point is inside one of the active balls.
template<typename _T, std::size_t _D>
inline
bool
isInside(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::vector<std::size_t>& active,
         const std::array<_T, _D>& x)
{
  for (std::size_t i = 0; i != active.size(); ++i) {
#ifdef STLIB_DEBUG
    assert(active[i] < balls.size());
#endif
    if (isInside(balls[active[i]], x)) {
      return true;
    }
  }
  return false;
}


// Get the set of balls that intersect the n_th ball.
template<typename _T, std::size_t _D>
inline
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                const std::size_t n, std::vector<std::size_t>* intersecting)
{
  assert(n < balls.size());
  intersecting->clear();
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Don't consider self-intersection.
    if (i == n) {
      continue;
    }
    if (doIntersect(balls[n], balls[i])) {
      intersecting->push_back(i);
    }
  }
}


// Get the set of balls that intersect one of the two specified balls.
template<typename _T, std::size_t _D>
inline
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                const std::size_t m, const std::size_t n,
                std::vector<std::size_t>* intersecting)
{
#ifdef STLIB_DEBUG
  assert(m < balls.size() && n < balls.size());
#endif
  intersecting->clear();
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Don't consider self-intersection.
    if (i == m || i == n) {
      continue;
    }
    if (doIntersect(balls[m], balls[i]) || doIntersect(balls[n], balls[i])) {
      intersecting->push_back(i);
    }
  }
}


// Get the set of balls that intersect the specified ball.
template<typename _T, std::size_t _D>
inline
void
getIntersecting(const std::vector<geom::Ball<_T, _D> >& balls,
                const geom::Ball<_T, _D>& b,
                std::vector<std::size_t>* intersecting)
{
  intersecting->clear();
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (doIntersect(b, balls[i])) {
      intersecting->push_back(i);
    }
  }
}


// Return true if the intersection point is on the surface.
template<typename _T>
inline
bool
isOnSurface(const std::vector<geom::Ball<_T, 2> >& balls,
            const std::size_t index1, const std::size_t index2,
            const std::array<_T, 2>& x)
{
  // For each of the balls except the two that intersect at the specified
  // point.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i == index1 || i == index2) {
      continue;
    }
    // Return false if the point is inside the ball.
    if (isInside(balls[i], x)) {
      return false;
    }
  }
  // If the point was not found to be inside a ball, it is outside.
  return true;
}


// Return true if the intersection point is on the surface.
template<typename _T>
inline
bool
isOnSurface(const std::vector<geom::Ball<_T, 3> >& balls,
            const std::size_t index1, const std::size_t index2,
            const std::size_t index3,
            const std::array<_T, 3>& x)
{
  // For each of the balls except the three that intersect at the specified
  // point.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i == index1 || i == index2 || i == index3) {
      continue;
    }
    // Return false if the point is inside the ball.
    if (isInside(balls[i], x)) {
      return false;
    }
  }
  // If the point was not found to be inside a ball, it is outside.
  return true;
}


} // namespace levelSet
}
