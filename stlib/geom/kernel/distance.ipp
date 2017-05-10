// -*- C++ -*-

#if !defined(__geom_distance_ipp__)
#error This file is an implementation detail of distance.
#endif

namespace stlib
{
namespace geom
{

template<typename _T>
inline
_T
computeUpperBoundSquaredDistance(const BBox<_T, 3>& a, const BBox<_T, 3>& b)
{
  typedef typename BBox<_T, 3>::Point Point;

  // Bound on the squared distance.
  _T bound = std::numeric_limits<_T>::max();
  // For each dimension.
  for (std::size_t x = 0; x != 3; ++x) {
    const Point& p0 = a.lower;
    const Point& p1 = a.upper;
    const Point& q0 = b.lower;
    const Point& q1 = b.upper;
    // Compute the squared distance between the supporting planes of the
    // closest pair of faces that are orthogonal to this dimension.
    _T squaredDistance =
      std::min(std::min((p0[x] - q0[x]) * (p0[x] - q0[x]),
                        (p0[x] - q1[x]) * (p0[x] - q1[x])),
               std::min((p1[x] - q0[x]) * (p1[x] - q0[x]),
                        (p1[x] - q1[x]) * (p1[x] - q1[x])));
    // Add the maximum of the vertex distances in the other dimensions to
    // obtain the maximum distance between the closest faces.
    std::size_t i = (x + 1) % 3;
    squaredDistance +=
      std::max(std::max((p0[i] - q0[i]) * (p0[i] - q0[i]),
                        (p0[i] - q1[i]) * (p0[i] - q1[i])),
               std::max((p1[i] - q0[i]) * (p1[i] - q0[i]),
                        (p1[i] - q1[i]) * (p1[i] - q1[i])));
    i = (x + 2) % 3;
    squaredDistance +=
      std::max(std::max((p0[i] - q0[i]) * (p0[i] - q0[i]),
                        (p0[i] - q1[i]) * (p0[i] - q1[i])),
               std::max((p1[i] - q0[i]) * (p1[i] - q0[i]),
                        (p1[i] - q1[i]) * (p1[i] - q1[i])));
    if (squaredDistance < bound) {
      bound = squaredDistance;
    }
  }
  return bound;
}


template<std::size_t N, typename _T>
inline
_T
computeLowerBoundSquaredDistance(const BBox<_T, N>& box,
                                 const std::array<_T, N>& x)
{
  _T d2 = 0;
  for (std::size_t i = 0; i != N; ++i) {
    _T const d = std::max(box.lower[i] - x[i], _T(0)) +
      std::max(x[i] - box.upper[i], _T(0));
    d2 += d * d;
  }
  return d2;

#if 0
  // This implementation is less efficient.
  _T d2;
  _T dist2 = 0;
  for (std::size_t n = 0; n != N; ++n) {
    if (x[n] <= box.lower[n]) {
      d2 = (box.lower[n] - x[n]) *
           (box.lower[n] - x[n]);
    }
    else if (x[n] >= box.upper[n]) {
      d2 = (x[n] - box.upper[n]) *
           (x[n] - box.upper[n]);
    }
    else {
      d2 = 0;
    }
    dist2 += d2;
  }
  return dist2;
#endif
}


template<std::size_t N, typename _T>
inline
_T
computeUpperBoundSquaredDistance(const BBox<_T, N>& box,
                                 const std::array<_T, N>& x)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif
  // Squared distances in each dimension to the supporting planes of the faces.
  _T lo[N], hi[N];
  _T farthest = 0;
  for (std::size_t n = 0; n != N; ++n) {
    lo[n] = (x[n] - box.lower[n]) *
            (x[n] - box.lower[n]);
    hi[n] = (x[n] - box.upper[n]) *
            (x[n] - box.upper[n]);
    if (hi[n] < lo[n]) {
      std::swap(lo[n], hi[n]);
    }
    farthest += hi[n];
  }

  // Compute the minimum of the distances to the farthest point on the
  // closer faces.
  _T bound = std::numeric_limits<_T>::max();
  _T d2;
  for (std::size_t n = 0; n != N; ++n) {
    d2 = farthest - hi[n] + lo[n];
    if (d2 < bound) {
      bound = d2;
    }
  }
  return bound;
}


//
// N-D
//

template<std::size_t N, typename T>
inline
T
computeUpperBoundOnSignedDistance
(const BBox<T, N>& box,
 const typename BBox<T, N>::Point& x,
 std::true_type /*General case*/)
{
  typedef typename BBox<T, N>::Point Point;
  Point p;
  // The mid-point.
  Point mp;

#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif

  // Compute the mid point.
  mp = box.lower;
  mp += box.upper;
  mp *= 0.5;

  // Determine the closest vertex of the box.
  for (std::size_t n = 0; n != N; ++n) {
    if (x[n] < mp[n]) {
      p[n] = box.lower[n];
    }
    else {
      p[n] = box.upper[n];
    }
  }

  // Examine the N neighbors of the closest vertex to find the second
  // closest vertex.
  T d = std::numeric_limits<T>::max();
  T t;
  for (std::size_t n = 0; n != N; ++n) {
    if (p[n] == box.upper[n]) {
      p[n] = box.lower[n];
      t = ext::euclideanDistance(x, p);
      if (t < d) {
        d = t;
      }
      p[n] = box.upper[n];
    }
    else {
      p[n] = box.upper[n];
      t = ext::euclideanDistance(x, p);
      if (t < d) {
        d = t;
      }
      p[n] = box.lower[n];
    }
  }
  return d;
}



template<std::size_t N, typename T>
inline
T
computeLowerBoundOnSignedDistance
(const BBox<T, N>& box,
 const typename BBox<T, N>::Point& x,
 std::true_type /*General case*/)
{
  typedef typename BBox<T, N>::Point Point;
  Point p;

#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif

  //
  // If the point is inside the box.
  //
  if (isInside(box, x)) {
    // The negative distance to the nearest wall is a lower bound on the
    // distance.
    T d = -std::numeric_limits<T>::max();
    T t;
    // For each coordinate direction.
    for (std::size_t n = 0; n != N; ++n) {
      // Check the lower wall distance.
      t = box.lower[n] - x[n];
      if (t > d) {
        d = t;
      }
      // Check the upper wall distance.
      t = x[n] - box.upper[n];
      if (t > d) {
        d = t;
      }
    }
    return d;
  }

  //
  // Else the point is outside the box.
  //

  // For each coordinate direction.
  for (std::size_t n = 0; n != N; ++n) {
    // If below the box in the x coordinate.
    if (x[n] < box.lower[n]) {
      p[n] = box.lower[n];
    }
    // If above the box in the x coordinate.
    else if (box.upper[n] < x[n]) {
      p[n] = box.upper[n];
    }
    // If in the box in the x coordinate.
    else {
      p[n] = x[n];
    }
  }
  return ext::euclideanDistance(x, p);
}







//
// 1-D
//

template<typename T>
inline
T
computeUpperBoundOnSignedDistance
(const BBox<T, 1>& box,
 const typename BBox<T, 1>::Point& x,
 const std::false_type /*Special case*/)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif
  if (x[0] <= box.lower[0]) {
    return box.lower[0] - x[0];
  }
  if (x[0] >= box.upper[0]) {
    return x[0] - box.upper[0];
  }
  return std::min(box.upper[0] - x[0],
                  x[0] - box.lower[0]);
}


template<typename T>
inline
T
computeLowerBoundOnSignedDistance
(const BBox<T, 1>& box,
 const typename BBox<T, 1>::Point& x,
 const std::false_type /*Special case*/)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif
  if (x[0] <= box.lower[0]) {
    return box.lower[0] - x[0];
  }
  if (x[0] >= box.upper[0]) {
    return x[0] - box.upper[0];
  }
  return std::max(x[0] - box.upper[0],
                  box.lower[0] - x[0]);
}






//
// N-D
//

// CONTINUE: These are not declared in the header.
// Are they used or tested?
template<std::size_t N, typename T>
inline
T
computeUpperBoundOnUnsignedDistance(const BBox<T, N>& box,
                                    const typename BBox<T, N>::Point& x)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif

  // Determine the closest face.
  std::size_t min_n = 0;
  T min_d = std::numeric_limits<T>::max();
  T d;
  for (std::size_t n = 0; n != N; ++n) {
    d = std::min(std::abs(x[n] - box.upper[n]),
                 std::abs(x[n] - box.lower[n]));
    if (d < min_d) {
      min_n = n;
      min_d = d;
    }
  }

  // Compute the distance to the farthest point on the closest face.
  T dist = d * d;
  for (std::size_t n = 0; n != N; ++n) {
    if (n != min_n) {
      d = std::max(std::abs(x[n] - box.upper[n]),
                   std::abs(x[n] - box.lower[n]));
      dist += d * d;
    }
  }
  return std::sqrt(dist);
}


template<std::size_t N, typename T>
inline
T
computeLowerBoundOnUnsignedDistance(const BBox<T, N>& box,
                                    const typename BBox<T, N>::Point& x)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! isEmpty(box));
#endif

  T d;
  T dist = 0;
  for (std::size_t n = 0; n != N; ++n) {
    if (x[n] <= box.lower[n]) {
      d = box.lower[n] - x[n];
    }
    else if (x[n] >= box.upper[n]) {
      d = x[n] - box.upper[n];
    }
    else {
      d = 0;
    }
    dist += d * d;
  }
  return std::sqrt(dist);
}



//
// 1-D
//

// CONTINUE
#if 0
template<typename T>
inline
T
computeUpperBoundOnUnsignedDistance(const BBox<T, 1>& box,
                                    const typename BBox<T, 1>::Point& x)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! box.isEmpty());
#endif
  if (x[0] <= box.lower[0]) {
    return box.lower[0] - x[0];
  }
  if (x[0] >= box.upper[0]) {
    return x[0] - box.upper[0];
  }
  return std::min(box.upper[0] - x[0],
                  x[0] - box.lower[0]);
}


template<typename T>
inline
T
computeLowerBoundOnUnsignedDistance(const BBox<T, 1>& box,
                                    const typename BBox<T, 1>::Point& x)
{
#ifdef STLIB_DEBUG
  // The box should not be degenerate.
  assert(! box.isEmpty());
#endif
  if (x[0] <= box.lower[0]) {
    return box.lower[0] - x[0];
  }
  if (x[0] >= box.upper[0]) {
    return x[0] - box.upper[0];
  }
  return 0;
}
#endif

} // namespace geom
}
