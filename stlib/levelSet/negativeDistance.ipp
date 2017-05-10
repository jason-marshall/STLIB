// -*- C++ -*-

#if !defined(__levelSet_negativeDistance_ipp__)
#error This file is an implementation detail of negativeDistance.
#endif

namespace stlib
{
namespace levelSet
{


/*
  This function is used in negativeDistance(). It is not a
  general purpose function.  If the point is outside the specified
  ball, return infinity to indicate some unknown positive distance.
  If the point is inside the specified ball and the closest point on
  that ball is not inside any of the intersecting balls, return the
  distance.  Otherwise return negative infinity to indicate that the
  point is some unknown negative distance.
*/
template<typename _T, std::size_t _D>
inline
_T
distance(const std::vector<geom::Ball<_T, _D> >& balls,
         const std::size_t n, const std::vector<std::size_t>& intersecting,
         const std::array<_T, _D>& x)
{
  std::array<_T, _D> closest;
  const _T d = closestPoint(balls[n], x, &closest);
  if (d > 0) {
    return std::numeric_limits<_T>::infinity();
  }
  for (std::size_t i = 0; i != intersecting.size(); ++i) {
    if (isInside(balls[intersecting[i]], closest)) {
      return -std::numeric_limits<_T>::infinity();
    }
  }
  return d;
}


/*
  This function is used in negativeDistance(). It is not a
  general purpose function.  If the point is outside the specified
  ball, return infinity to indicate some unknown positive distance.
  If the point is inside the specified ball and the closest point on
  that ball is not inside any of the other balls, return the
  distance.  Otherwise return negative infinity to indicate that the
  point is some unknown negative distance.
*/
template<typename _T, std::size_t _D>
inline
_T
negativeDistance(const std::vector<geom::Ball<_T, _D> >& balls,
                 const std::size_t n, const std::array<_T, _D>& x)
{
  std::array<_T, _D> closest;
  const _T d = closestPoint(balls[n], x, &closest);
  if (d > 0) {
    return std::numeric_limits<_T>::infinity();
  }
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i != n && isInside(balls[i], closest)) {
      return -std::numeric_limits<_T>::infinity();
    }
  }
  return d;
}


template<typename _T>
inline
void
negativeDistance(container::SimpleMultiArrayRef<_T, 2>* grid,
                 const geom::BBox<_T, 2>& domain,
                 const std::vector<geom::Ball<_T, 2> >& balls)
{
  const std::size_t D = 2;

  typedef container::SimpleMultiArrayRef<_T, D> SimpleMultiArray;
  typedef typename SimpleMultiArray::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Initialize the grid by setting all of the distances to NaN.
  std::fill(grid->begin(), grid->end(), std::numeric_limits<_T>::quiet_NaN());

  geom::SimpleRegularGrid<_T, D> regularGrid(grid->extents(), domain);
  std::vector<std::size_t> intersecting;
  geom::BBox<_T, D> window;
  Range range;
  std::array<_T, D> p;
  _T d;
  // For each ball.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Get the intersecting balls.
    getIntersecting(balls, i, &intersecting);
    // Build a bounding box around the ball.
    window = geom::specificBBox<geom::BBox<_T, D> >(balls[i]);
    // Compute the index range for the bounding box.
    range = regularGrid.computeRange(window);
    // For each index in the range.
    const Iterator end = Iterator::end(range);
    for (Iterator index = Iterator::begin(range); index != end; ++index) {
      // The grid value.
      _T& g = (*grid)(*index);
      // Convert the index to a Cartesian location.
      p = regularGrid.indexToLocation(*index);
      // Compute the signed distance to the surface of the ball.
      d = distance(balls, i, intersecting, p);
      // If the computed distance is negative and smaller in magnitude
      // than the current value, store the distance.
      if (d < 0 && (g != g || d > g)) {
        g = d;
      }
    }
  }

  // For each intersection of balls.
  std::array<IntersectionPoint<_T, D>, 2> pointsOnSurface;
  geom::Ball<_T, 2> pointBall;
  std::array<_T, D> v;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      if (! makeBoundaryIntersection(balls[i], balls[j], &pointsOnSurface[0],
                                     &pointsOnSurface[1])) {
        continue;
      }
      // Compute distances for each of the two points on the surface.
      for (std::size_t k = 0; k != pointsOnSurface.size(); ++k) {
        const IntersectionPoint<_T, D>& pos = pointsOnSurface[k];
        // Skip if this point is not on the surface.
        if (! isOnSurface(balls, i, j, pos.location)) {
          continue;
        }
        // Get the balls that may contain the points with negative distance
        // up to the larger of the radii.
        pointBall.center = pos.location;
        pointBall.radius = pos.radius;
        getIntersecting(balls, pointBall, &intersecting);
        // Make the bounding box that contains the points with negative
        // distance.
        boundNegativeDistance(pos, &window);
        // Compute the index range for the bounding box.
        range = regularGrid.computeRange(window);
        // For each index in the range.
        const Iterator end = Iterator::end(range);
        for (Iterator index = Iterator::begin(range); index != end;
             ++index) {
          // The grid value.
          _T& g = (*grid)(*index);
          // Convert the index to a Cartesian location.
          p = regularGrid.indexToLocation(*index);
          // Of the two following tests, the former is less expensive.
          // Thus it is evaluated first.
          // Ignore points with positive distance.
          v = p;
          v -= pos.location;
          if (ext::dot(v, pos.normal) > 0) {
            continue;
          }
          // Ignore points that are not inside the union of balls.
          if (! isInside(balls, intersecting, p)) {
            continue;
          }
          // Compute the signed distance to the point on the surface.
          d = - ext::euclideanDistance(p, pos.location);
          // If the computed distance is smaller in magnitude than
          // the current value, store the distance.
          if (g != g || d > g) {
            g = d;
          }
        }
      }
    }
  }
}


template<typename _T, std::size_t _D>
inline
void
updateDistance(const std::vector<geom::Ball<_T, _D> >& balls,
               const std::size_t n, std::array<_T, _D> p,
               _T* distance)
{
  const geom::Ball<_T, _D>& ball = balls[n];
  // Translate the center to the origin.
  p -= ball.center;
  // The squared distance from the center.
  const _T r2 = ext::squaredMagnitude(p);
  // If the point is not inside the ball, return.
  if (r2 >= ball.radius * ball.radius) {
    return;
  }
  // The distance from the center.
  const _T r = std::sqrt(r2);
  // The signed distance.
  const _T d = r - ball.radius;
  // If this distance is less than the current, return.
  if (d <= *distance) {
    return;
  }
  // Move to the surface.
  // Special case that the point is at the center.
  if (r < std::numeric_limits<_T>::epsilon()) {
    // Pick an arbitrary closest point.
    std::fill(p.begin(), p.end(), _T(0));
    p[0] = ball.radius;
  }
  else {
    p *= (ball.radius / r);
  }
  // Translate back to the ball.
  p += ball.center;
  // If the closest point is inside another ball, return.
  if (isInside(balls, n, p)) {
    return;
  }
  // Update the distance.
  *distance = d;
}


// Negative distance for a union of balls.
// This function differs from the one for multi-arrays in that we don't use
// scan conversion. We loop over grid points and geometric entities. This is
// efficient because the patches are small.
template<typename _T, std::size_t N, typename _Base>
inline
void
negativeDistance(container::EquilateralArrayImp<_T, 2, N, _Base>* patch,
                 const std::array<_T, 2>& lowerCorner,
                 const _T spacing,
                 const std::vector<geom::Ball<_T, 2> >& balls,
                 const std::vector<IntersectionPoint<_T, 2> >& points)
{
  const std::size_t D = 2;
  const _T Inf = std::numeric_limits<_T>::infinity();
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Initialize to positive infinity.
  std::fill(patch->begin(), patch->end(), Inf);

  std::array<_T, D> p;
  _T d;

  // For each grid point.
  const Iterator end = Iterator::end(patch->extents());
  for (Iterator i = Iterator::begin(patch->extents()); i != end; ++i) {
    _T& g = (*patch)(*i);
    p = lowerCorner + spacing * stlib::ext::convert_array<_T>(*i);
    //
    // Ignore points that are not inside the union of balls.
    //
    for (std::size_t n = 0; n != balls.size(); ++n) {
      if (isInside(balls[n], p)) {
        g = - Inf;
        break;
      }
    }
    if (g >= 0) {
      continue;
    }
    //
    // For each point on the surface. Note that we work with the squared
    // distance.
    //
    g = Inf;
    for (std::size_t n = 0; n != points.size(); ++n) {
      const IntersectionPoint<_T, D>& pos = points[n];
      // Compute the squared distance to the point on the surface.
      d = ext::squaredDistance(p, pos.location);
      if (d < g) {
        g = d;
      }
    }
    g = - std::sqrt(g);
    //
    // For each ball.
    //
    for (std::size_t n = 0; n != balls.size(); ++n) {
      updateDistance(balls, n, p, &g);
    }
  }
}


// Construct the intersection points and then call another function to
// compute the distance using the balls and the intersection points.
template<typename _T, std::size_t N, typename _Base>
inline
void
negativeDistance(container::EquilateralArrayImp<_T, 2, N, _Base>* patch,
                 const std::array<_T, 2>& lowerCorner,
                 const _T spacing,
                 const std::vector<geom::Ball<_T, 2> >& balls)
{
  const std::size_t D = 2;

  // Calculate the intersection points.
  std::vector<IntersectionPoint<_T, D> > intersectionPoints;
  // For each intersection of balls.
  IntersectionPoint<_T, D> a, b;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      if (! makeBoundaryIntersection(balls[i], balls[j], &a, &b)) {
        continue;
      }
      // For each of the two intersection points, record the point if
      // it is on the surface.
      if (isOnSurface(balls, i, j, a.location)) {
        intersectionPoints.push_back(a);
      }
      if (isOnSurface(balls, i, j, b.location)) {
        intersectionPoints.push_back(b);
      }
    }
  }

  // Compute the distance.
  negativeDistance(patch, lowerCorner, spacing, balls, intersectionPoints);
}


template<typename _T>
inline
void
negativeDistance(container::SimpleMultiArrayRef<_T, 3>* grid,
                 const geom::BBox<_T, 3>& domain,
                 const std::vector<geom::Ball<_T, 3> >& balls)
{
  const std::size_t D = 3;

  typedef container::SimpleMultiArrayRef<_T, D> SimpleMultiArray;
  typedef typename SimpleMultiArray::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Initialize the grid by setting all of the distances to NaN.
  std::fill(grid->begin(), grid->end(), std::numeric_limits<_T>::quiet_NaN());

  geom::SimpleRegularGrid<_T, D> regularGrid(grid->extents(), domain);
  std::vector<std::size_t> intersecting;
  geom::BBox<_T, D> window;
  Range range;
  std::array<_T, D> p;
  _T d;
  // For each ball.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    // Get the intersecting balls.
    getIntersecting(balls, i, &intersecting);
    // Build a bounding box around the ball.
    window = geom::specificBBox<geom::BBox<_T, D> >(balls[i]);
    // Compute the index range for the bounding box.
    range = regularGrid.computeRange(window);
    // For each index in the range.
    const Iterator end = Iterator::end(range);
    for (Iterator index = Iterator::begin(range); index != end; ++index) {
      // The grid value.
      _T& g = (*grid)(*index);
      // Convert the index to a Cartesian location.
      p = regularGrid.indexToLocation(*index);
      // Compute the signed distance to the surface of the ball.
      d = distance(balls, i, intersecting, p);
      // If the computed distance is negative and smaller in magnitude
      // than the current value, store the distance.
      if (d < 0 && (g != g || d > g)) {
        g = d;
      }
    }
  }

  // For each intersection of balls.
  geom::Circle3<_T> circle;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      if (! makeBoundaryIntersection(balls[i], balls[j], &circle)) {
        continue;
      }
      //
      // Compute distances for the circle.
      //
      // Get the intersecting balls.
      getIntersecting(balls, i, j, &intersecting);
      // Make the bounding box that contains the points with negative
      // distance.
      boundNegativeDistance(balls[i], balls[j], circle, &window);
      // Compute the index range for the bounding box.
      range = regularGrid.computeRange(window);
      // For each index in the range.
      const Iterator end = Iterator::end(range);
      for (Iterator index = Iterator::begin(range); index != end;
           ++index) {
        // The grid value.
        _T& g = (*grid)(*index);
        // Convert the index to a Cartesian location.
        p = regularGrid.indexToLocation(*index);
        // Ignore points that are not inside one of the two balls.
        if (!(isInside(balls[i], p) || isInside(balls[j], p))) {
          continue;
        }
        // Compute the signed distance to the circle.
        d = distance(circle, p, balls, intersecting);
        // Ignore points with positive distance.
        if (d > 0) {
          continue;
        }
        // If the computed distance is smaller in magnitude than
        // the current value, store the distance.
        if (g != g || d > g) {
          g = d;
        }
      }
    }
  }

  // For each intersection of three balls.
  std::array<IntersectionPoint<_T, D>, 2> pointsOnSurface;
  geom::Ball<_T, D> pointBall;
  std::array<_T, D> v;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      for (std::size_t k = j + 1; k != balls.size(); ++k) {
        if (! makeBoundaryIntersection(balls[i], balls[j], balls[k],
                                       &pointsOnSurface[0],
                                       &pointsOnSurface[1])) {
          continue;
        }
        // Compute distances for each of the two points on the surface.
        for (std::size_t m = 0; m != pointsOnSurface.size(); ++m) {
          const IntersectionPoint<_T, D>& pos = pointsOnSurface[m];
          // Skip if this point is not on the surface.
          if (! isOnSurface(balls, i, j, k, pos.location)) {
            continue;
          }
          // Get the balls that may contain the points with negative
          // distance up to the larger of the radii.
          pointBall.center = pos.location;
          pointBall.radius = pos.radius;
          getIntersecting(balls, pointBall, &intersecting);
          // Make the bounding box that contains the points with negative
          // distance.
          boundNegativeDistance(pos, &window);
          // Compute the index range for the bounding box.
          range = regularGrid.computeRange(window);
          // For each index in the range.
          const _T squaredRadius = pos.radius * pos.radius;
          const Iterator end = Iterator::end(range);
          for (Iterator index = Iterator::begin(range); index != end;
               ++index) {
            // The grid value.
            _T& g = (*grid)(*index);
            // Convert the index to a Cartesian location.
            p = regularGrid.indexToLocation(*index);
            // Of the three following tests, the first two are
            // less expensive.  Thus they are evaluated first.
            // Ignore points that are too far away. We only correctly
            // compute distance up to the radius defined in pos.
            const _T sd = ext::squaredDistance(p, pos.location);
            if (sd > squaredRadius) {
              continue;
            }
            // Ignore points with positive distance.
            v = p;
            v -= pos.location;
            if (ext::dot(v, pos.normal) > 0) {
              continue;
            }
            // CONTINUE: This is fairly expensive. 6.6% of execution time.
            // Ignore points that are not inside the union of balls.
            if (! isInside(balls, intersecting, p)) {
              continue;
            }
            // Compute the signed distance to the point on the surface.
            //d = - euclideanDistance(p, pos.location);
            d = - std::sqrt(sd);
            // If the computed distance is smaller in magnitude than
            // the current value, store the distance.
            if (g != g || d > g) {
              g = d;
            }
          }
        }
      }
    }
  }
}


template<typename _T>
inline
void
updateDistance(const std::vector<geom::Ball<_T, 3> >& balls,
               const geom::Circle3<_T>& circle,
               const std::pair<std::size_t, std::size_t>& ip,
               const std::array<_T, 3>& x, _T* squaredDist)
{
  // Ignore points that are not inside one of the two balls.
  if (!(isInside(balls[ip.first], x) ||
        isInside(balls[ip.second], x))) {
    return;
  }


  // Let c be the circle center and r its radius.
  // Let a be the (signed) distance to the supporting plane of the
  // circle. Let b be the distance from the projection of the point
  // on the plane to the circle center.
  // |x - c|^2 = a^2 + b^2
  // b = sqrt(|x - circle.center|^2 - a^2)
  // The unsigned distance to the circle is d.
  // d^2 = a^2 + (b-r)^2
  // The signed distance is positive iff b > r.
  const _T sa = ext::dot(x - circle.center, circle.normal);
  const _T a2 = sa * sa;
  const _T b2 = ext::squaredDistance(x, circle.center) - a2;
  // If the signed distance is positive, return.
  if (b2 >= circle.radius * circle.radius) {
    return;
  }

  // Compute the squared distance.
  const _T b = std::sqrt(b2);
  const _T d = a2 + (b - circle.radius) * (b - circle.radius);
  // If this squared distance is greater than the current squared distance,
  // return.
  if (d >= *squaredDist) {
    return;
  }

  // Compute the closest point on the circle.
  std::array<_T, 3> cp;
  computeClosestPoint(circle, x, &cp);
  // If the closest point is inside one of the other spheres, return.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    if (i == ip.first || i == ip.second) {
      continue;
    }
    if (isInside(balls[i], cp)) {
      return;
    }
  }

  // Update the squared distance.
  *squaredDist = d;
}


// Negative distance for a union of balls.
// This function differs from the one for multi-arrays in that we don't use
// scan conversion. We loop over grid points and geometric entities. This is
// efficient because the patches are small.
template<typename _T, std::size_t N, typename _Base>
inline
void
negativeDistance
(container::EquilateralArrayImp<_T, 3, N, _Base>* patch,
 const std::array<_T, 3>& lowerCorner,
 const _T spacing,
 const std::vector<geom::Ball<_T, 3> >& balls,
 const std::vector<std::pair<std::size_t, std::size_t> >& intersectionPairs,
 const std::vector<geom::Circle3<_T> >& intersectionCircles,
 const std::vector<IntersectionPoint<_T, 3> >& intersectionPoints)
{
  const std::size_t D = 3;
  const _T Inf = std::numeric_limits<_T>::infinity();
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Initialize to positive infinity.
  std::fill(patch->begin(), patch->end(), Inf);

  std::array<_T, D> p;
  _T d;

  // For each grid point.
  const Iterator end = Iterator::end(patch->extents());
  for (Iterator i = Iterator::begin(patch->extents()); i != end; ++i) {
    _T& g = (*patch)(*i);
    p = lowerCorner + spacing * stlib::ext::convert_array<_T>(*i);
    //
    // Ignore points that are not inside the union of balls.
    //
    for (std::size_t n = 0; n != balls.size(); ++n) {
      if (isInside(balls[n], p)) {
        g = - Inf;
        break;
      }
    }
    if (g >= 0) {
      continue;
    }
    // Start working with the squared distance.
    g = Inf;
    // For each intersection point on the surface.
    for (std::size_t n = 0; n != intersectionPoints.size(); ++n) {
      const IntersectionPoint<_T, D>& pos = intersectionPoints[n];
      // Compute the squared distance to the point on the surface.
      d = ext::squaredDistance(p, pos.location);
      if (d < g) {
        g = d;
      }
    }
    // For each intersection circle.
    for (std::size_t n = 0; n != intersectionCircles.size(); ++n) {
      updateDistance(balls, intersectionCircles[n], intersectionPairs[n],
                     p, &g);
    }
    // Convert the squared distance to signed distance.
    g = - std::sqrt(g);
    // For each ball.
    for (std::size_t n = 0; n != balls.size(); ++n) {
      updateDistance(balls, n, p, &g);
    }
  }
}


// Construct the intersection circles and points and then call another function
// to compute the distance using the balls and the intersections.
template<typename _T, std::size_t N, typename _Base>
inline
void
negativeDistance(container::EquilateralArrayImp<_T, 3, N, _Base>* patch,
                 const std::array<_T, 3>& lowerCorner,
                 const _T spacing,
                 const std::vector<geom::Ball<_T, 3> >& balls)
{
  const std::size_t D = 3;

  // Calculate the intersection circles.
  std::vector<std::pair<std::size_t, std::size_t> > intersectionPairs;
  std::vector<geom::Circle3<_T> > intersectionCircles;
  geom::Circle3<_T> circle;
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      if (makeBoundaryIntersection(balls[i], balls[j], &circle)) {
        intersectionPairs.push_back(std::make_pair(i, j));
        intersectionCircles.push_back(circle);
      }
    }
  }

  // Calculate the intersection points.
  std::vector<IntersectionPoint<_T, D> > intersectionPoints;
  IntersectionPoint<_T, D> a, b;
  // For each intersection of balls.
  for (std::size_t i = 0; i != balls.size(); ++i) {
    for (std::size_t j = i + 1; j != balls.size(); ++j) {
      for (std::size_t k = j + 1; k != balls.size(); ++k) {
        // If the three balls intersect at a pair of points.
        if (makeBoundaryIntersection(balls[i], balls[j], balls[k],
                                     &a, &b)) {
          // If this point is on the surface (not contained within a ball).
          if (isOnSurface(balls, i, j, k, a.location)) {
            intersectionPoints.push_back(a);
          }
          if (isOnSurface(balls, i, j, k, b.location)) {
            intersectionPoints.push_back(b);
          }
        }
      }
    }
  }

  // Compute the distance.
  negativeDistance(patch, lowerCorner, spacing, balls, intersectionPairs,
                   intersectionCircles, intersectionPoints);
}


template<typename _T, std::size_t _D>
inline
void
negativeDistance(GridUniform<_T, _D>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls)
{
  return negativeDistance(grid, grid->domain(), balls);
}


// Compute the negative distances for a union of balls.
template<typename _T, std::size_t _D, std::size_t N>
inline
void
negativeDistance(Grid<_T, _D, N>* grid,
                 const std::vector<geom::Ball<_T, _D> >& balls)
{
  typedef container::SimpleMultiArrayRef<_T, _D> MultiArrayRef;
  typedef typename MultiArrayRef::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Determine the patch/ball dependencies.
  container::StaticArrayOfArrays<unsigned> dependencies;
  {
    // Calculate the largest radius.
    _T maxRadius = 0;
    for (std::size_t i = 0; i != balls.size(); ++i) {
      if (balls[i].radius > maxRadius) {
        maxRadius = balls[i].radius;
      }
    }
    // A ball may influence points up to maxRadius beyond its surface.
    // Consider the case that one ball barely intersects the largest ball.
    // Then points all the way up to the center of the largest ball may
    // be closest to the intersection of the two balls.
    std::vector<geom::Ball<_T, _D> > offsetBalls = balls;
    for (std::size_t i = 0; i != offsetBalls.size(); ++i) {
      offsetBalls[i].radius += maxRadius;
    }
    patchDependencies(*grid, offsetBalls.begin(), offsetBalls.end(),
                      &dependencies);
  }

  // Refine the appropriate patches and set the rest to have an unknown
  // distance.
  grid->refine(dependencies);

  // Use a multi-array to wrap the patches.
  const IndexList patchExtents = ext::filled_array<IndexList>(N);
  container::SimpleMultiArrayRef<_T, _D> patch(0, patchExtents);
  std::vector<geom::Ball<_T, _D> > influencingBalls;
  // Loop over the patches.
  const Iterator end = Iterator::end(grid->extents());
  for (Iterator i = Iterator::begin(grid->extents()); i != end; ++i) {
    const std::size_t index = grid->arrayIndex(*i);
    if (!(*grid)[index].isRefined()) {
      continue;
    }
    // Build the parameters.
    patch.rebuild((*grid)[index].data(), patchExtents);
    influencingBalls.clear();
    for (std::size_t n = 0; n != dependencies.size(index); ++n) {
      influencingBalls.push_back(balls[dependencies(index, n)]);
    }
    // Compute the distance.
    negativeDistance(&patch, grid->getVertexPatchDomain(*i),
                     influencingBalls);
  }
}


} // namespace levelSet
}
