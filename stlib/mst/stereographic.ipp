// -*- C++ -*-

#if !defined(__stereographic_ipp__)
#error This file is an implementation detail of stereographic.
#endif

namespace stlib
{
namespace mst
{



// Consider clipping the unit sphere by a sphere of the specified "radius".
// "distance" is the distance between the spheres' centers.  Clipping by
// the sphere is equivalent to clipping by a plane.  This function returns
// the signed distance of the clipping plane from the center of the unit
// sphere along the direction to the clipping sphere center.  See the file
// doc/Stereographic.nb for details.
inline
Number
computeClippingPlanePosition(const Number radius, const Number distance)
{
  // We us the notation in Stereographic.nb.
  // a = 1;
  const Number b = radius;
  const Number c = distance;
  const Number infinity = std::numeric_limits<Number>::max();

  // If the balls are disjoint, or meet at a point.
  if (1.0 + b <= c) {
    // The clipping has no effect.
    return infinity;
  }

  // If the unit ball contains the clipping ball.
  if (1.0 - b >= c) {
    // The clipping has no effect.
    return infinity;
  }

  // If the clipping ball contains the unit ball.
  if (b - 1.0 >= c) {
    // The clipping erases the unit ball.
    return - infinity;
  }

  assert(c != 0);
  return (1.0 - b * b + c * c) / (2.0 * c);
}



// Return the index of the best clipping ball.
inline
int
determineBestClippingBall(const std::vector<Ball>& clippingBalls)
{
  assert(clippingBalls.size() != 0);

  // Initialize with the first ball.
  std::size_t index = 0;
  Number position = computeClippingPlanePosition(clippingBalls.front());

  // Examine the rest.
  const std::size_t i_end = clippingBalls.size();
  Number p;
  for (std::size_t i = 1; i != i_end; ++i) {
    p = computeClippingPlanePosition(clippingBalls[i]);
    if (p < position) {
      index = i;
      position = p;
    }
  }

  return index;
}


inline
void
normalize(const Ball& ball,
          std::vector<Ball>& clippingBalls,
          Point& rotation)
{
  // First handle the trivial case that there are no clipping balls.
  if (clippingBalls.size() == 0) {
    // Identity rotation of the z axis.
    rotation[0] = 0;
    rotation[1] = 0;
    rotation[2] = 1;
    return;
  }

  // The translation and scaling of the clipping balls.
  const Point translation = - ball.getCenter();
  assert(ball.getRadius() != 0);
  const Number scalingFactor = 1.0 / ball.getRadius();

  // Translate and scale the clipping balls to correspond to a unit ball
  // at the origin.
  for (std::vector<Ball>::iterator i = clippingBalls.begin();
       i != clippingBalls.end(); ++i) {
    i->translate(translation);
    i->scale(scalingFactor);
  }

  // Determine the best clipping ball.
  const std::size_t best_index = determineBestClippingBall(clippingBalls);

  // Put the best clipping ball first.
  std::swap(clippingBalls[0], clippingBalls[best_index]);


}


} // namespace mst
}
