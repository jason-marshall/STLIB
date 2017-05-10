// -*- C++ -*-

#if !defined(__contact_pinballContact_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace contact
{

template<typename T, std::size_t N>
inline
T
computeMidpointRadius(const std::array < std::array<T, N>, N + 1 > & simplex,
                      const std::array<T, N>& centroid)
{
  std::array<T, N> x;
  T squaredRadius = 0;
  for (std::size_t i = 0; i != simplex.size() - 1; ++i) {
    for (std::size_t j = i + 1; j != simplex.size(); ++j) {
      // The midpoint.
      x = simplex[i];
      x += simplex[j];
      x *= 0.5;
      const T d = ext::squaredDistance(centroid, x);
      if (d > squaredRadius) {
        squaredRadius = d;
      }
    }
  }
  return std::sqrt(squaredRadius);
}

// Special case for 1-D.
template<typename T>
inline
T
computeMidpointRadius(const std::array<std::array<T, 1>, 2>& simplex,
                      const std::array<T, 1>& centroid)
{
  return std::abs(centroid[0] - simplex[0][0]);
}

// Report restoring forces for contact using the pinball method.
template<std::size_t N, typename T>
template < typename _ForceOutputIterator, typename _CountOutputIterator,
           typename _RelPenOutputIterator >
inline
T
PinballRestoringForces<N, T>::
operator()(const std::vector<std::array<T, N> >& vertices,
           const std::vector<std::size_t>& vertexIdentifiers,
           const std::vector<std::array<T, N> >& velocities,
           const std::vector < std::array < std::size_t, N + 1 > > &
           identifierSimplices,
           const std::vector<T>& masses,
           _ForceOutputIterator elementForces,
           _CountOutputIterator interactionCounts,
           _RelPenOutputIterator relativePenetrations)
{
  typedef geom::IndSimpSetIncAdj<N, N> Mesh;
  typedef typename Mesh::Number Number;
  typedef std::array<Number, N> Point;
  typedef typename HashTable::value_type HashTableValue;

  assert(vertices.size() == vertexIdentifiers.size());
  assert(vertices.size() == velocities.size());
  assert(identifierSimplices.size() == masses.size());

  // Make a mesh.
  Mesh mesh;
  build(&mesh, vertices, vertexIdentifiers, identifierSimplices);

  // Get the penetrations in terms of the simplex indices.
  std::vector<std::tuple<std::size_t, std::size_t, Point, Number> >
  penetrations;
  const Number stableTimeStep =
    pinballContact(mesh, velocities, _maximumRelativePenetration,
                   std::back_inserter(penetrations));

  // If there are no penetrations, do nothing.
  if (penetrations.empty()) {
    return stableTimeStep;
  }

  // Make a copy of the spring constants.
  // CONTINUE: Using the copy constructor causes a compilation error.
  //HashTable savedSpringConstants(_springAndDampingConstants);
  HashTable savedSpringConstants(_springAndDampingConstants.begin(),
                                 _springAndDampingConstants.end());
  _springAndDampingConstants.clear();

  // The forces to apply at the element centroids.
  std::vector<Point> forces(mesh.indexedSimplices.size(),
                            ext::filled_array<Point>(0.));
  // The number of contact interactions for each element.
  std::vector<std::size_t> interactions(mesh.indexedSimplices.size(), 0);
  Point v, force;
  Number forceMagnitude;
  // The indices of the source and target simplices. (Simplex Index)
  std::array<std::size_t, 2> si;
  // The masses of the balls.
  std::array<Number, 2> bm;
  // The normal component of the velocity.
  std::array<Number, 2> nv;
  Number springConstant, dampingConstant;
  // For each contact.
  for (std::size_t n = 0; n != penetrations.size(); ++n) {
    // The source simplex.
    si[0] = std::get<0>(penetrations[n]);
    ++interactions[si[0]];
    // The target simplex.
    si[1] = std::get<1>(penetrations[n]);
    ++interactions[si[1]];
    // The spring displacement.
    const Number displacement = ext::magnitude(std::get<2>(penetrations[n]));
    Point direction = std::get<2>(penetrations[n]);
    ext::normalize(&direction);

    // The sum of the ball radii.
    const Number sumOfRadii = std::get<3>(penetrations[n]);
    // Record the relative penetration.
    *relativePenetrations++ = displacement / sumOfRadii;

    typename HashTable::iterator springConstantIterator =
      savedSpringConstants.find(si);
    // If this is a new contact.
    if (springConstantIterator == savedSpringConstants.end()) {
      // Compute the spring constant.
      // For the source and target simplex.
      for (std::size_t i = 0; i != 2; ++i) {
        // The mass of the ball.
        bm[i] = masses[si[i]];

        // The velocity of the ball is the average of the node velocities.
        std::fill(v.begin(), v.end(), 0);
        for (std::size_t m = 0; m != Mesh::M + 1; ++m) {
          v += velocities[mesh.indexedSimplices[si[i]][m]];
        }
        v *= 1. / (Mesh::M + 1);
        // Component along penetration vector.
        nv[i] = ext::dot(v, direction);
      }

      // The maximum spring displacement.
      const Number maximumDisplacement = _maximumRelativePenetration *
                                         sumOfRadii;
      // The spring constant is determined by equating the combined kinetic
      // energy with the potential energy in the spring.
      springConstant = (bm[0] * nv[0] * nv[0] + bm[1] * nv[1] * nv[1]) /
                       (maximumDisplacement * maximumDisplacement);
      dampingConstant = 0;
      if (_dampingFraction != 0) {
        dampingConstant = std::abs(bm[1] * nv[1] - bm[0] * nv[0]) /
                          maximumDisplacement;
      }
      // Save the spring constant and the damping constant for next time.
      _springAndDampingConstants.insert
      (HashTableValue(si, std::make_pair(springConstant, dampingConstant)));
    }
    // This is an old contact.
    else {
      // Use the saved values.
      springConstant = springConstantIterator->second.first;
      dampingConstant = springConstantIterator->second.second;
      // Save the spring constant for next time.
      _springAndDampingConstants.insert(*springConstantIterator);
    }

    // Apply forces to the source and target simplex.
    // The spring force.
    forceMagnitude = _springFraction * springConstant * displacement;
    if (_dampingFraction != 0) {
      // For the source and target simplex.
      for (std::size_t i = 0; i != 2; ++i) {
        // The velocity of the ball is the average of the node velocities.
        std::fill(v.begin(), v.end(), 0);
        for (std::size_t m = 0; m != Mesh::M + 1; ++m) {
          v += velocities[mesh.indexedSimplices[si[i]][m]];
        }
        v *= 1. / (Mesh::M + 1);
        // Component along the penetration vector.
        nv[i] = ext::dot(v, direction);
      }
      // The damping force.
      forceMagnitude += _dampingFraction * dampingConstant * (nv[1] - nv[0]);
    }
    force = direction;
    force *= forceMagnitude;
    forces[si[0]] += force;
    forces[si[1]] -= force;
  }
  // Report the non-zero forces at the elements.
  const Point zero = ext::filled_array<Point>(0.);
  std::tuple<std::size_t, Point> report;
  for (std::size_t i = 0; i != forces.size(); ++i) {
    if (forces[i] != zero) {
      std::get<0>(report) = i;
      std::get<1>(report) = forces[i];
      *elementForces++ = report;
      *interactionCounts++ = interactions[i];
    }
  }
  return stableTimeStep;
}

// Report contact using the pinball method.
template<std::size_t N, typename T, typename OutputIterator>
inline
T
pinballContact(const std::vector<std::array<T, N> >& vertices,
               const std::vector<std::size_t>& vertexIdentifiers,
               const std::vector<std::array<T, N> >& velocities,
               const std::vector < std::array < std::size_t, N + 1 > > & simplices,
               const T maximumRelativePenetration,
               OutputIterator contacts)
{
  typedef geom::IndSimpSetIncAdj<N, N> Mesh;
  // Make a mesh.
  Mesh mesh;
  build(&mesh, vertices, vertexIdentifiers, simplices);
  // Get the penetrations in terms of the simplex indices.
  return pinballContact(mesh, velocities, maximumRelativePenetration,
                        contacts);
}

// Report contact using the pinball method.
template<std::size_t N, typename T, typename OutputIterator>
inline
T
pinballContact(const geom::IndSimpSetIncAdj<N, N, T>& mesh,
               const std::vector<std::array<T, N> >& velocities,
               const T maximumRelativePenetration,
               OutputIterator contacts)
{
  typedef geom::IndSimpSetIncAdj<N, N, T> Mesh;
  typedef typename Mesh::Simplex Simplex;
  typedef typename Mesh::Number Number;
  typedef std::array<Number, N> Point;
  typedef typename std::vector<Point>::iterator Record;
  typedef geom::CellArrayStatic<N, ads::Dereference<Record> > Orq;
  typedef typename Orq::BBox BBox;

  // Label the mesh components and count the number of components.
  std::vector<std::size_t> components;
  const std::size_t numComponents = labelComponents(mesh, &components);

  // No need to check for contact if there are less than 2 components.
  if (numComponents < 2) {
    // If there is no potential contact, there is no limit on the stable time
    // step.
    return std::numeric_limits<Number>::max();
  }

  // Compute the centroids and radii of the spheres.
  std::vector<Point> centroids(mesh.indexedSimplices.size());
  std::vector<Number> radii(mesh.indexedSimplices.size());
  {
    Simplex s;
    //std::array<Number, Mesh::M+1> r;
    for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      mesh.getSimplex(i, &s);
      geom::computeCentroid(s, &centroids[i]);
#if 0
      // Radius of the equivalent content ball.
      radii[i] = computeRadiusFromSimplex(s);
#endif
#if 0
      // Radius of the ball that contains all vertices.
      for (std::size_t m = 0; m != r.size(); ++m) {
        r[m] = squaredDistance(centroids[i], s[m]);
      }
      radii[i] = std::sqrt(max(r));
#endif
      // Radius of the ball that contains all midpoint nodes.
      radii[i] = computeMidpointRadius(s, centroids[i]);
    }
  }

  std::vector<Point> centroidsInComponent;
  std::vector<std::size_t> indices;
  BBox bb;
  Point p;
  std::tuple<std::size_t, std::size_t, Point, Number> penetration;
  std::vector<Record> potential;
  Number offset;
  // For each connected component.
  for (std::size_t component = 0; component != numComponents; ++component) {

    // Get the centroids in this component.
    centroidsInComponent.clear();
    indices.clear();
    for (std::size_t i = 0; i != centroids.size(); ++i) {
      if (components[i] == component) {
        centroidsInComponent.push_back(centroids[i]);
        indices.push_back(i);
      }
    }
    // Build an ORQ data structure for the centroids.
    Orq orq(centroidsInComponent.begin(), centroidsInComponent.end());

    // Search for contacts among the simplices not in the current component.
    for (std::size_t i = 0; i != centroids.size(); ++i) {
      if (components[i] == component) {
        continue;
      }
      // Make a bounding box that contains all possible contacts.
      offset = radii[i] * (2. + 10. * std::numeric_limits<Number>::epsilon());
      p = centroids[i];
      p -= offset;
      bb.lower = p;
      p = centroids[i];
      p += offset;
      bb.upper = p;
      // Get the potential contacts.
      potential.clear();
      orq.computeWindowQuery(std::back_inserter(potential), bb);
      // For each potential contact.
      for (std::size_t n = 0; n != potential.size(); ++n) {
        // The simplex index of the potential contact.
        const std::size_t j =
          indices[std::distance(centroidsInComponent.begin(), potential[n])];
        // In reporting contacts, either the first radius is greater than the
        // second or the radii are equal and the first index is less than the
        // second.
        if (!(radii[i] > radii[j] || (radii[i] == radii[j] && i < j))) {
          continue;
        }
        // The squared distance between the centroids.
        const Number sd = ext::squaredDistance(centroids[i], centroids[j]);
        const Number radiiSum = radii[i] + radii[j];
        // If the balls are in contact.
        if (sd < radiiSum * radiiSum) {
          // The penetration vector.
          p = centroids[i];
          p -= centroids[j];
          ext::normalize(&p);
          p *= radiiSum - std::sqrt(sd);
          // Report the penetration.
          std::get<0>(penetration) = i;
          std::get<1>(penetration) = j;
          std::get<2>(penetration) = p;
          std::get<3>(penetration) = radiiSum;
          *contacts++ = penetration;
        }
      }
    }
  }

  //
  // Compute the stable time step. Check for the worst case scenario of each
  // element colliding head-on with a mirror image of itself.
  //
  Number stableTimeStep = std::numeric_limits<Number>::max();
  Point v;
  // For each element in the mesh.
  for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
    // The velocity of the ball is the average of the node velocities.
    std::fill(v.begin(), v.end(), 0);
    for (std::size_t m = 0; m != Mesh::M + 1; ++m) {
      v += velocities[mesh.indexedSimplices[i][m]];
    }
    v *= 1. / (Mesh::M + 1);
    const Number speed = ext::magnitude(v);
    // The factor of 1/2 is because the mirror image is moving in the
    // opposite direction.
    const Number d = 0.5 * maximumRelativePenetration * radii[i];
    // Ensure we don't divide by zero.
    const Number t = d / (speed + d * std::numeric_limits<Number>::epsilon());
    if (t < stableTimeStep) {
      stableTimeStep = t;
    }
  }
  return stableTimeStep;
}

#if 0
// CONTINUE: Not currently used.
template<typename T>
inline
T
computeRadiusFromContent(const T content,
                         std::integral_constant<std::size_t, 1> /*Dimension*/)
{
  return 0.5 * content;
}

template<typename T>
inline
T
computeRadiusFromContent(const T content,
                         std::integral_constant<std::size_t, 2> /*Dimension*/)
{
  return std::sqrt((1. / numerical::Constants<T>::Pi()) * content);
}

template<typename T>
inline
T
computeRadiusFromContent(const T content,
                         std::integral_constant<std::size_t, 3> /*Dimension*/)
{
  return std::pow(3. / (4. * numerical::Constants<T>::Pi()) * content, 1. / 3.);
}

template<typename T>
inline
T
computeRadiusFromSimplex
(const std::array<std::array<T, 1>, 2>& simplex)
{
  // c = 2 r
  // r = c / 2
  return 0.5 * computeContent(simplex);
}

template<typename T>
inline
T
computeRadiusFromSimplex
(const std::array<std::array<T, 2>, 3>& simplex)
{
  // c = pi r^2
  // r = (c / pi)^(1/2)
  return std::sqrt((1. / numerical::Constants<T>::Pi()) *
                   computeContent(simplex));
}

template<typename T>
inline
T
computeRadiusFromSimplex
(const std::array<std::array<T, 3>, 4>& simplex)
{
  // c = (4/3) pi r^3
  // r = (3 c / (4 pi))^(1/3)
  return std::pow(3. / (4. * numerical::Constants<T>::Pi()) *
                  computeContent(simplex), 1. / 3.);
}

template<std::size_t N, typename T>
struct
    InverseTotalAngle;

template<typename T>
struct
    InverseTotalAngle<1, T> {
  static
  T
  value()
  {
    return 0.5;
  }
};

template<typename T>
struct
    InverseTotalAngle<2, T> {
  static
  T
  value()
  {
    return 0.5 / numerical::Constants<T>::Pi();
  }
};

template<typename T>
struct
    InverseTotalAngle<3, T> {
  static
  T
  value()
  {
    return 0.25 / numerical::Constants<T>::Pi();
  }
};

#endif

} // namespace contact
}
