// -*- C++ -*-

#if !defined(__contact_ProxyBallContact_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace contact
{

template<std::size_t N>
inline
std::ostream&
operator<<(std::ostream& out, const ProxyBall<N>& x)
{
  out << "center = " << x.center << '\n'
      << "velocity = " << x.velocity << '\n'
      << "mass = " << x.mass << '\n'
      << "radius = " << x.radius << '\n'
      << "component = " << x.component << '\n'
      << "processor = " << x.processor << '\n'
      << "elementIndex = " << x.elementIndex << '\n'
      << "elementIdentifier = " << x.elementIdentifier << '\n';
  return out;
}

template<std::size_t N>
inline
bool
operator==(const ProxyBall<N>& x, const ProxyBall<N>& y)
{
  return x.center == y.center &&
         x.velocity == y.velocity &&
         x.mass == y.mass &&
         x.radius == y.radius &&
         x.component == y.component &&
         x.processor == y.processor &&
         x.elementIndex == y.elementIndex &&
         x.elementIdentifier == y.elementIdentifier;
}

template<std::size_t N, typename _Identifier>
ProxyBallContact<N, _Identifier>::
ProxyBallContact(const std::size_t numberOfComponents,
                 const std::size_t numberOfNodes,
                 const Number* nodeCoordinates,
                 const Identifier* nodeIdentifiers,
                 const std::size_t numberOfElements,
                 const Identifier* identifierSimplices,
                 const Number maximumRelativePenetration,
                 const Number springFraction,
                 const Number dampingFraction) :
  _numberOfComponents(numberOfComponents),
  _numberOfNodes(numberOfNodes),
  _numberOfElements(numberOfElements),
  _indexSimplices(numberOfElements),
  _proxyBalls(numberOfElements),
  _numberOfLocalProxyBalls(numberOfElements),
  _components(numberOfComponents),
  _maximumRelativePenetration(maximumRelativePenetration),
  _springFraction(springFraction),
  _dampingFraction(dampingFraction),
  _springAndDampingConstants()
{
  // Check some parameters.
  assert(_springFraction >= 0);
  assert(_dampingFraction >= 0);
  assert(maximumRelativePenetration > 0);
  // Start with the identifier simplices.
  for (std::size_t i = 0; i != _indexSimplices.size(); ++i) {
    for (std::size_t n = 0; n != N + 1; ++n) {
      _indexSimplices[i][n] = *identifierSimplices++;
    }
  }
  // Make a mapping from node identifiers to node indices.
  std::unordered_map<std::size_t, std::size_t> identifierToIndex;
  for (std::size_t i = 0; i != _numberOfNodes; ++i) {
    identifierToIndex[nodeIdentifiers[i]] = i;
  }
  // Convert the identifier simplices to index simplices.
  for (std::size_t i = 0; i != _indexSimplices.size(); ++i) {
    for (std::size_t n = 0; n != N + 1; ++n) {
      _indexSimplices[i][n] = identifierToIndex[_indexSimplices[i][n]];
    }
  }
  // Compute the radii using the initial mesh configuration.
  computeRadii(nodeCoordinates);
  // Record the the processor index for each element.
  for (std::size_t i = 0; i != _proxyBalls.size(); ++i) {
    // Invalid value indicate an inactive element.
    _proxyBalls[i].component = _numberOfComponents;
    // Since this is the serial version the processor index is 0. Also, we
    // can use the element index for the identifier.
    _proxyBalls[i].processor = 0;
    _proxyBalls[i].elementIndex = i;
    _proxyBalls[i].elementIdentifier = i;
  }
}

template<std::size_t N, typename _Identifier>
template < typename _ForceOutputIterator, typename _CountOutputIterator,
           typename _RelPenOutputIterator >
inline
typename ProxyBallContact<N, _Identifier>::Number
ProxyBallContact<N, _Identifier>::
operator()(const Number* nodeCoordinates,
           const Number* velocityCoordinates,
           const Number* masses,
           const std::size_t* components,
           _ForceOutputIterator elementForces,
           _CountOutputIterator interactionCounts,
           _RelPenOutputIterator relativePenetrations)
{
  initializeProxyBalls(nodeCoordinates, velocityCoordinates, masses,
                       components);
  // Get the penetrations in terms of the simplex indices.
  std::vector<std::tuple<std::size_t, std::size_t, Point> > penetrations;
  computeContact(std::back_inserter(penetrations));
  computeForces(penetrations, elementForces, interactionCounts,
                relativePenetrations);
  return computeStableTimeStep(components);
}

// Write the radii of the proxy balls. Write the spring and damping constants.
template<std::size_t N, typename _Identifier>
inline
std::ostream&
operator<<(std::ostream& out, const ProxyBallContact<N, _Identifier>& x)
{
  USING_STLIB_EXT_PAIR_IO_OPERATORS;

  // The radii of the proxy balls.
  out << x._proxyBalls.size() << '\n';
  for (std::size_t i = 0; i != x._proxyBalls.size(); ++i) {
    out << x._proxyBalls[i].radius << '\n';
  }
  // The spring constants and damping constants.
  out << x._springAndDampingConstants.size() << '\n';
  for (auto i = x._springAndDampingConstants.begin();
       i != x._springAndDampingConstants.end(); ++i) {
    out << *i << '\n';
  }
  return out;
}

// Read the proxy ball radii and the spring constants.
template<std::size_t N, typename _Identifier>
inline
std::istream&
operator>>(std::istream& in, ProxyBallContact<N, _Identifier>& x)
{
  USING_STLIB_EXT_ARRAY_IO_OPERATORS;

  typedef typename ProxyBallContact<N, _Identifier>::HashTable HashTable;
  typedef typename HashTable::value_type value_type;
  typedef typename HashTable::key_type key_type;
  typedef typename HashTable::mapped_type mapped_type;

  // Read the radii.
  std::size_t size;
  in >> size;
  assert(size == x._proxyBalls.size());
  for (std::size_t i = 0; i != x._proxyBalls.size(); ++i) {
    in >> x._proxyBalls[i].radius;
  }
  // Clear the any old spring constants.
  x._springAndDampingConstants.clear();
  // Get the number of spring constants.
  in >> size;
  // Read each of them.
  key_type key;
  mapped_type mapped;
  for (size_t i = 0; i != size; ++i) {
    in >> key >> mapped;
    x._springAndDampingConstants.insert(value_type(key, mapped));
  }
  return in;
}

//! Equality.
/*! \relates ProxyBallContact */
template<std::size_t N, typename _Identifier>
inline
bool
operator==(const ProxyBallContact<N, _Identifier>& x,
           const ProxyBallContact<N, _Identifier>& y)
{
  // Don't check _components as it is set with initializeProxyBalls().
  return x._numberOfComponents == y._numberOfComponents &&
         x._numberOfNodes == y._numberOfNodes &&
         x._numberOfElements == y._numberOfElements &&
         x._indexSimplices == y._indexSimplices &&
         x._proxyBalls == y._proxyBalls &&
         x._maximumRelativePenetration == y._maximumRelativePenetration &&
         x._springFraction == y._springFraction &&
         x._dampingFraction == y._dampingFraction &&
         x._springAndDampingConstants.size() == y._springAndDampingConstants.size();
  // CONTINUE: Equality should work, but doesn't.
  // x._springAndDampingConstants == y._springAndDampingConstants;
}


template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContact<N, _Identifier>::
initializeProxyBalls(const Number* nodeCoordinates,
                     const Number* velocityCoordinates, const Number* masses,
                     const std::size_t* components)
{
  // Clear the old lists of active proxy balls.
  for (std::size_t i = 0; i != _components.size(); ++i) {
    _components[i].clear();
  }
  // For the active elements set the following fields: center, velocity,
  // mass, and component. Group the proxy balls by component.
  Simplex s;
  std::size_t index = 0;
  for (ProxyBallIterator i = _proxyBalls.begin(); i != _proxyBalls.end();
       ++i, ++index) {
    i->component = components[index];
    if (i->component < _numberOfComponents) {
      // The centroid.
      computeSimplex(&s, nodeCoordinates, index);
      geom::computeCentroid(s, &i->center);
      // The velocity.
      computeVelocity(&i->velocity, velocityCoordinates, index);
      // The mass.
      i->mass = masses[index];
      _components[i->component].push_back(i);
    }
  }
}

template<std::size_t N, typename _Identifier>
template < typename _ForceOutputIterator, typename _CountOutputIterator,
           typename _RelPenOutputIterator >
inline
void
ProxyBallContact<N, _Identifier>::
computeForces(const std::vector < std::tuple < std::size_t, std::size_t,
              Point > > penetrations,
              _ForceOutputIterator elementForces,
              _CountOutputIterator interactionCounts,
              _RelPenOutputIterator relativePenetrations)
{
  typedef typename HashTable::value_type HashTableValue;
  // If there are no penetrations, do nothing.
  if (penetrations.empty()) {
    return;
  }

  // Make a copy of the spring constants.
  // CONTINUE: Using the copy constructor causes a compilation error.
  //HashTable savedSpringConstants(_springAndDampingConstants);
  HashTable savedSpringConstants(_springAndDampingConstants.begin(),
                                 _springAndDampingConstants.end());
  _springAndDampingConstants.clear();

  // The forces to apply at the element centroids.
  std::vector<Point> forces(_proxyBalls.size(), ext::filled_array<Point>(0.));
  // The number of contact interactions for each element.
  std::vector<std::size_t> interactions(_proxyBalls.size(), 0);
  Point force;
  Number forceMagnitude;
  // The indices of the source and target simplices. (Simplex Index)
  std::array<std::size_t, 2> si;
  // The source and target proxy balls.
  std::array<ProxyBallConstIterator, 2> pb;
  // The identifiers of the source and target proxy balls.
  std::array<std::size_t, 2> id;
  // The normal component of the velocity.
  std::array<Number, 2> nv;
  Number springConstant, dampingConstant;
  // For each contact.
  for (std::size_t n = 0; n != penetrations.size(); ++n) {
    // The source proxy ball.
    si[0] = std::get<0>(penetrations[n]);
    // The target simplex.
    si[1] = std::get<1>(penetrations[n]);
    for (std::size_t i = 0; i != 2; ++i) {
      pb[i] = _proxyBalls.begin() + si[i];
      id[i] = pb[i]->elementIdentifier;
      ++interactions[si[i]];
    }
    // The spring displacement.
    Point direction = std::get<2>(penetrations[n]);
    const Number displacement = ext::magnitude(direction);
    ext::normalize(&direction);

    // The sum of the ball radii.
    const Number sumOfRadii = _proxyBalls[si[0]].radius +
                              _proxyBalls[si[1]].radius;
    // Record the relative penetration.
    *relativePenetrations++ = displacement / sumOfRadii;

    typename HashTable::iterator springConstantIterator =
      savedSpringConstants.find(id);
    // If this is a new contact.
    if (springConstantIterator == savedSpringConstants.end()) {
      // Compute the spring constant.
      // For the source and target simplex.
      for (std::size_t i = 0; i != 2; ++i) {
        // Component of the velocity along penetration vector.
        nv[i] = ext::dot(pb[i]->velocity, direction);
      }

      // The maximum spring displacement.
      const Number maximumDisplacement = _maximumRelativePenetration *
                                         sumOfRadii;
      // The spring constant is determined by equating the combined kinetic
      // energy with the potential energy in the spring.
      springConstant = (pb[0]->mass * nv[0] * nv[0] + pb[1]->mass * nv[1] * nv[1]) /
                       (maximumDisplacement * maximumDisplacement);
      dampingConstant = 0;
      if (_dampingFraction != 0) {
        dampingConstant = std::abs(pb[1]->mass * nv[1] - pb[0]->mass * nv[0]) /
                          maximumDisplacement;
      }
      // Save the spring constant and the damping constant for next time.
      _springAndDampingConstants.insert
      (HashTableValue(id, std::make_pair(springConstant, dampingConstant)));
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
        // Component along the penetration vector.
        nv[i] = ext::dot(pb[i]->velocity, direction);
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
}

// Report contact using the proxy ball method.
template<std::size_t N, typename _Identifier>
template<typename _OutputIterator>
inline
void
ProxyBallContact<N, _Identifier>::
computeContact(_OutputIterator contacts)
{
  typedef typename std::vector<ProxyBall>::iterator ProxyBallIterator;
  // The ORQ stores iterators to iterator to the proxy balls.
  typedef typename std::vector<ProxyBallIterator>::iterator Record;
  typedef geom::CellArrayStatic<N, GetProxyBallCenter<Record, Point> > Orq;
  typedef typename Orq::BBox BBox;

#if 0
  // CONTINUE
  for (std::size_t i = 0; i != _components.size(); ++i) {
    for (std::size_t j = 0; j != _components[i].size(); ++j) {
      std::cout << i << '\n' << *_components[i][j] << '\n';
    }
  }
#endif

  // No need to check for contact if there are less than 2 active components.
  if (countComponents() < 2) {
    return;
  }

  BBox bb;
  Point p;
  std::tuple<std::size_t, std::size_t, Point> penetration;
  std::vector<Record> potential;
  Number offset;
  // For each connected component.
  for (std::size_t component = 0; component != _numberOfComponents;
       ++component) {
    if (_components[component].empty()) {
      continue;
    }
    // Build an ORQ data structure for the active proxy balls in this component.
    Orq orq(_components[component].begin(), _components[component].end());

    // For each of the other components.
    for (std::size_t other = 0; other != _numberOfComponents; ++other) {
      if (other == component) {
        continue;
      }
      // For each proxy ball in the component.
      for (std::size_t k = 0; k != _components[other].size(); ++k) {
        // The index of this proxy ball.
        const std::size_t sourceIndex =
          std::distance(_proxyBalls.begin(), _components[other][k]);
        // Only consider contact in which the source proxy balls is local
        // (not ghost).
        if (sourceIndex >= _numberOfLocalProxyBalls) {
          continue;
        }
        const ProxyBall& source = *_components[other][k];
        // Make a bounding box that contains all possible contacts.
        offset = source.radius *
                 (2. + 10. * std::numeric_limits<Number>::epsilon());
        p = source.center;
        p -= offset;
        bb.lower = p;
        p = source.center;
        p += offset;
        bb.upper = p;
        // Get the potential contacts.
        potential.clear();
        orq.computeWindowQuery(std::back_inserter(potential), bb);
        // For each potential contact.
        for (std::size_t n = 0; n != potential.size(); ++n) {
          // The index of the potential contact.
          const std::size_t targetIndex =
            std::distance(_proxyBalls.begin(), *potential[n]);
          const ProxyBall& target = **potential[n];
          // In reporting contacts, either the first radius is greater than the
          // second or the radii are equal and the first index is less than the
          // second.
          if (!(source.radius > target.radius ||
                (source.radius == target.radius &&
                 source.elementIdentifier < target.elementIdentifier))) {
            continue;
          }
          // The squared distance between the centroids.
          const Number sd = ext::squaredDistance(source.center, target.center);
          const Number radiiSum = source.radius + target.radius;
          // If the balls are in contact.
          if (sd < radiiSum * radiiSum) {
            // The penetration vector.
            p = source.center;
            p -= target.center;
            ext::normalize(&p);
            p *= radiiSum - std::sqrt(sd);
            // Report the penetration.
            std::get<0>(penetration) = sourceIndex;
            std::get<1>(penetration) = targetIndex;
            std::get<2>(penetration) = p;
            *contacts++ = penetration;
          }
        }
      }
    }
  }
}

template<std::size_t N, typename _Identifier>
inline
typename ProxyBallContact<N, _Identifier>::Number
ProxyBallContact<N, _Identifier>::
computeStableTimeStep(const std::size_t* components) const
{
  //
  // Compute the stable time step. Check for the worst case scenario of each
  // element colliding head-on with a mirror image of itself.
  //
  Number stableTimeStep = std::numeric_limits<Number>::max();
  // For each active proxy ball.
  for (std::size_t i = 0; i != _proxyBalls.size(); ++i) {
    if (components[i] >= _numberOfComponents) {
      continue;
    }
    const Number speed = ext::magnitude(_proxyBalls[i].velocity);
    // The factor of 1/2 is because the mirror image is moving in the
    // opposite direction.
    const Number d = 0.5 * _maximumRelativePenetration * _proxyBalls[i].radius;
    // Ensure we don't divide by zero.
    const Number t = d / (speed + d * std::numeric_limits<Number>::epsilon());
    if (t < stableTimeStep) {
      stableTimeStep = t;
    }
  }
  return stableTimeStep;
}


template<typename T, std::size_t N>
inline
T
computeMidpointRadius
(const std::array < std::array<T, N>, N + 1 > & simplex,
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
      const T d = squaredDistance(centroid, x);
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

template<std::size_t N, typename _Identifier>
void
ProxyBallContact<N, _Identifier>::
computeRadii(const Number* nodeCoordinates)
{
  Simplex s;
  Point centroid;
  // For each simplex.
  for (std::size_t i = 0; i != _indexSimplices.size(); ++i) {
    computeSimplex(&s, nodeCoordinates, i);
    // Compute the centroid for the simplex.
    geom::computeCentroid(s, &centroid);
    // Compute the radius.
    _proxyBalls[i].radius = computeMidpointRadius(s, centroid);
  }
}

template<std::size_t N, typename _Identifier>
void
ProxyBallContact<N, _Identifier>::
computeSimplex(Simplex* s, const Number* nodeCoordinates, const std::size_t i)
const
{
  // For each vertex.
  for (std::size_t m = 0; m != N + 1; ++m) {
    // The index of the first vertex coordinate.
    const std::size_t v = _indexSimplices[i][m] * N;
    // For each coordinate.
    for (std::size_t n = 0; n != N; ++n) {
      (*s)[m][n] = nodeCoordinates[v + n];
    }
  }
}

template<std::size_t N, typename _Identifier>
void
ProxyBallContact<N, _Identifier>::
computeVelocity(Point* velocity, const Number* velocityCoordinates,
                const std::size_t i)
{
  std::fill(velocity->begin(), velocity->end(), 0);
  // For each vertex.
  for (std::size_t m = 0; m != N + 1; ++m) {
    // The index of the first velocity coordinate.
    const std::size_t v = _indexSimplices[i][m] * N;
    // For each coordinate.
    for (std::size_t n = 0; n != N; ++n) {
      (*velocity)[n] += velocityCoordinates[v + n];
    }
  }
  // Divide by the number of nodes in the simplex to get the average.
  *velocity *= 1. / (N + 1);
}

template<std::size_t N, typename _Identifier>
std::size_t
ProxyBallContact<N, _Identifier>::
countComponents() const
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != _components.size(); ++i) {
    count += ! _components.empty();
  }
  return count;
}

} // namespace contact
}
