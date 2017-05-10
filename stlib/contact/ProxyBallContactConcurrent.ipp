// -*- C++ -*-

#if !defined(__contact_ProxyBallContactConcurrent_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace contact
{

template<std::size_t N, typename _Identifier>
ProxyBallContactConcurrent<N, _Identifier>::
ProxyBallContactConcurrent(const MPI::Intracomm& comm,
                           const std::size_t numberOfComponents,
                           const std::size_t numberOfNodes,
                           const Number* nodeCoordinates,
                           const Identifier* nodeIdentifiers,
                           const std::size_t numberOfElements,
                           const Identifier* identifierSimplices,
                           const std::size_t* components,
                           const Number maximumRelativePenetration,
                           const Number springFraction,
                           const Number dampingFraction) :
  _comm(comm.Dup()),
  _numberOfLocalProxyBallsInComponent(numberOfComponents, 0),
  _maxRadii(comm.Get_size()),
  _areReceivingProxyBallsFrom(comm.Get_size()),
  _areSendingProxyBallsTo(comm.Get_size()),
  Base(numberOfComponents, numberOfNodes, nodeCoordinates, nodeIdentifiers,
       numberOfElements, identifierSimplices, maximumRelativePenetration,
       springFraction, dampingFraction)
{
  // Initialize the component information so that we can compute the max radii.
  {
    // Dummy velocities.
    std::vector<Number> velocityCoordinates(numberOfNodes * N, 0);
    // Dummy masses.
    std::vector<Number> masses(numberOfNodes, 0);
    Base::initializeProxyBalls(nodeCoordinates, &velocityCoordinates[0],
                               &masses[0], components);
  }
  computeGlobalIdentifiers();
  computeMaxRadii();
  // Record the number of local proxy balls in each component.
  for (std::size_t i = 0; i != _numberOfLocalProxyBallsInComponent.size();
       ++i) {
    _numberOfLocalProxyBallsInComponent[i] = _components.size();
  }
}

template<std::size_t N, typename _Identifier>
template < typename _ForceOutputIterator, typename _CountOutputIterator,
           typename _RelPenOutputIterator >
inline
typename ProxyBallContactConcurrent<N, _Identifier>::Number
ProxyBallContactConcurrent<N, _Identifier>::
operator()(const Number* nodeCoordinates,
           const Number* velocityCoordinates,
           const Number* masses,
           const std::size_t* components,
           _ForceOutputIterator elementForces,
           _CountOutputIterator interactionCounts,
           _RelPenOutputIterator relativePenetrations)
{
  Base::initializeProxyBalls(nodeCoordinates, velocityCoordinates, masses,
                             components);
  exchangeProxyBalls(components);
#if 0
  // CONTINUE: The proxy balls look OK.
  if (_comm.Get_rank() == 0) {
    for (std::size_t i = 0; i != _components.size(); ++i) {
      for (std::size_t j = 0; j != _components[i].size(); ++j) {
        std::cout << i << ' ' << j << '\n' << *_components[i][j] << '\n';
      }
    }
  }
#endif
  // Get the penetrations in terms of the simplex indices.
  std::vector<std::tuple<std::size_t, std::size_t, Point> > penetrations;
  Base::computeContact(std::back_inserter(penetrations));
#if 0
  // CONTINUE The penetration have bad element indices.
  if (_comm.Get_rank() == 0) {
    for (std::size_t i = 0; i != penetrations.size(); ++i) {
      std::cout << std::get<0>(penetrations[i]) << ' '
                << std::get<1>(penetrations[i]) << ' '
                << std::get<2>(penetrations[i]) << '\n';
    }
  }
#endif
  std::vector<std::tuple<std::size_t, Point> > allForces;
  Base::computeForces(penetrations, std::back_inserter(allForces),
                      interactionCounts, relativePenetrations);
  //std::cout << "exchangeForces\n";
  exchangeForces(allForces, elementForces);
  //std::cout << "purgeGhosts\n";
  purgeGhosts();
  //std::cout << "computeStableTimeStep\n";
  return Base::computeStableTimeStep(components);
}

template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
computeGlobalIdentifiers()
{
  // Gather the number of elements in each processor.
  std::vector<std::size_t> sizes(_comm.Get_size());
  const std::size_t size = _proxyBalls.size();
  _comm.Allgather(&size, sizeof(std::size_t), MPI::BYTE,
                  &sizes[0], sizeof(std::size_t), MPI::BYTE);
  // Compute the index offset.
  const std::size_t offset = std::accumulate(sizes.begin(),
                             sizes.begin() + _comm.Get_rank(),
                             std::size_t(0));
  // Set the processor ranks and element identifiers for each proxy ball.
  const std::size_t rank = _comm.Get_rank();
  for (std::size_t i = 0; i != _proxyBalls.size(); ++i) {
    _proxyBalls[i].processor = rank;
    _proxyBalls[i].elementIdentifier = offset + i;
  }
}

template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
computeMaxRadii()
{
  // Allocate memory.
  for (std::size_t i = 0; i != _maxRadii.size(); ++i) {
    _maxRadii[i].resize(_numberOfComponents);
  }
  //
  // Compute the maximum radii for the local proxy balls.
  //
  std::vector<Number>& localMaxRadii = _maxRadii[_comm.Get_rank()];
  // Compute the maximum proxy ball radius in each component.
  for (std::size_t i = 0; i != _components.size(); ++i) {
    localMaxRadii[i] = 0;
    for (std::size_t j = 0; j != _components[i].size(); ++j) {
      if (_components[i][j]->radius > localMaxRadii[i]) {
        localMaxRadii[i] = _components[i][j]->radius;
      }
    }
    // Expand to avoid truncation errors.
    localMaxRadii[i] *=
      (1 + 10 * std::numeric_limits<Number>::epsilon());
  }
  //
  // Exchange the maximum radii to get the information from the other
  // processors.
  //
  std::vector<Number> buffer(_numberOfComponents * _comm.Get_size());
  const std::size_t size = sizeof(Number) * _numberOfComponents;
  _comm.Allgather(&localMaxRadii[0], size, MPI::BYTE, &buffer[0],
                  size, MPI::BYTE);
  // Copy from the buffer.
  std::size_t i = 0;
  for (std::size_t rank = 0; rank != _comm.Get_size(); ++rank) {
    for (std::size_t c = 0; c != _numberOfComponents; ++c) {
      _maxRadii[rank][c] = buffer[i++];
    }
  }
}

template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
exchangeProxyBalls(const std::size_t* components)
{
  //
  // Exchange the bounding boxes.
  //
  const BBox empty(ext::filled_array<Point>(0), ext::filled_array<Point>(-1));
  // Compute the bounding boxes for each component.
  std::vector<BBox> boundingBoxes(_numberOfComponents, empty);
  for (std::size_t i = 0; i != _components.size(); ++i) {
    if (! _components[i].empty()) {
      // Initialize with the first proxy ball center.
      boundingBoxes[i].setLowerCorner(_components[i][0]->center);
      boundingBoxes[i].setUpperCorner(_components[i][0]->center);
      // Add the rest of the center points.
      for (std::size_t j = 1; j != _components[i].size(); ++j) {
        boundingBoxes[i].add(_components[i][j]->center);
      }
    }
  }
  // Exchange the bounding boxes.
  std::vector<BBox> boundingBoxBuffer(_numberOfComponents * _comm.Get_size());
  _comm.Allgather(&boundingBoxes[0], sizeof(BBox) * _numberOfComponents,
                  MPI::BYTE, &boundingBoxBuffer[0],
                  sizeof(BBox) * _numberOfComponents, MPI::BYTE);

  // The regions of interest for the local components.
  std::vector<BBox> regionsOfInterest(boundingBoxes);
  for (std::size_t i = 0; i != regionsOfInterest.size(); ++i) {
    expandBoundingBox(&regionsOfInterest[i], _comm.Get_rank(), i);
  }

  //
  // Determine the processors from which we will receive proxy balls.
  //
  std::fill(_areReceivingProxyBallsFrom.begin(),
            _areReceivingProxyBallsFrom.end(), false);
  // For each other processor.
  for (std::size_t i = 0; i != _areReceivingProxyBallsFrom.size(); ++i) {
    if (i == _comm.Get_rank()) {
      continue;
    }
    // For each component.
    for (std::size_t j = 0; j != _numberOfComponents; ++j) {
      const BBox& region = regionsOfInterest[j];
      if (region.isEmpty()) {
        continue;
      }
      // For each other component.
      for (std::size_t k = 0; k != _numberOfComponents; ++k) {
        const BBox& box = boundingBoxBuffer[i * _numberOfComponents + k];
        if (k == j || box.isEmpty()) {
          continue;
        }
        // If the region of interest the local part of component j overlaps
        // the bounding box of component k on processor i.
        if (doOverlap(region, box)) {
          _areReceivingProxyBallsFrom[i] = true;
        }
      }
    }
  }

  //
  // Determine the proxy balls to send.
  //
  // Convert the distributed bounding boxes to regions of interest.
  for (std::size_t i = 0; i != _comm.Get_size(); ++i) {
    for (std::size_t j = 0; j != _numberOfComponents; ++j) {
      expandBoundingBox(&boundingBoxBuffer[i * _numberOfComponents + j], i, j);
    }
  }
  std::fill(_areSendingProxyBallsTo.begin(), _areSendingProxyBallsTo.end(),
            false);
  std::vector<std::size_t> numberToSend(_comm.Get_size(), 0);
  std::vector<std::set<std::size_t> > proxyBallsToSend(_comm.Get_size());
  // For each other processor.
  for (std::size_t i = 0; i != proxyBallsToSend.size(); ++i) {
    if (i == _comm.Get_rank()) {
      continue;
    }
    // For each component.
    for (std::size_t j = 0; j != _numberOfComponents; ++j) {
      const BBox& box = boundingBoxes[j];
      if (box.isEmpty()) {
        continue;
      }
      // For each other component.
      for (std::size_t k = 0; k != _numberOfComponents; ++k) {
        const BBox& region = boundingBoxBuffer[i * _numberOfComponents + k];
        if (k == j || region.isEmpty()) {
          continue;
        }
        // If the bounding box around the local part of component j overlaps
        // the region of interest of component k on processor i.
        if (doOverlap(box, region)) {
          _areSendingProxyBallsTo[i] = true;
          // Determine which proxy balls we should send.
          for (std::size_t m = 0; m != _components[j].size(); ++m) {
            if (components[_components[j][m]->elementIndex] <
                _numberOfComponents &&
                region.isIn(_components[j][m]->center)) {
              proxyBallsToSend[i].insert(std::distance(_proxyBalls.begin(),
                                         _components[j][m]));
            }
          }
        }
      }
    }
    numberToSend[i] = proxyBallsToSend[i].size();
  }
  // Copy the proxy balls to send into buffers.
  std::vector<std::vector<ProxyBall> >
  proxyBallSendBuffers(proxyBallsToSend.size());
  for (std::size_t i = 0; i != proxyBallSendBuffers.size(); ++i) {
    for (std::set<std::size_t>::const_iterator iter =
           proxyBallsToSend[i].begin();
         iter != proxyBallsToSend[i].end(); ++iter) {
      proxyBallSendBuffers[i].push_back(_proxyBalls[*iter]);
    }
  }

  //
  // Communicate the number of proxy balls that we will send.
  //
  // Post the receives.
  std::vector<MPI::Request> receiveSizes;
  std::vector<std::size_t> numberToReceive(_comm.Get_size(), 0);
  for (std::size_t i = 0; i != _areReceivingProxyBallsFrom.size(); ++i) {
    if (_areReceivingProxyBallsFrom[i]) {
      receiveSizes.push_back(_comm.Irecv(&numberToReceive[i],
                                         sizeof(std::size_t),
                                         MPI::BYTE, i, TagSize));
    }
  }
  // Post the sends.
  std::vector<MPI::Request> sendSizes;
  for (std::size_t i = 0; i != _areSendingProxyBallsTo.size(); ++i) {
    if (_areSendingProxyBallsTo[i]) {
      sendSizes.push_back(_comm.Isend(&numberToSend[i], sizeof(std::size_t),
                                      MPI::BYTE, i, TagSize));
    }
  }
  // Wait for receiving the number of proxy balls to complete.
  MPI::Request::Waitall(receiveSizes.size(), &receiveSizes[0]);
  // Wait for sending the number of proxy balls to complete.
  MPI::Request::Waitall(sendSizes.size(), &sendSizes[0]);

  //
  // Communicate the proxy balls.
  //
  // Allocate buffers to receive the proxy balls.
  std::vector<std::vector<ProxyBall> > proxyBallBuffers(_comm.Get_size());
  for (std::size_t i = 0; i != proxyBallBuffers.size(); ++i) {
    proxyBallBuffers[i].resize(numberToReceive[i]);
  }
  // Post the receives.
  std::vector<MPI::Request> receiveProxyBalls;
  for (std::size_t i = 0; i != proxyBallBuffers.size(); ++i) {
    if (! proxyBallBuffers[i].empty()) {
      receiveProxyBalls.push_back
      (_comm.Irecv(&proxyBallBuffers[i][0],
                   sizeof(ProxyBall) * proxyBallBuffers[i].size(),
                   MPI::BYTE, i, TagProxyBalls));
    }
  }
  // Post the sends.
  std::vector<MPI::Request> sendProxyBalls;
  for (std::size_t i = 0; i != proxyBallSendBuffers.size(); ++i) {
    if (! proxyBallSendBuffers[i].empty()) {
      sendProxyBalls.push_back
      (_comm.Isend(&proxyBallSendBuffers[i][0],
                   sizeof(ProxyBall) * proxyBallSendBuffers[i].size(),
                   MPI::BYTE, i, TagProxyBalls));
    }
  }
  // Wait for receiving the proxy balls to complete.
  MPI::Request::Waitall(receiveProxyBalls.size(), &receiveProxyBalls[0]);

  // CONTINUE: Do I need to free the requests?

  //
  // Append the ghost proxy balls. Note: This invalidates the iterators to
  // the proxy balls.
  //
  for (std::size_t i = 0; i != proxyBallBuffers.size(); ++i) {
    for (std::size_t j = 0; j != proxyBallBuffers[i].size(); ++j) {
      _proxyBalls.push_back(proxyBallBuffers[i][j]);
    }
  }

  // Rebuild the lists of active proxy balls.
  for (std::size_t i = 0; i != _components.size(); ++i) {
    _components[i].clear();
  }
  std::size_t index = 0;
  for (ProxyBallIterator i = _proxyBalls.begin(); i != _proxyBalls.end();
       ++i, ++index) {
    if (index >= _numberOfLocalProxyBalls ||
        components[index] < _numberOfComponents) {
      _components[i->component].push_back(i);
    }
  }

  // Wait for sending the proxy balls to complete.
  MPI::Request::Waitall(sendProxyBalls.size(), &sendProxyBalls[0]);
}

template<std::size_t N, typename _Identifier>
template<typename _ForceOutputIterator>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
exchangeForces(const std::vector<std::tuple<std::size_t, Point> >&
               allForces, _ForceOutputIterator elementForces)
{
  //
  // Separate the forces for the local and ghost elements.
  //
  Force force;
  std::vector<std::vector<Force> > forcesToSend(_comm.Get_size());
  for (std::size_t i = 0; i != allForces.size(); ++i) {
    const std::size_t index = std::get<0>(allForces[i]);
    // If this is a force on a local proxy ball.
    if (index < _numberOfLocalProxyBalls) {
      *elementForces++ = allForces[i];
#if 0
      if (_comm.Get_rank() == 0) {
        std::cout << "local " << std::get<0>(allForces[i]) << ' '
                  << std::get<1>(allForces[i]) << '\n';
      }
#endif
    }
    else {
      force = allForces[i];
#ifdef STLIB_DEBUG
      assert(index < _proxyBalls.size());
#endif
      // Convert the ghost index to a real index on the appropriate processor.
      std::get<0>(force) = _proxyBalls[index].elementIndex;
      // Record the force.
      forcesToSend[_proxyBalls[index].processor].push_back(force);
#if 0
      if (_comm.Get_rank() == 0) {
        std::cout << "ghost " << std::get<0>(force) << ' '
                  << std::get<1>(force) << '\n';
      }
#endif
    }
  }

  //
  // Communicate the number of forces that we will send.
  //
  // Copy the numbers into an array.
  std::vector<std::size_t> numberToSend(_comm.Get_size(), 0);
  for (std::size_t i = 0; i != numberToSend.size(); ++i) {
    numberToSend[i] = forcesToSend[i].size();
  }
  // Post the receives.
  std::vector<MPI::Request> receiveSizes;
  std::vector<std::size_t> numberToReceive(_comm.Get_size(), 0);
  for (std::size_t i = 0; i != _areSendingProxyBallsTo.size(); ++i) {
    if (_areSendingProxyBallsTo[i]) {
      receiveSizes.push_back(_comm.Irecv(&numberToReceive[i],
                                         sizeof(std::size_t),
                                         MPI::BYTE, i, TagSize));
    }
  }
  // Post the sends.
  std::vector<MPI::Request> sendSizes;
  for (std::size_t i = 0; i != _areReceivingProxyBallsFrom.size(); ++i) {
    if (_areReceivingProxyBallsFrom[i]) {
      sendSizes.push_back(_comm.Isend(&numberToSend[i], sizeof(std::size_t),
                                      MPI::BYTE, i, TagSize));
    }
  }
  // Wait for receiving the number of forces to complete.
  MPI::Request::Waitall(receiveSizes.size(), &receiveSizes[0]);

  //
  // Communicate the forces.
  //
  // Allocate buffers to receive the forces.
  std::vector<std::vector<Force> > forceBuffers(_comm.Get_size());
  for (std::size_t i = 0; i != forceBuffers.size(); ++i) {
    forceBuffers[i].resize(numberToReceive[i]);
  }
  // Post the receives.
  std::vector<MPI::Request> receiveForces;
  for (std::size_t i = 0; i != forceBuffers.size(); ++i) {
    if (! forceBuffers[i].empty()) {
      receiveForces.push_back
      (_comm.Irecv(&forceBuffers[i][0],
                   sizeof(Force) * forceBuffers[i].size(),
                   MPI::BYTE, i, TagForces));
    }
  }
  // Post the sends.
  std::vector<MPI::Request> sendForces;
  for (std::size_t i = 0; i != forcesToSend.size(); ++i) {
    if (! forcesToSend[i].empty()) {
      sendForces.push_back
      (_comm.Isend(&forcesToSend[i][0],
                   sizeof(Force) * forcesToSend[i].size(),
                   MPI::BYTE, i, TagForces));
    }
  }
  // Wait for receiving the forces to complete.
  MPI::Request::Waitall(receiveForces.size(), &receiveForces[0]);

  // Append the forces from other processors.
  for (std::size_t i = 0; i != forceBuffers.size(); ++i) {
    for (std::size_t j = 0; j != forceBuffers[i].size(); ++j) {
      *elementForces++ = forceBuffers[i][j];
    }
  }

  // Wait for sending the number of forces to complete.
  MPI::Request::Waitall(sendSizes.size(), &sendSizes[0]);
  // Wait for sending the forces to complete.
  MPI::Request::Waitall(sendForces.size(), &sendForces[0]);
}

template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
purgeGhosts()
{
  _proxyBalls.resize(_numberOfLocalProxyBalls);
  for (std::size_t i = 0; i != _components.size(); ++i) {
    _components[i].resize(_numberOfLocalProxyBallsInComponent[i]);
  }
}

template<std::size_t N, typename _Identifier>
inline
void
ProxyBallContactConcurrent<N, _Identifier>::
expandBoundingBox(BBox* box, const std::size_t rank,
                  const std::size_t component) const
{
  const Number radius = _maxRadii[rank][component];
  if (radius != 0) {
    Point corner = box->getLowerCorner();
    corner -= 2 * radius;
    box->setLowerCorner(corner);
    corner = box->getUpperCorner();
    corner += 2 * radius;
    box->setUpperCorner(corner);
  }
}

} // namespace contact
}
