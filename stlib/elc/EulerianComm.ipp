// -*- C++ -*-

#if !defined(__elc_EulerianComm_ipp__)
#error This file is an implementation detail of the class EulerianComm.
#endif

namespace stlib
{
namespace elc
{


//
// Communication
//


template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
receiveMesh(const BBox& domain)
{
#ifdef ELC_USE_CPP_INTERFACE
  const int commWorldRank = _comm.Get_rank();
#else
  int commWorldRank;
  MPI_Comm_rank(_comm, &commWorldRank);
#endif

  //
  // Determine the point-to-point communication pattern.
  //

  _lagrangianData.clear();
  // We don't use the overlapping Lagrangian domains, so we use a
  // trivial output iterator.
  _pointToPoint.solve(domain, commWorldRank,
                      ads::constructTrivialOutputIterator(),
                      std::back_inserter(_lagrangianData));

  //
  // Resize the vectors of boundary information.
  //

  const std::size_t numComm = _lagrangianData.size();
  _identifiers.resize(numComm);
  _positions.resize(numComm);
  _velocities.resize(numComm);
  _connectivities.resize(numComm);

  //
  // Receive the boundary from each relevant Lagrangian processor.
  //

  _identifierRequests.resize(numComm);
  _positionRequests.resize(numComm);
  _velocityRequests.resize(numComm);
  _connectivityRequests.resize(numComm);

  for (std::size_t i = 0; i != numComm; ++i) {
    const int lagrangianProc = _lagrangianData[i][0];
    const int numNodesToReceive = _lagrangianData[i][1];
    const int numFacesToReceive = _lagrangianData[i][2];

    _identifiers[i].resize(numNodesToReceive);
    _positions[i].resize(numNodesToReceive);
    _velocities[i].resize(numNodesToReceive);
    _connectivities[i].resize(numFacesToReceive);
#ifdef ELC_USE_CPP_INTERFACE
    _identifierRequests[i] =
      _comm.Irecv(&_identifiers[i][0], numNodesToReceive, MPI::INT,
                  lagrangianProc, TagIdentifiers);

    _positionRequests[i] =
      _comm.Irecv(&_positions[i][0], numNodesToReceive * N, _mpiNumber,
                  lagrangianProc, TagPositions);

    _velocityRequests[i] =
      _comm.Irecv(&_velocities[i][0], numNodesToReceive * N, _mpiNumber,
                  lagrangianProc, TagVelocities);

    _connectivityRequests[i] =
      _comm.Irecv(&_connectivities[i][0], numFacesToReceive * N, MPI::INT,
                  lagrangianProc, TagFaceData);
#else
    MPI_Irecv(&_identifiers[i][0], numNodesToReceive, MPI_INT,
              lagrangianProc, TagIdentifiers, _comm,
              &_identifierRequests[i]);

    MPI_Irecv(&_positions[i][0], numNodesToReceive * N, _mpiNumber,
              lagrangianProc, TagPositions, _comm, &_positionRequests[i]);

    MPI_Irecv(&_velocities[i][0], numNodesToReceive * N, _mpiNumber,
              lagrangianProc, TagVelocities, _comm,
              &_velocityRequests[i]);

    MPI_Irecv(&_connectivities[i][0], numFacesToReceive * N, MPI_INT,
              lagrangianProc, TagFaceData, _comm,
              &_connectivityRequests[i]);
#endif
  }
}



template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
waitForMesh()
{
  //
  // Receive the identifiers and the positions.
  // If the solid does not provide global identifiers, the positions are
  // needed to construct these identifiers.
  //
  bool shouldGenerateIdentifiers = true;
  MpiStatus status;
  const std::size_t numComm = _lagrangianData.size();
  for (std::size_t i = 0; i != numComm; ++i) {
    const int numNodesToReceive = _lagrangianData[i][1];
    const int numNodesN = numNodesToReceive * N;
    int count;
    // Wait for the identifier receives to complete.
#ifdef ELC_USE_CPP_INTERFACE
    _identifierRequests[i].Wait(status);
    count = status.Get_count(MPI::INT);
#else
    MPI_Wait(&_identifierRequests[i], &status);
    MPI_Get_count(&status, MPI_INT, &count);
#endif
    if (count != 0) {
      shouldGenerateIdentifiers = false;
    }
    if (shouldGenerateIdentifiers) {
      assert(count == 0);
    }
    else {
      assert(count == numNodesToReceive);
    }

    // Wait for the position receives to complete.
#ifdef ELC_USE_CPP_INTERFACE
    _positionRequests[i].Wait(status);
    count = status.Get_count(_mpiNumber);
#else
    MPI_Wait(&_positionRequests[i], &status);
    MPI_Get_count(&status, _mpiNumber, &count);
#endif
    assert(count == numNodesN);

    // Wait for the velocity receives to complete.
#ifdef ELC_USE_CPP_INTERFACE
    _velocityRequests[i].Wait(status);
    count = status.Get_count(_mpiNumber);
#else
    MPI_Wait(&_velocityRequests[i], &status);
    MPI_Get_count(&status, _mpiNumber, &count);
#endif
    assert(count == numNodesN);

    // Wait for the connectivity receives to complete.
    const int numFacesSentN = _lagrangianData[i][2] * N;
#ifdef ELC_USE_CPP_INTERFACE
    _connectivityRequests[i].Wait(status);
    count = status.Get_count(MPI::INT);
#else
    MPI_Wait(&_connectivityRequests[i], &status);
    MPI_Get_count(&status, MPI_INT, &count);
#endif
    assert(count == numFacesSentN);
  }

#if 0
  // Disabled.
  if (shouldGenerateIdentifiers) {
    generateIdentifiers();
  }
#endif

  //
  // Build the mapping from node identifiers to node indices in the assembled
  // boundary.
  //

  _identifierToIndex.clear();
  std::pair<std::map<int, int>::iterator, bool> insertResult;
  // Loop over the boundary portions.
  for (std::size_t i = 0; i != numComm; ++i) {
    // Loop over nodes.
    for (std::size_t n = 0; n != _identifiers[i].size(); ++n) {
      // If the identifier is not in the mapping.
      if (! _identifierToIndex.count(_identifiers[i][n])) {
        // Add it to the mapping.
        std::map<int, int>::value_type
        insertValue(_identifiers[i][n], _identifierToIndex.size());
        insertResult = _identifierToIndex.insert(insertValue);
        // Make sure that it was inserted.
        assert(insertResult.second);
      }
    }
  }

  //
  // Build the assembled boundary.
  //

  // Determine the number of nodes and faces.
  const std::size_t numAssembledNodes = _identifierToIndex.size();
  std::size_t numAssembledFaces = 0;
  for (std::size_t i = 0; i != _connectivities.size(); ++i) {
    numAssembledFaces += _connectivities[i].size();
  }

  // Allocate memory.
  _assembledPositions.resize(numAssembledNodes);
  _assembledVelocities.resize(numAssembledNodes);
  _assembledConnectivities.resize(numAssembledFaces);

  // Set the positions and velocities.
  {
    std::map<int, int>::const_iterator pairIterator;
    int index;
    for (std::size_t i = 0; i != numComm; ++i) {
      for (std::size_t n = 0; n != _positions[i].size(); ++n) {
        // Find the identifier.
        pairIterator = _identifierToIndex.find(_identifiers[i][n]);
        // Make sure that we found it.
        assert(pairIterator != _identifierToIndex.end());
        // Extract the index.
        index = pairIterator->second;
        // Set the position and velocity.
        _assembledPositions[index] = _positions[i][n];
        _assembledVelocities[index] = _velocities[i][n];
      }

      // Free the position and velocity memory for the i_th processor.
      {
        // I do this to placate the xlC compiler on frost.
        int size = 0;
        _positions[i].resize(size);
        _velocities[i].resize(size);
      }
      // Don't free the identifier memory.  We'll need that for sending
      // the pressure.
    }
  }

  // Set the connectivities.
  {
    std::map<int, int>::const_iterator pairIterator;
    std::size_t localIndex, cellIndex = 0;
    int globalIdentifier;
    for (std::size_t i = 0; i != numComm; ++i) {
      for (std::size_t j = 0; j != _connectivities[i].size(); ++j) {
        for (std::size_t n = 0; n != N; ++n) {
          if (_vertexIdentifierStyle == LocalIndices) {
            // The local node index in the i_th Lagrangrian processors
            // with which we communicate.
            localIndex = _connectivities[i][j][n];
            // Make sure the local index is in the right range.
            assert(localIndex < _identifiers[i].size());
            // Switch from the local Lagrangian identifiers to the global
            // Lagrangian identifiers.
            globalIdentifier = _identifiers[i][localIndex];
          }
          else {
            assert(_vertexIdentifierStyle == GlobalIdentifiers);
            globalIdentifier = _connectivities[i][j][n];
          }
          // Find the identifier.
          pairIterator = _identifierToIndex.find(globalIdentifier);
          // Make sure that we found it.
          assert(pairIterator != _identifierToIndex.end());
          // Extract the node index.
          _assembledConnectivities[cellIndex][n] = pairIterator->second;
        }
        ++cellIndex;
      }
      // Free the connectivities memory for the i_th processor.
      _connectivities[i].resize(0);
    }
    // Sanity check.
    assert(cellIndex == _assembledConnectivities.size());
  }

  initializePressure();
}



template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
generateIdentifiers()
{
  //
  // Initially assign each node an identifier.
  //
  const std::size_t numComm = _lagrangianData.size();
  int globalIdentifier = 0;
  // Loop over the boundary portions.
  for (std::size_t i = 0; i != numComm; ++i) {
    // Loop over nodes in this patch.
    for (std::size_t n = 0; n != _identifiers[i].size(); ++n) {
      _identifiers[i][n] = globalIdentifier++;
    }
  }

  //
  // Now find the nodes that are on the boundary of each patch.
  //
  std::vector<std::pair<int*, const Point*> > boundaryNodes;
  std::pair<int*, const Point*> value;
  std::set<int> boundaryIndices;
  // Loop over the boundary portions.
  for (std::size_t i = 0; i != numComm; ++i) {
    // Get the boundary nodes from this patch.
    geom::IndSimpSetIncAdj < N, N - 1, Number >
    patch(_positions[i].size(), _positions[i].data(),
          _connectivities[i].size(), _connectivities[i].data());

    geom::determineBoundaryVertices
    (patch, std::inserter(boundaryIndices, boundaryIndices.end()));
    for (std::set<int>::const_iterator iter = boundaryIndices.begin();
         iter != boundaryIndices.end(); ++iter) {
      value.first = &_identifiers[i][*iter];
      value.second = &_positions[i][*iter];
      boundaryNodes.push_back(value);
    }
    boundaryIndices.clear();
  }

  //
  // Sort the boundary nodes.  Use the position as a composite number.
  //
  // CONTINUE: An ORQ would have better computational complexity.
  {
    // Functor that selects the second of the pair and then dereferences.
    typedef
    ads::unary_compose_unary_unary < ads::Dereference<const Point*>,
        ads::Select2nd< std::pair<int*, const Point*> > > GetPoint;
    // Functor that compares the std::pairs.
    typedef
    ads::binary_compose_binary_unary < ads::less_composite<N, Point>,
        GetPoint, GetPoint > Comp;

    // Set the x coordinate as the first in the composite number.
    ads::less_composite<N, Point> lc;
    lc.set(0);

    // Make the comparison functor.
    GetPoint gp;
    Comp comp(lc, gp, gp);

    // Sort the nodes.
    std::sort(boundaryNodes.begin(), boundaryNodes.end(), comp);
  }

  //
  // Detect which nodes coincide and fix those identifiers.
  //

  // Any nodes closer than epsilon to each other will be considered the
  // same node.
  // CONTINUE: Use the bounding box of all of the nodes.
  const Number epsilon =
    std::sqrt(std::numeric_limits<Number>::epsilon());
  const Number epsilonSquared = epsilon * epsilon;

  // Loop over the boundary nodes.
  const std::size_t size = boundaryNodes.size();
  Number x;
  int m;
  for (std::size_t n = 1; n < size; ++n) {
    // Look for nodes that coincide.
    x = (*boundaryNodes[n].second)[0];
    m = n - 1;
    while (m >= 0 && x - (*boundaryNodes[m].second)[0] < epsilon) {
      if (squaredDistance(*boundaryNodes[m].second,
                          *boundaryNodes[n].second) <
          epsilonSquared) {
        // Fix the identifier.
        *boundaryNodes[n].first = *boundaryNodes[m].first;
        break;
      }
      --m;
    }
  }
}


namespace internal
{

template<typename T>
inline
void
computeFaceNormal(const std::vector<std::array<T, 2> >& pos,
                  const std::vector<std::array<int, 2> >& conn,
                  const std::size_t n, std::array<T, 2>* normal,
                  std::integral_constant<std::size_t, 2> /* dummy */)
{
  std::array<T, 2> tangent(pos[conn[n][1]]);
  tangent -= pos[conn[n][0]];
  normalize(&tangent);
  (*normal)[0] = tangent[1];
  (*normal)[1] = - tangent[0];
}

template<typename T>
inline
void
computeFaceNormal(const std::vector<std::array<T, 3> >& pos,
                  const std::vector<std::array<int, 3> >& conn,
                  const std::size_t n, std::array<T, 3>* normal,
                  std::integral_constant<std::size_t, 3> /* dummy */)
{
  const std::array<int, 3>& face = conn[n];
  cross(pos[face[2]] - pos[face[1]], pos[face[0]] - pos[face[1]], normal);
  normalize(normal);
}

template<std::size_t N, typename T>
inline
void
computeFaceNormal(const std::vector<std::array<T, N> >& pos,
                  const std::vector<std::array<int, N> >& conn,
                  const std::size_t n, std::array<T, N>* normal)
{
  computeFaceNormal(pos, conn, n, normal,
                    std::integral_constant<std::size_t, N>());
}

}


// Compute the face normals.
template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
computeFaceNormals()
{
  // Resize the array.
  _faceNormals.resize(getNumberOfFaces());

  // Loop over the faces.
  for (std::size_t i = 0; i != getNumberOfFaces(); ++i) {
    // Compute the face normal.
    computeFaceNormal(i, &_faceNormals[i]);
  }
}


// Compute the face centroids.
template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
computeFaceCentroids()
{
  // Resize the array.
  _faceCentroids.resize(getNumberOfFaces());

  // Loop over the faces.
  for (std::size_t i = 0; i != getNumberOfFaces(); ++i) {
    // Compute the face centroid.
    computeFaceCentroid(i, &_faceCentroids[i]);
  }
}


template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
computeFaceNormal(const std::size_t n, Point* normal) const
{
  // Get the face normal.
  internal::computeFaceNormal<N>(_assembledPositions,
                                 _assembledConnectivities, n, normal);
}


template<std::size_t N, typename T>
inline
void
EulerianComm<N, T>::
computeFaceCentroid(const std::size_t n, Point* centroid) const
{
  // The centroid is the arithmetic mean of the face node positions.

  std::fill(centroid->begin(), centroid->end(), 0);
  // CONTINUE
  //assert(0 <= n && n < _assembledConnectivities.size());
  const IndexedFace& face = _assembledConnectivities[n];
  // Loop over the nodes of the face.
  for (std::size_t j = 0; j != N; ++j) {
    //centroid += _assembledPositions[ _assembledConnectivities[n][j] ];
    *centroid += _assembledPositions[face[j]];
  }
  *centroid /= Number(N);
}

} // namespace elc
}
