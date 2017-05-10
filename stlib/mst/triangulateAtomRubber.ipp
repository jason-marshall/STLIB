// -*- C++ -*-

#if !defined(__mst_triangulateAtomRubber_ipp__)
#error This file is an implementation detail of triangulateAtomRubber.
#endif

namespace stlib
{
namespace mst
{



template<typename T>
inline
void
makeIntersectionCircle(const geom::Ball<T, 3>& atom,
                       const geom::Ball<T, 3>& clippingAtom,
                       geom::Circle3<T>* circle)
{
  typedef typename geom::Ball<T, 3>::Point Point;
  // Normal.  In the direction from the atom to the clipping atom.
  Point x = clippingAtom.center;
  x -= atom.center;
  ext::normalize(&x);
  circle->normal = x;
  // Center
  const T distance = computeClippingPlaneDistance(atom, clippingAtom);
  x *= distance;
  x += atom.center;
  circle->center = x;
  // Radius
  circle->radius = std::sqrt(atom.radius * atom.radius - distance * distance);
#ifdef STLIB_DEBUG
  assert(circle->isValid());
#endif
}




// Return true if the incident edge lengths are within the specified range.
template<typename T>
inline
bool
areEdgeLengthsAcceptable(const geom::IndSimpSetIncAdj<3, 2, T>* mesh,
                         const std::size_t vertexIndex,
                         const T minimumAllowedEdgeLength,
                         const T maximumAllowedEdgeLength)
{
  assert(0 < minimumAllowedEdgeLength &&
         maximumAllowedEdgeLength < std::numeric_limits<T>::max());

  const typename geom::IndSimpSetIncAdj<3, 2, T>::Vertex&
  vertex = mesh->vertices[vertexIndex];
  const std::size_t numberOfIncidentSimplices =
    mesh->incident.size(vertexIndex);

  std::size_t simplexIndex, otherVertexIndex;
  T length;
  // For each incident triangle.
  for (std::size_t i = 0; i != numberOfIncidentSimplices; ++i) {
    simplexIndex = mesh->incident(vertexIndex, i);
    // For each edge of the triangle.
    for (std::size_t j = 0; j != 3; ++j) {
      otherVertexIndex = mesh->indexedSimplices[simplexIndex][j];
      // Skip the edge that is opposite the specified vertex.
      if (otherVertexIndex == vertexIndex) {
        continue;
      }
      // This edge is incident to the specified vertex.
      // Compute its length.
      length = geom::computeDistance(vertex,
                                     mesh->vertices[otherVertexIndex]);
      // If the incident edge length is not acceptable.
      if (length < minimumAllowedEdgeLength ||
          length > maximumAllowedEdgeLength) {
        return false;
      }
    }
  }

  // If we made it here, then all of the incident edge lengths were acceptable.
  return true;
}


// Move the vertices onto the clipping curves.
template<typename T>
inline
void
moveVertices(geom::IndSimpSetIncAdj<3, 2, T>* mesh,
             std::vector<int>* signOfDistance,
             const std::vector< std::pair<int, int> >& crossingEdges,
             const std::vector<typename geom::Ball<T, 3>::Point>& closestPoints,
             const std::vector<T>& distances,
             const std::vector<bool>& isOnBoundary,
             const T minimumAllowedEdgeLength,
             const T maximumAllowedEdgeLength)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;

  // If we are not allowing any movement.
  if (minimumAllowedEdgeLength == maximumAllowedEdgeLength) {
    return;
  }

  //
  // First process the boundary edges.
  //
  std::vector<bool> isMovable(mesh->vertices.size(), true);
  std::size_t neg, pos;
  typename Mesh::Vertex oldPosition;
  // For each crossing edge.
  for (std::size_t i = 0; i != crossingEdges.size(); ++i) {
    // Skip the interior edges.
    if (!(isOnBoundary[crossingEdges[i].first] &&
          isOnBoundary[crossingEdges[i].second])) {
      continue;
    }
    // If the edge no longer crosses the clipping curve, skip it.
    if ((*signOfDistance)[crossingEdges[i].first] *
        (*signOfDistance)[crossingEdges[i].second] != -1) {
      continue;
    }
    // Record which vertex has positive distance and which has
    // negative distance.
    if ((*signOfDistance)[crossingEdges[i].first] == -1) {
      neg = crossingEdges[i].first;
      pos = crossingEdges[i].second;
    }
    else {
      neg = crossingEdges[i].second;
      pos = crossingEdges[i].first;
    }
    // CONTINUE: reconsider this.
    // Since this is a boundary edge, we try to move the positive distance
    // vertex onto the curve.
    // Record the old position and move the vertex.
    oldPosition = mesh->vertices[pos];
    mesh->vertices[pos] = closestPoints[pos];
    // If the edge lengths are acceptable.
    if (minimumAllowedEdgeLength == 0 ||
        areEdgeLengthsAcceptable(mesh, pos, minimumAllowedEdgeLength,
                                 maximumAllowedEdgeLength)) {
      // Accept the move.
      (*signOfDistance)[pos] = 0;
    }
    // If the edge lengths are not acceptable.
    else {
      // Reject the move.
      mesh->vertices[pos] = oldPosition;
      // We indicate that this vertex may not be moved.  Otherwise
      // it may be moved using an interior crossing edge below.
      isMovable[pos] = false;
    }
  }

  //
  // Next process the interior edges.
  //
  // For each crossing edge.
  for (std::size_t i = 0; i != crossingEdges.size(); ++i) {
    // If the edge no longer crosses the clipping curve, skip it.
    if ((*signOfDistance)[crossingEdges[i].first] *
        (*signOfDistance)[crossingEdges[i].second] != -1) {
      continue;
    }
    // Record which vertex has positive distance and which has
    // negative distance.
    if ((*signOfDistance)[crossingEdges[i].first] == -1) {
      neg = crossingEdges[i].first;
      pos = crossingEdges[i].second;
    }
    else {
      neg = crossingEdges[i].second;
      pos = crossingEdges[i].first;
    }
    // If we are dealing with an interior edge and the negative
    // distance vertex is on the boundary, we can't move the negative
    // distance vertex.  Thus we move the positive distance vertex.
    if (isOnBoundary[neg]) {
      // CONTINUE
#if 0
      std::cout << "On Boundary.\n"
                << "  " << pos << "\n"
                << "  " << distances[neg] << "  "
                <<  distances[pos] << "\n"
                << "  " << mesh->getVertices()[pos] << "\n"
                << "  " << closestPoints[pos] << "\n";
#endif
      if (! isMovable[pos]) {
        continue;
      }
      // Record the old position and move the vertex.
      oldPosition = mesh->vertices[pos];
      mesh->vertices[pos] = closestPoints[pos];
      // If the edge lengths are acceptable.
      if (minimumAllowedEdgeLength == 0 ||
          areEdgeLengthsAcceptable(mesh, pos, minimumAllowedEdgeLength,
                                   maximumAllowedEdgeLength)) {
        // Accept the move.
        (*signOfDistance)[pos] = 0;
      }
      // If the edge lengths are not acceptable.
      else {
        // Reject the move.
        mesh->vertices[pos] = oldPosition;
      }
    }
    else {
      // If the negative distance vertex is closer to the curve.
      if (distances[neg] < distances[pos]) {
        // CONTINUE
#if 0
        std::cout << "Negative distance is closer.\n"
                  << "  " << distances[neg] << "  "
                  <<  distances[pos] << "\n"
                  << "  " << neg << "\n"
                  << "  " << mesh->getVertices()[neg] << "\n"
                  << "  " << closestPoints[neg] << "\n";
#endif
        if (! isMovable[neg]) {
          continue;
        }
        // Record the old position and move the vertex.
        oldPosition = mesh->vertices[neg];
        mesh->vertices[neg] = closestPoints[neg];
        // If the edge lengths are acceptable.
        if (minimumAllowedEdgeLength == 0 ||
            areEdgeLengthsAcceptable(mesh, neg, minimumAllowedEdgeLength,
                                     maximumAllowedEdgeLength)) {
          // Accept the move.
          (*signOfDistance)[neg] = 0;
        }
        // If the edge lengths are not acceptable.
        else {
          // Reject the move.
          mesh->vertices[neg] = oldPosition;
        }
      }
      // Otherwise the positive distance vertex is closer to the curve.
      else {
        // CONTINUE
#if 0
        std::cout << "Positive distance is closer.\n"
                  << "  " << distances[neg] << "  "
                  <<  distances[pos] << "\n"
                  << "  " << pos << "\n"
                  << "  " << mesh->getVertices()[pos] << "\n"
                  << "  " << closestPoints[pos] << "\n";
#endif
        if (! isMovable[pos]) {
          continue;
        }
        // Record the old position and move the vertex.
        oldPosition = mesh->vertices[pos];
        mesh->vertices[pos] = closestPoints[pos];
        // If the edge lengths are acceptable.
        if (minimumAllowedEdgeLength == 0 ||
            areEdgeLengthsAcceptable(mesh, pos, minimumAllowedEdgeLength,
                                     maximumAllowedEdgeLength)) {
          // Accept the move.
          (*signOfDistance)[pos] = 0;
        }
        // If the edge lengths are not acceptable.
        else {
          // Reject the move.
          mesh->vertices[pos] = oldPosition;
        }
      }
    }
  }
}




// Clip the mesh to remove non-visible portions.
// Vertices with negative distance are visible.
// Only perform the clipping if the resulting edge lengths are within the
// specified range.
// CONTINUE: Deal with special cases, like only zero/one exterior vertex or
// only zero/one interior vertex.
template<typename T>
inline
void
clipWithRubberClipping(const geom::Ball<T, 3>& atom,
                       const geom::Ball<T, 3>& clippingAtom,
                       geom::IndSimpSetIncAdj<3, 2, T>* mesh,
                       std::vector<int>* signOfDistance,
                       const T minimumAllowedEdgeLength,
                       const T maximumAllowedEdgeLength,
                       const bool areUsingCircularEdges)
{
  // A Cartesian point.
  typedef typename geom::Ball<T, 3>::Point Point;
  // The mesh type.
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;
  // An iterator over 1-faces (edges) in the mesh.
  typedef typename Mesh::FaceIterator FaceIterator;
  // An indexed simplex.
  typedef typename Mesh::IndexedSimplex IndexedSimplex;
  // A pair of vertex indices define an edge.
  typedef std::pair<int, int> Edge;

  // The simplex dimension.
  const std::size_t M = 2;

  // CONTINUE
#if 0
  std::cout << "\n";
  geom::writeAscii(std::cout, *mesh);
  std::cout << "\n";
  for (std::size_t i = 0; i != signOfDistance->size(); ++i) {
    std::cout << i << " " << (*signOfDistance)[i] << "\n";
  }
#endif

  //
  // Get the edges that cross the clipping surface.
  //
  std::vector<Edge> crossingEdges;
  std::vector<bool> isEdgeOnBoundary;
  {
    std::size_t simplexIndex, localIndex, vertexIndex1, vertexIndex2;
    // For each edge.
    for (FaceIterator i = mesh->getFacesBeginning(); i != mesh->getFacesEnd();
         ++i) {
      simplexIndex = i->first;
      localIndex = i->second;
      vertexIndex1 =
        mesh->indexedSimplices[simplexIndex][(localIndex + 1) % (M + 1)];
      vertexIndex2 =
        mesh->indexedSimplices[simplexIndex][(localIndex + 2) % (M + 1)];
      // If the edge crosses the clipping curve.
      if ((*signOfDistance)[vertexIndex1] *
          (*signOfDistance)[vertexIndex2] == -1) {
        // Record the edge.
        crossingEdges.push_back(Edge(vertexIndex1, vertexIndex2));
        // Record if the edge is on the boundary.
        isEdgeOnBoundary.push_back(mesh->isOnBoundary(i));
      }
    }
  }

  // Make the circle that is the intersection of the two spheres.
  geom::Circle3<T> circle;
  makeIntersectionCircle(atom, clippingAtom, &circle);
  // CONTINUE
  //std::cout << "\n" << circle << "\n";

  //
  // For the vertices of crossing edges, compute the unsigned distance
  // to the circle and the closest point on the circle.
  //

  // Although we will only compute the quantities for the vertices of
  // crossing edges, make arrays with elements for all of the vertices.
  std::vector<Point> closestPoints(mesh->vertices.size());
  const T Infinity = std::numeric_limits<T>::max();
  std::vector<T> distances(mesh->vertices.size(), Infinity);
  std::vector<bool> isOnBoundary(mesh->vertices.size(), false);
  geom::CircularArc3<T> circularArc;
  std::size_t sourceIndex, targetIndex;
  Point closestPoint;
  T distance;
  // For each crossing edge.
  for (std::size_t i = 0; i != crossingEdges.size(); ++i) {
    // The source and target vertex of the edge.
    sourceIndex = crossingEdges[i].first;
    targetIndex = crossingEdges[i].second;

    // If this is a boundary edge.
    if (isEdgeOnBoundary[i]) {
      if (areUsingCircularEdges) {
        // Interpret the boundary edge as a circular arc.  Make the circular
        // arc from the sphere center and the source and target vertices.
        circularArc.make(atom.center, mesh->vertices[sourceIndex],
                         mesh->vertices[targetIndex]);
        // Compute the closest point on the circle from the edge.
        geom::computeClosestPoint(circle, circularArc, &closestPoint);
      }
      else {
        // Otherwise, use a straight line edge for computing the closest point.
        geom::computeClosestPoint(circle, mesh->vertices[sourceIndex],
                                  mesh->vertices[targetIndex],
                                  &closestPoint);
      }
      // Compute the distance for the source vertex using the closest point.
      distance = ext::euclideanDistance(mesh->vertices[sourceIndex],
                                        closestPoint);
      // If the distance is smaller than any we have found before.
      if (distance < distances[sourceIndex]) {
        distances[sourceIndex] = distance;
        closestPoints[sourceIndex] = closestPoint;
      }
      // Compute the distance for the target vertex using the closest point.
      distance = ext::euclideanDistance(mesh->vertices[targetIndex],
                                   closestPoint);
      // If the distance is smaller than any we have found before.
      if (distance < distances[targetIndex]) {
        distances[targetIndex] = distance;
        closestPoints[targetIndex] = closestPoint;
      }
      // The vertices are on the boundary.
      isOnBoundary[sourceIndex] = isOnBoundary[targetIndex] = true;
    }
    // Otherwise, we have an interior edge.
    else {
      // Compute distance and closest points for the vertices of the edge.

      // If we have not yet computed the distance/closest point for the source.
      if (distances[sourceIndex] == Infinity) {
        // Compute the closest point.
        geom::computeClosestPoint(circle, mesh->vertices[sourceIndex],
                                  &closestPoints[sourceIndex]);
        // Compute the distance using the closest point.
        distances[sourceIndex] =
          geom::computeDistance(mesh->vertices[sourceIndex],
                                closestPoints[sourceIndex]);
        // Determine if the vertex is on the boundary.
        isOnBoundary[sourceIndex] = mesh->isVertexOnBoundary(sourceIndex);
      }
      // If we have not yet computed the distance/closest point for the target.
      if (distances[targetIndex] == Infinity) {
        // Compute the closest point.
        geom::computeClosestPoint(circle, mesh->vertices[targetIndex],
                                  &closestPoints[targetIndex]);
        // Compute the distance using the closest point.
        distances[targetIndex] =
          geom::computeDistance(mesh->vertices[targetIndex],
                                closestPoints[targetIndex]);
        // Determine if the vertex is on the boundary.
        isOnBoundary[targetIndex] = mesh->isVertexOnBoundary(targetIndex);
      }
    }
  }


  //
  // Sort the crossing edges by distance.
  // Note: This makes sense because it minimizes the distortion to the mesh.
  // It makes the rubber clipping a greedy method.  However, in practice this
  // has little effect.  I may want to reconsider this.
  //
  {
    std::vector<T> crossingEdgeDistances(crossingEdges.size());
    for (std::size_t i = 0; i != crossingEdges.size(); ++i) {
      // Record the minimum of the source distance and the target distance.
      crossingEdgeDistances[i] = std::min(distances[crossingEdges[i].first],
                                          distances[crossingEdges[i].second]);
    }
    ads::sortTogether(crossingEdgeDistances.begin(),
                      crossingEdgeDistances.end(),
                      crossingEdges.begin(), crossingEdges.end());
  }

  //
  // Perform the first part of the clipping by moving vertices onto the
  // clipping curve.
  //
  moveVertices(mesh, signOfDistance, crossingEdges,
               closestPoints, distances, isOnBoundary,
               minimumAllowedEdgeLength, maximumAllowedEdgeLength);

  //
  // Perform the second part of the clipping by removing simplices.
  //

  // Build the set of simplices to keep.
  std::vector<std::size_t> keep;
  typename Mesh::Vertex centroid;
  // For each simplex.
  for (std::size_t i = 0; i != mesh->indexedSimplices.size(); ++i) {
    const IndexedSimplex& s = mesh->indexedSimplices[i];
    // If any of the vertices are on the inside.
    if ((*signOfDistance)[s[0]] < 0 || (*signOfDistance)[s[1]] < 0 ||
        (*signOfDistance)[s[2]] < 0) {
      keep.push_back(i);
    }
    // If all of the vertices are on the clipping curve.
    else if ((*signOfDistance)[s[0]] <= 0 && (*signOfDistance)[s[1]] <= 0 &&
             (*signOfDistance)[s[2]] <= 0) {
      // If the centroid has negative distance.
      geom::getCentroid(*mesh, i, &centroid);
      if (clippingAtom.radius -
          geom::computeDistance(clippingAtom.center, centroid) < 0) {
        keep.push_back(i);
      }
    }
  }

  // If not all of the simplices will be kept.
  if (keep.size() != mesh->indexedSimplices.size()) {
    // Build the mesh from the subset of simplices.
    Mesh m;
    geom::buildFromSubsetSimplices(*mesh, keep.begin(), keep.end(), &m);
    mesh->swap(m);
  }

  // CONTINUE
#if 0
  {
    T min, max, mean;
    geom::computeContentStatistics(*mesh, &min, &max, &mean);
    if (max > 1) {
      std::cout << "\nMaximum content = " << max << "\n";
      geom::writeAscii(std::cout, *mesh);
    }
  }
#endif
}


// Clip the mesh.  Return true if any clipping was done.
// Only do the clipping if a vertex penetrates the clipping atom by
// tolerance times the meshed atom's radius.
// Only perform the clipping if the resulting edge lengths are within the
// specified range.
template<typename T>
inline
bool
clipWithRubberClipping(const geom::Ball<T, 3>& atom,
                       const geom::Ball<T, 3>& clippingAtom,
                       geom::IndSimpSetIncAdj<3, 2, T>* mesh,
                       const T tolerance,
                       const T minimumAllowedEdgeLength,
                       const T maximumAllowedEdgeLength,
                       const bool areUsingCircularEdges)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;

  assert(tolerance >= 0);

  // Compute the negative of the signed distance from the vertices to the
  // clipping atom.  Then the vertices with negative distance are visible.
  std::vector<T> distance(mesh->vertices.size());
  for (std::size_t i = 0; i != distance.size(); ++i) {
    distance[i] = clippingAtom.radius
                  - geom::computeDistance(clippingAtom.center,
                                          mesh->vertices[i]);
  }

  // If the mesh does not significantly penetrate the clipping atom.
  if (*std::max_element(distance.begin(), distance.end()) <=
      tolerance * atom.radius) {
    // Do not perform any clipping.
    return false;
  }

  // Make an array of the sign of the distance.
  std::vector<int> signOfDistance(distance.size());
  for (std::size_t i = 0; i != distance.size(); ++i) {
    signOfDistance[i] = ads::sign(distance[i]);
  }

  // The number of vertices outside the visible domain.
  const std::size_t numOutside = std::count(signOfDistance.begin(),
                                 signOfDistance.end(), 1);

  // If all of the vertices are visible.
  if (numOutside == 0) {
    // Do not alter the mesh.
    return false;
  }
  // If none of the vertices are visible.
  if (numOutside == 0) {
    // Clear the mesh.
    *mesh = Mesh();
  }
  else {
    // Otherwise, clip the mesh.
    clipWithRubberClipping(atom, clippingAtom, mesh, &signOfDistance,
                           minimumAllowedEdgeLength, maximumAllowedEdgeLength,
                           areUsingCircularEdges);
  }
  return true;
}



// Compute the clipping distances.  Sort the atoms (and identifiers) by
// this distance.
template<typename T>
inline
void
computeClippingDistances(const geom::Ball<T, 3>& atom,
                         std::vector<std::size_t>* clippingIdentifiers,
                         std::vector< geom::Ball<T, 3> >* clippingAtoms,
                         std::vector<T>* clippingDistances)
{
  // Make sure the vectors are the correct size.
  assert(clippingIdentifiers->size() == clippingAtoms->size() &&
         clippingIdentifiers->size() == clippingDistances->size());

  // Compute the distance.
  for (std::size_t i = 0; i != clippingAtoms->size(); ++i) {
    (*clippingDistances)[i] =
      computeClippingPlaneDistance(atom, (*clippingAtoms)[i]);
  }
  // CONTINUE: Write a sortTogether for triples.
  {
    std::vector<T> tmp(*clippingDistances);
    // Sort the clipping identifiers by distance.
    ads::sortTogether(tmp.begin(), tmp.end(),
                      clippingIdentifiers->begin(),
                      clippingIdentifiers->end());
  }
  // Sort the clipping atoms by distance.
  ads::sortTogether(clippingDistances->begin(), clippingDistances->end(),
                    clippingAtoms->begin(), clippingAtoms->end());
}




// Tesselate the atom.  Return the target edge length.
template<typename T>
inline
T
tesselateAtom(const geom::Ball<T, 3>& atom,
              const T edgeLengthSlope, const T edgeLengthOffset,
              geom::IndSimpSetIncAdj<3, 2, T>* mesh)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;
  typedef typename Mesh::VertexIterator VertexIterator;

  // The edge length function is
  // edgeLengthSlope * atom.getRadius() + edgeLengthOffset.
  // We divide by the radius to get the edge length for the unit sphere.

  // Tesselate the unit sphere.
  const T edgeLength = edgeLengthSlope + edgeLengthOffset / atom.radius;
  tesselateUnitSphere(edgeLength, mesh);
  // Scale and translate to map to the atom.
  for (VertexIterator i = mesh->vertices.begin();
       i != mesh->vertices.end(); ++i) {
    *i *= atom.radius;
    *i += atom.center;
  }
  return edgeLength;
}



// Tesselate the atom.  Return the target edge length.
template<typename T>
inline
T
tesselateAtom(const geom::Ball<T, 3>& atom,
              const int refinementLevel,
              geom::IndSimpSetIncAdj<3, 2, T>* mesh)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;
  typedef typename Mesh::VertexIterator VertexIterator;

  // Tesselate the unit sphere.
  tesselateUnitSphere(refinementLevel, mesh);
  // Scale and translate to map to the atom.
  for (VertexIterator i = mesh->vertices.begin(); i != mesh->vertices.end();
       ++i) {
    *i *= atom.radius;
    *i += atom.center;
  }
  // For the target edge length, use the first edge of the first triangle.
  return geom::computeDistance(mesh->getSimplexVertex(0, 0),
                               mesh->getSimplexVertex(0, 1));
}


// Compute the clipping atoms and form the initial tesselation.
// If the atom is completely removed by the clipping, the initial
// tesselation is an empty mesh.  Otherwise, we use tesselateAtom() to
// generate the initial tesselation.
// Return the target edge length.
template<typename T, typename IntOutputIterator>
inline
T
computeClippingAtomsAndTesselate
(const geom::Ball<T, 3>& atom,
 std::vector<std::size_t>& clippingIdentifiers,
 std::vector<geom::Ball<T, 3> >& clippingAtoms,
 const T edgeLengthSlope,
 const T edgeLengthOffset,
 const int refinementLevel,
 geom::IndSimpSetIncAdj<3, 2, T>* mesh,
 IntOutputIterator actuallyClip)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;

  // Compute the clipping distances.
  std::vector<T> clippingDistances(clippingAtoms.size());
  computeClippingDistances(atom, &clippingIdentifiers,
                           &clippingAtoms, &clippingDistances);

  // If the atom's surface is erased.
  if (! clippingDistances.empty() &&
      clippingDistances.front() == - std::numeric_limits<T>::max()) {
    // Make an empty mesh.
    *mesh = Mesh();
    // Record the clipping atom.
    *actuallyClip++ = clippingIdentifiers.front();
    // The target edge length is meaningless, because the mesh is empty.
    return 0;
  }

  // Tesselate the atom.
  if (refinementLevel >= 0) {
    return tesselateAtom(atom, refinementLevel, mesh);
  }
  // else
  return tesselateAtom(atom, edgeLengthSlope, edgeLengthOffset, mesh);
}





template<typename T, typename IntOutputIterator>
inline
void
clipWithRubberClipping
(const geom::Ball<T, 3>& atom,
 std::vector<std::size_t>& clippingIdentifiers,
 std::vector<geom::Ball<T, 3> >& clippingAtoms,
 geom::IndSimpSetIncAdj<3, 2, T>* mesh,
 IntOutputIterator actuallyClip,
 const T maximumStretchFactor,
 const bool areUsingCircularEdges)
{
  assert(0 <= maximumStretchFactor && maximumStretchFactor <= 1);

  // Determine the minimum and maximum allowed edge lengths.
  // Initially assume infinite stretching is allowed.
  T minimumAllowedEdgeLength = 0,
    maximumAllowedEdgeLength = std::numeric_limits<T>::max();
  // If we do not allow infinite stretching.
  if (maximumStretchFactor != 0) {
    const T meanEdgeLength = geom::computeMeanEdgeLength(*mesh);
    minimumAllowedEdgeLength = meanEdgeLength * maximumStretchFactor;
    maximumAllowedEdgeLength = meanEdgeLength / maximumStretchFactor;
  }

  // CONTINUE: Used for debugging.
  //static std::size_t count = 0;

  // For each clipping atom, and while we have not erased the mesh
  // with clipping.
  const T tolerance = 0;
  for (std::size_t i = 0; i != clippingAtoms.size() &&
       mesh->indexedSimplices.size() != 0; ++i) {
    // See if we clip the mesh using that atom.
    if (clipWithRubberClipping(atom, clippingAtoms[i], mesh, tolerance,
                               minimumAllowedEdgeLength,
                               maximumAllowedEdgeLength,
                               areUsingCircularEdges)) {
      // CONTINUE: Used for debugging.
#if 0
      std::ostringstream name;
      name << "clipped" << count++ << ".txt";
      std::ofstream out(name.str().c_str());
      geom::writeAscii(out, *mesh);
#endif
      // Record the clipping atom.
      *actuallyClip++ = clippingIdentifiers[i];
    }
  }
}



template<typename T, typename IntOutputIterator>
inline
T
triangulateVisibleSurfaceWithRubberClipping
(const geom::Ball<T, 3>& atom,
 std::vector<std::size_t>& clippingIdentifiers,
 std::vector<geom::Ball<T, 3> >& clippingAtoms,
 const T edgeLengthSlope,
 const T edgeLengthOffset,
 const int refinementLevel,
 geom::IndSimpSetIncAdj<3, 2, T>* mesh,
 IntOutputIterator actuallyClip,
 const T maximumStretchFactor,
 const bool areUsingCircularEdges)
{
  // Compute the clipping atoms and form the initial tesselation.
  const T targetEdgeLength =
    computeClippingAtomsAndTesselate(atom, clippingIdentifiers,
                                     clippingAtoms, edgeLengthSlope,
                                     edgeLengthOffset, refinementLevel,
                                     mesh, actuallyClip);

  // If the atom is completely erased by the clipping.
  if (mesh->indexedSimplices.size() == 0) {
    return targetEdgeLength;
  }

  clipWithRubberClipping(atom, clippingIdentifiers, clippingAtoms, mesh,
                         actuallyClip, maximumStretchFactor,
                         areUsingCircularEdges);

  return targetEdgeLength;
}


} // namespace mst
}
