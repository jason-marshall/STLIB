// -*- C++ -*-

#if !defined(__mst_triangulateAtomBot_ipp__)
#error This file is an implementation detail of triangulateAtomBot.
#endif

namespace stlib
{
namespace mst
{


// Clip the triangle.
template<typename T, typename Triangle, typename TriangleOutputIterator>
inline
bool
clipTriangle(const geom::Ball<T, 3>& atom, const geom::Ball<T, 3>& clippingAtom,
             const Triangle& triangle,
             TriangleOutputIterator clippedTriangles)
{
  typedef std::array<T, 3> Point;

  // Compute the negative of the signed distance from the vertices to the
  // clipping atom.  Then the vertices with negative distance are visible.
  std::array<T, 3> distance;
  for (std::size_t i = 0; i != distance.size(); ++i) {
    distance[i] = clippingAtom.radius
                  - geom::computeDistance(clippingAtom.center, triangle[i]);
  }

  // Make an array of the sign of the distance.
  std::array<int, 3> signOfDistance;
  for (std::size_t i = 0; i != distance.size(); ++i) {
    signOfDistance[i] = (distance[i] <= 0 ? -1 : 1);
  }

  const int sos = ext::sum(signOfDistance);
  // If all three vertices are visible.
  if (sos == -3) {
    *clippedTriangles = triangle;
    ++clippedTriangles;
    return false;
  }
  // If two vertices are visible.
  else if (sos == -1) {
    //
    // Move the non-visible vertex to the third position.
    //
    Triangle t(triangle);
    if (signOfDistance[0] == 1) {
      std::swap(t[0], t[1]);
      std::swap(t[1], t[2]);
    }
    else if (signOfDistance[1] == 1) {
      std::swap(t[1], t[2]);
      // Do the second swap to preserve the orientation.
      std::swap(t[0], t[1]);
    }

    // Make the circle that is the intersection of the two spheres.
    geom::Circle3<T> circle;
    makeIntersectionCircle(atom, clippingAtom, &circle);

    // Calculate the intersection points.
    Point a, b;
    geom::computeClosestPoint(circle, t[1], t[2], &a);
    geom::computeClosestPoint(circle, t[2], t[0], &b);

    // The two triangles that compose the visible quadrilateral.
    // We choose the better of the two triangulations.

    // Compute the minimum quality of the first option.
    geom::SimplexModCondNum<2, T> modifiedConditionNumber;
    Triangle s(t);
    s[2] = a;
    modifiedConditionNumber.setFunction(s);
    T quality1 = 1.0 / modifiedConditionNumber();
    s[1] = a;
    s[2] = b;
    modifiedConditionNumber.setFunction(s);
    quality1 = std::min(quality1, 1.0 / modifiedConditionNumber());

    // Compute the minimum quality of the second option.
    s = t;
    s[2] = b;
    modifiedConditionNumber.setFunction(s);
    T quality2 = 1.0 / modifiedConditionNumber();
    s[0] = t[1];
    s[1] = a;
    s[2] = b;
    modifiedConditionNumber.setFunction(s);
    quality2 = std::min(quality2, 1.0 / modifiedConditionNumber());

    // Choose the better triangulation.
    if (quality1 > quality2) {
      Triangle s = t;
      s[2] = a;
      *clippedTriangles = s;
      ++clippedTriangles;
      s[1] = a;
      s[2] = b;
      *clippedTriangles = s;
      ++clippedTriangles;
    }
    else {
      s = t;
      s[2] = b;
      *clippedTriangles = s;
      ++clippedTriangles;
      s[0] = t[1];
      s[1] = a;
      s[2] = b;
      *clippedTriangles = s;
      ++clippedTriangles;
    }
    return true;
  }
  // If one vertex is visible.
  else if (sos == 1) {
    //
    // Move the visible vertex to the first position.
    //
    Triangle t(triangle);
    if (signOfDistance[1] == -1) {
      std::swap(t[0], t[1]);
      // Do the second swap to preserve the orientation.
      std::swap(t[1], t[2]);
    }
    else if (signOfDistance[2] == -1) {
      std::swap(t[1], t[2]);
      std::swap(t[0], t[1]);
    }

    // Make the circle that is the intersection of the two spheres.
    geom::Circle3<T> circle;
    makeIntersectionCircle(atom, clippingAtom, &circle);

    // Calculate the intersection points.
    Point a, b;
    geom::computeClosestPoint(circle, t[0], t[1], &a);
    geom::computeClosestPoint(circle, t[2], t[0], &b);

    // The visible portion of the triangle is also a triangle.
    t[1] = a;
    t[2] = b;
    *clippedTriangles = t;
    ++clippedTriangles;

    return true;
  }
  // Otherwise no vertices are visible.
  assert(sos == 3);
  return true;
}




// Clip the mesh.  Return true if any clipping was done.
template<typename T, typename Triangle>
inline
bool
clip(const geom::Ball<T, 3>& atom, const geom::Ball<T, 3>& clippingAtom,
     std::vector<Triangle>* triangles)
{
  bool result = false;

  std::vector<Triangle> newTriangles;
  // For each triangle.
  for (typename std::vector<Triangle>::const_iterator
       i = triangles->begin(); i != triangles->end(); ++i) {
    // Get the vertices.
    if (clipTriangle(atom, clippingAtom, *i,
                     std::back_inserter(newTriangles))) {
      result = true;
    }
  }

  // Swap to get the new vertices.
  triangles->swap(newTriangles);
  // Return true if any clipping was done.  Otherwise return false.
  return result;
}



template<typename T, typename TriangleOutputIterator,
         typename IntOutputIterator>
inline
T
triangulateVisibleSurfaceWithBot(const geom::Ball<T, 3>& atom,
                                 std::vector<std::size_t>& clippingIdentifiers,
                                 std::vector<geom::Ball<T, 3> >& clippingAtoms,
                                 const T edgeLengthSlope,
                                 const T edgeLengthOffset,
                                 const int refinementLevel,
                                 TriangleOutputIterator outputTriangles,
                                 IntOutputIterator actuallyClip)
{
  typedef geom::IndSimpSetIncAdj<3, 2, T> Mesh;
  typedef typename Mesh::Simplex Triangle;

  // Compute the clipping atoms and form the initial tesselation.
  Mesh mesh;
  const T targetEdgeLength =
    computeClippingAtomsAndTesselate(atom, clippingIdentifiers,
                                     clippingAtoms, edgeLengthSlope,
                                     edgeLengthOffset, refinementLevel,
                                     &mesh, actuallyClip);

  // If the atom is completely erased by the clipping.
  if (mesh.indexedSimplices.size() == 0) {
    return targetEdgeLength;
  }

  // Get the triangle vertices.
  std::vector<Triangle> triangles(mesh.getSimplicesBegin(),
                                  mesh.getSimplicesEnd());

  // For each clipping atom, and while we have not erased the mesh
  // with clipping.
  for (std::size_t i = 0; i != clippingAtoms.size() &&
       triangles.size() != 0; ++i) {
    // See if we clip the mesh using that atom.
    if (clip(atom, clippingAtoms[i], &triangles)) {
      // Record the clipping atom.
      *actuallyClip++ = clippingIdentifiers[i];
    }
  }

  // Write the output triangles.
  std::copy(triangles.begin(), triangles.end(), outputTriangles);

  return targetEdgeLength;
}


} // namespace mst
}
