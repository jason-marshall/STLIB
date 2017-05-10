// -*- C++ -*-

#if !defined(__mst_MolecularSurface_ipp__)
#error This file is an implementation detail of MolecularSurface.
#endif

namespace stlib
{
namespace mst
{


//--------------------------------------------------------------------------
// Accessors.
//--------------------------------------------------------------------------


template<typename T>
template<typename IntOutputIterator>
inline
void
MolecularSurface<T>::
getModifiedTriangleIndices(IntOutputIterator out) const
{
  for (std::set<std::size_t>::const_iterator i = _modifiedTriangles.begin();
       i != _modifiedTriangles.end(); ++i) {
    *out++ = *i;
  }
}


// Get each of the triangles in the surface.
// The output is a range of vertices.  Triples of vertices define the
// triangles.
template<typename T>
template<typename PointOutputIterator>
inline
void
MolecularSurface<T>::
getTriangleVertices(PointOutputIterator out)
{
  // For each triangle (non-null and null).
  for (std::size_t i = 0; i != getTrianglesSize(); ++i) {
    // If this is a non-null triangle.
    if (isTriangleNonNull(i)) {
      // Write the vertices of the triangle.
      *out++ = getTriangle(i)[0];
      *out++ = getTriangle(i)[1];
      *out++ = getTriangle(i)[2];
    }
  }
}




// Build the mesh for the specified atom.  Use rubber clipping.
template<typename T>
inline
void
MolecularSurface<T>::
buildMeshUsingRubberClipping(const std::size_t atomIdentifier,
                             geom::IndSimpSetIncAdj<3, 2, Number>* mesh) const
{
  // Get the specified atom.
  AtomType atom = _molecule.getAtom(atomIdentifier);
  // Get the atoms that might clip the mesh.
  std::vector<std::size_t> possibleIdentifiers;
  std::vector<AtomType> possibleAtoms;
  _molecule.getClippingAtoms(atomIdentifier,
                             std::back_inserter(possibleIdentifiers),
                             std::back_inserter(possibleAtoms));

  // Triangulate the visible surface.
  // Note: We don't need the clipping atoms, so we pass a trivial output
  // iterator.
  // Ignore the target edge length returned.
  triangulateVisibleSurfaceWithRubberClipping
  (atom, possibleIdentifiers, possibleAtoms,
   _edgeLengthSlope, _edgeLengthOffset, _refinementLevel, mesh,
   ads::constructTrivialOutputIterator(),
   _maximumStretchFactor, getAreUsingCircularEdges());
}



//--------------------------------------------------------------------------
// Manipulators.
//--------------------------------------------------------------------------


// CONTINUE: Track the atoms which actually clip each atom.  This is a
// smaller set than the affected atoms.

// Update the surface.  Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
updateSurface()
{
  // Clear the old the modified triangle indices.
  _modifiedTriangles.clear();

  // First clear all of the triangles that we are going to remove.
  // This makes room for the new triangles.

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // Clear the old triangulation.
    clearAtom(*i);
  }

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // If the atom is currently in the molecule.
    if (hasAtom(*i)) {
      // Triangulate the atom's visible surface.
      triangulateAtom(*i);
    }
  }

  // Clear the affected atoms.
  _affectedAtoms.clear();
}



// Update the surface using the bucket of triangles approach.  Update the
// set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
updateSurfaceUsingBot()
{
  // Clear the old the modified triangle indices.
  _modifiedTriangles.clear();

  // First clear all of the triangles that we are going to remove.
  // This makes room for the new triangles.

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // Clear the old triangulation.
    clearAtom(*i);
  }

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // If the atom is currently in the molecule.
    if (hasAtom(*i)) {
      // Triangulate the atom's visible surface.
      triangulateAtomWithBot(*i);
    }
  }

  // Clear the affected atoms.
  _affectedAtoms.clear();
}



// Update the surface using cut clipping.  Update the set of modified
// triangles.
template<typename T>
inline
void
MolecularSurface<T>::
updateSurfaceWithCutClipping()
{
  // Clear the old the modified triangle indices.
  _modifiedTriangles.clear();

  // First clear all of the triangles that we are going to remove.
  // This makes room for the new triangles.

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // Clear the old triangulation.
    clearAtom(*i);
  }

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // If the atom is currently in the molecule.
    if (hasAtom(*i)) {
      // Triangulate the atom's visible surface.
      triangulateAtomWithCutClipping(*i);
    }
  }

  // Clear the affected atoms.
  _affectedAtoms.clear();
}



// Update the surface using cut clipping.  Update the set of modified
// triangles.
template<typename T>
inline
void
MolecularSurface<T>::
updateSurfaceWithRubberClipping()
{
  // Clear the old the modified triangle indices.
  _modifiedTriangles.clear();

  // First clear all of the triangles that we are going to remove.
  // This makes room for the new triangles.

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // Clear the old triangulation.
    clearAtom(*i);
  }

  // For each affected atom.
  for (IdentifierSet::const_iterator i = _affectedAtoms.begin();
       i != _affectedAtoms.end(); ++i) {
    // If the atom is currently in the molecule.
    if (hasAtom(*i)) {
      // Triangulate the atom's visible surface.
      triangulateAtomWithRubberClipping(*i);
    }
  }

  // Clear the affected atoms.
  _affectedAtoms.clear();
}




// Print information about the triangulation.
template<typename T>
inline
void
MolecularSurface<T>::
printInformation(std::ostream& out) const
{
  out << "There are " << _atomTriangleIndices.size() << " atoms.\n";
  // For each atom.
  for (typename AtomTriangleMap::const_iterator i =
         _atomTriangleIndices.begin(); i != _atomTriangleIndices.end(); ++i) {
    // Print the number of triangles for the atom.
    out << "Atom " << i->first << " has " << i->second.size()
        << " triangles.\n";
  }
}



// Compute triangulation error statistics.
template<typename T>
inline
void
MolecularSurface<T>::
computeErrorStatistics(Number* minimumRadiusDeviation,
                       Number* maximumRadiusDeviation,
                       Number* meanRadiusDeviation,
                       Number* minimumPenetration,
                       Number* maximumPenetration,
                       Number* meanPenetration) const
{
  // Initialize.
  *minimumRadiusDeviation = std::numeric_limits<Number>::max();
  *maximumRadiusDeviation = 0;
  *meanRadiusDeviation = 0;
  *minimumPenetration = std::numeric_limits<Number>::max();
  *maximumPenetration = 0;
  *meanPenetration = 0;

  std::size_t i, atomIdentifier;
  Point vertex;
  Number radiusDeviation, distance, penetration;
  // For each triangle.
  for (std::size_t triangleIndex = 0; triangleIndex != _surface.size();
       ++triangleIndex) {
    // Skip the null triangles.
    if (_surface.isNull(triangleIndex)) {
      continue;
    }
    // The atom identifier associated with the triangle.
    atomIdentifier = _surface.get(triangleIndex).first;
    // The atom associated with the triangle.
    const AtomType& atom = _molecule.getAtom(atomIdentifier);
    // For each vertex.
    for (std::size_t vertexIndex = 0; vertexIndex != 3; ++vertexIndex) {
      // The triangle vertex.
      vertex = _surface.get(triangleIndex).second[vertexIndex];

      // The radius deviation.
      radiusDeviation =
        std::abs(ext::euclideanDistance(vertex, atom.center) -
                 atom.radius) / atom.radius;
      if (radiusDeviation < *minimumRadiusDeviation) {
        *minimumRadiusDeviation = radiusDeviation;
      }
      if (radiusDeviation > *maximumRadiusDeviation) {
        *maximumRadiusDeviation = radiusDeviation;
      }
      *meanRadiusDeviation += radiusDeviation;

      // The penetration.
      for (typename Molecule<Number>::IdentifierAndAtomConstIterator iter
           = _molecule.getIdentifiersAndAtomsBeginning();
           iter != _molecule.getIdentifiersAndAtomsEnd(); ++iter) {
        // The atom identifier.
        i = iter->first;
        // Skip the atom associated with the triangle.
        if (i == atomIdentifier) {
          continue;
        }
        const AtomType& a = iter->second;
        distance = ext::euclideanDistance(a.center, vertex);
        if (distance < a.radius) {
          penetration = (a.radius - distance) / a.radius;
        }
        else {
          penetration = 0;
        }
        if (penetration < *minimumPenetration) {
          *minimumPenetration = penetration;
        }
        if (penetration > *maximumPenetration) {
          *maximumPenetration = penetration;
        }
        *meanPenetration += penetration;
      }
    }
  }
  // Special case for the radius deviation: No atoms.
  if (_surface.size() == 0) {
    *minimumRadiusDeviation = 0;
  }
  // Special case for the penetration: No other atoms.
  else if (_surface.size() == 1) {
    *minimumPenetration = 0;
  }
  else {
    // Divide by the number of vertices.
    *meanRadiusDeviation /= (3 * _surface.size());
    *meanPenetration /= (3 * _surface.size());
  }
}



//--------------------------------------------------------------------------
// Private member functions.
//--------------------------------------------------------------------------


// Clear the triangulation for the specified atom.
// Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
clearAtom(const std::size_t identifier)
{
  // Get the old triangle indices for this atom.
  typename AtomTriangleMap::iterator i = _atomTriangleIndices.find(identifier);
  // If there is an old triangulation.
  if (i != _atomTriangleIndices.end()) {
    IndexContainer& indices = i->second;

    // Since we are removing them, add the old triangles to the modified set.
    _modifiedTriangles.insert(indices.begin(), indices.end());

    // Remove the old triangles from the surface.
    _surface.erase(indices.begin(), indices.end());

    // Clear the old triangulation.
    indices.clear();
  }
}


// Triangulate the visible surface for the specified atom.
// Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
triangulateAtom(const std::size_t identifier)
{
  // Get the specified atom.
  AtomType atom = _molecule.getAtom(identifier);
  // Get the atoms that might clip the mesh.
  std::vector<std::size_t> possibleIdentifiers;
  std::vector<AtomType> possibleAtoms;
  _molecule.getClippingAtoms(identifier,
                             std::back_inserter(possibleIdentifiers),
                             std::back_inserter(possibleAtoms));

  // Triangulate the visible surface.
  // Record the clipping atoms.
  std::vector<std::size_t> clippingAtoms;
  IndexedMesh mesh;
  const Number targetEdgeLength =
    triangulateVisibleSurface(atom, possibleIdentifiers, possibleAtoms,
                              _edgeLengthSlope, _edgeLengthOffset,
                              _refinementLevel, &mesh,
                              std::back_inserter(clippingAtoms),
                              _maximumStretchFactor,
                              _epsilonForCutClipping,
                              getAreUsingCircularEdges());
  // Update the clipped atoms.
  for (std::vector<std::size_t>::const_iterator i = clippingAtoms.begin();
       i != clippingAtoms.end(); ++i) {
    _clippedAtoms[*i].insert(identifier);
  }

  // Add the triangles to the surface and update the set of modified triangles.
  insertTriangles(identifier, mesh.getSimplicesBegin(),
                  mesh.getSimplicesEnd(), targetEdgeLength);
}



// Triangulate the visible surface for the specified atom with a bucket of
// triangles.  Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
triangulateAtomWithBot(const std::size_t identifier)
{
  // Get the specified atom.
  AtomType atom = _molecule.getAtom(identifier);
  // Get the atoms that might clip the mesh.
  std::vector<std::size_t> possibleIdentifiers;
  std::vector<AtomType> possibleAtoms;
  _molecule.getClippingAtoms(identifier,
                             std::back_inserter(possibleIdentifiers),
                             std::back_inserter(possibleAtoms));

  // Triangulate the visible surface.
  // Record the clipping atoms.
  std::vector<std::size_t> clippingAtoms;
  std::vector<Triangle> triangles;
  const Number targetEdgeLength =
    triangulateVisibleSurfaceWithBot(atom, possibleIdentifiers, possibleAtoms,
                                     _edgeLengthSlope, _edgeLengthOffset,
                                     _refinementLevel,
                                     std::back_inserter(triangles),
                                     std::back_inserter(clippingAtoms));
  // Update the clipped atoms.
  for (std::vector<std::size_t>::const_iterator i = clippingAtoms.begin();
       i != clippingAtoms.end(); ++i) {
    _clippedAtoms[*i].insert(identifier);
  }

  // Add the triangles to the surface and update the set of modified triangles.
  insertTriangles(identifier, triangles.begin(), triangles.end(),
                  targetEdgeLength);
}



// Triangulate the visible surface for the specified atom with cut clipping.
// Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
triangulateAtomWithCutClipping(const std::size_t identifier)
{
  // Get the specified atom.
  AtomType atom = _molecule.getAtom(identifier);
  // Get the atoms that might clip the mesh.
  std::vector<std::size_t> possibleIdentifiers;
  std::vector<AtomType> possibleAtoms;
  _molecule.getClippingAtoms(identifier,
                             std::back_inserter(possibleIdentifiers),
                             std::back_inserter(possibleAtoms));

  // Triangulate the visible surface.
  // Record the clipping atoms.
  std::vector<std::size_t> clippingAtoms;
  IndexedMesh mesh;
  const Number targetEdgeLength =
    triangulateVisibleSurfaceWithCutClipping
    (atom, possibleIdentifiers, possibleAtoms, _edgeLengthSlope,
     _edgeLengthOffset, _refinementLevel, &mesh,
     std::back_inserter(clippingAtoms), _epsilonForCutClipping);
  // Update the clipped atoms.
  for (std::vector<std::size_t>::const_iterator i = clippingAtoms.begin();
       i != clippingAtoms.end(); ++i) {
    _clippedAtoms[*i].insert(identifier);
  }

  // Add the triangles to the surface and update the set of modified triangles.
  insertTriangles(identifier, mesh.getSimplicesBegin(),
                  mesh.getSimplicesEnd(), targetEdgeLength);
}




// Triangulate the visible surface for the specified atom with rubber clipping.
// Update the set of modified triangles.
template<typename T>
inline
void
MolecularSurface<T>::
triangulateAtomWithRubberClipping(const std::size_t identifier)
{
  // Get the specified atom.
  AtomType atom = _molecule.getAtom(identifier);
  // Get the atoms that might clip the mesh.
  std::vector<std::size_t> possibleIdentifiers;
  std::vector<AtomType> possibleAtoms;
  _molecule.getClippingAtoms(identifier,
                             std::back_inserter(possibleIdentifiers),
                             std::back_inserter(possibleAtoms));

  // Triangulate the visible surface.
  // Record the clipping atoms.
  std::vector<std::size_t> clippingAtoms;
  IndexedMesh mesh;
  const Number targetEdgeLength =
    triangulateVisibleSurfaceWithRubberClipping
    (atom, possibleIdentifiers, possibleAtoms,
     _edgeLengthSlope, _edgeLengthOffset,
     _refinementLevel, &mesh,
     std::back_inserter(clippingAtoms),
     _maximumStretchFactor,
     getAreUsingCircularEdges());
  // Update the clipped atoms.
  for (std::vector<std::size_t>::const_iterator i = clippingAtoms.begin();
       i != clippingAtoms.end(); ++i) {
    _clippedAtoms[*i].insert(identifier);
  }

  // Add the triangles to the surface and update the set of modified triangles.
  insertTriangles(identifier, mesh.getSimplicesBegin(),
                  mesh.getSimplicesEnd(), targetEdgeLength);
}





template<typename T>
template<typename TriangleInputIterator>
inline
void
MolecularSurface<T>::
insertTriangles(const std::size_t identifier, TriangleInputIterator beginning,
                TriangleInputIterator end, const Number targetEdgeLength)
{
  //
  // Get the index container for this atom.
  //
  typename AtomTriangleMap::iterator i = _atomTriangleIndices.find(identifier);
  // If there was not an old triangulation.
  if (i == _atomTriangleIndices.end()) {
    // Insert a new triangulation index container.
    std::pair<typename AtomTriangleMap::iterator, bool> result =
      _atomTriangleIndices.insert(std::make_pair(identifier,
                                  IndexContainer()));
    // Make sure it was inserted.
    assert(result.second);
    i = result.first;
  }
  IndexContainer& indices = i->second;
  // The index container should be empty.
  assert(indices.empty());

  Number minimumAllowedAreaForThisAtom = 0;
  if (_minimumAllowedArea != 0) {
    const Number Sqrt3 = 1.73205080757;
    minimumAllowedAreaForThisAtom = _minimumAllowedArea * targetEdgeLength
                                    * targetEdgeLength * Sqrt3 / 2.0;
  }

  //
  // Add the triangles to the surface and update the set of modified triangles.
  //
  std::size_t index;
  IdentifierAndTriangle identifierAndTriangle;
  // For each triangle.
  for (; beginning != end; ++beginning) {
    identifierAndTriangle.first = identifier;
    identifierAndTriangle.second = *beginning;
    // If we are checking the area of the triangle.
    if (minimumAllowedAreaForThisAtom != 0) {
      const Triangle& t = identifierAndTriangle.second;
      // If the area of the triangle is too small.
      if (geom::computeContent(t[0], t[1], t[2]) <
          minimumAllowedAreaForThisAtom) {
        // Skip this triangle.
        continue;
      }
    }
    // Add the triangle to the surface.
    index = _surface.insert(identifierAndTriangle);
    // Add the triangle to this atom's set of triangle indices.
    indices.push_back(index);
    // Add the index to the modified triangle set.
    _modifiedTriangles.insert(index);
  }
}




// Add the atom identifiers that the specified atom will clip to the affected
// set.  This function is called when inserting the specified atom.
template<typename T>
inline
void
MolecularSurface<T>::
insertAtomsThatWillBeClipped(const std::size_t identifier)
{
  // From the molecule, get the atoms that might be clipped.
  std::vector<std::size_t> mightBeClipped;
  _molecule.getClippedAtoms(identifier, std::back_inserter(mightBeClipped));

  // Get the specified atom.
  const AtomType& atom = getAtom(identifier);
  // For each atom that might be clipped.
  for (std::vector<std::size_t>::const_iterator i = mightBeClipped.begin();
       i != mightBeClipped.end(); ++i) {
    // If the specified atom clips this one.
    if (doesClipMesh(*i, atom)) {
      // Insert the identifier.
      _affectedAtoms.insert(*i);
    }
  }
}



// Add the atom identifiers that clipped the specified atom to the affected
// set.  This function is called when erasing the specified atom.
template<typename T>
inline
void
MolecularSurface<T>::
insertAtomsThatWereClipped(const std::size_t identifier)
{
  //std::cerr << "begin insertAtomsThatWereClipped\n";
  //std::cerr << "identifier = " << identifier << "\n";
  //std::cerr << "_clippedAtoms.size() = " << _clippedAtoms.size() << "\n";
  // For each atom that clipped the specified atom.
  const IdentifierSet& clippedAtoms = _clippedAtoms[identifier];
  //std::cerr << "clippedAtoms.size() = " << clippedAtoms.size() << "\n";
  for (IdentifierSet::const_iterator i = clippedAtoms.begin();
       i != clippedAtoms.end(); ++i) {
    //std::cerr << "*i = " << *i << "\n";
    // Insert the identifier.
    _affectedAtoms.insert(*i);
  }
  //std::cerr << "end insertAtomsThatWereClipped\n";
}



// Return true if the atom clips the mesh of the specified atom.
template<typename T>
inline
bool
MolecularSurface<T>::
doesClipMesh(std::size_t identifier, const AtomType& atom) const
{
  // Get the triangle indices for the specified atom.
  AtomTriangleMap::const_iterator iter = _atomTriangleIndices.find(identifier);
  // If the specified atom has not yet been triangulated.
  if (iter == _atomTriangleIndices.end()) {
    return false;
  }
  // The indices.
  const IndexContainer& indices = iter->second;

  std::size_t index;
  // For each triangle index.
  for (IndexContainer::const_iterator i = indices.begin(); i != indices.end();
       ++i) {
    index = *i;
    // The triangle must be valid.
    assert(_surface.isNonNull(index));
    // For each vertex of the triangle.
    for (std::size_t m = 0; m != 3; ++m) {
      // If the vertex is inside the atom.
      if (isInside(atom, _surface.get(index).second[m])) {
        // The mesh is clipped.
        return true;
      }
    }
  }

  // If the mesh is not clipped, return false.
  return false;
}


} // namespace mst
}
