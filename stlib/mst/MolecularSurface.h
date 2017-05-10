// -*- C++ -*-

/*!
  \file MolecularSurface.h
  \brief Class for generating a triangulation of the visible surface of a molecule.
*/

#if !defined(__mst_MolecularSurface_h__)
#define __mst_MolecularSurface_h__

#include "stlib/mst/Molecule.h"
#include "stlib/mst/triangulateAtom.h"
#include "stlib/mst/triangulateAtomBot.h"

#include "stlib/ads/array/ArrayWithNullHoles.h"

namespace stlib
{
namespace mst
{

//! Class for generating a triangulation of the visible surface of a molecule.
/*!
  This class uses Molecule to represent the geometry of a dynamically changing
  molecule.  It stores the locations, radii and identifiers for the atoms.
  It uses an orthogonal range query data structure to find intersections
  of the surfaces of the atoms.

  Triangulating the surface is done my tesselating atoms and then clipping the
  tesselations to triangulate the visible surface.  The tesselation is
  performed either with a specified maximum allowed edge length or with a
  specified level of refinement.  The maximum allowed edge length is a linear
  function of the atomic radius.  If a level of refinement is not specified,
  the edge length function will be used.  One can use either the constructor
  or manipulators to specify these two behaviors.

  Suppose that one wants to construct a Molecular surface and use a maximum
  allowed edge length function of 0.3 * r + 0.5.  We show how to do this
  using the constructor and then using the manipulators.

  \verbatim
  mst::MolecularSurface<double> molecularSurface(0.3, 0.5); \endverbatim

  \verbatim
  mst::MolecularSurface<double> molecularSurface;
  molecularSurface.setEdgeLengthSlope(0.3);
  molecularSurface.setEdgeLengthOffset(0.5); \endverbatim

  Next suppose that one wants to construct a Molecular surface and use a
  level 3 refinement (subdivided icosahedron) for tesselation.
  We show how to do this using the constructor and then using the
  manipulators.

  \verbatim
  mst::MolecularSurface<double> molecularSurface(0.0, 0.0, 3); \endverbatim

  \verbatim
  mst::MolecularSurface<double> molecularSurface;
  molecularSurface.setRefinementLevel(3); \endverbatim

  To insert atoms into the molecule, use one of the insert functions.
  If you would like this data structure to assign atom identifiers for you,
  use insert(const AtomType& atom).  This will return an identifiers that is
  the smallest non-negative, unused integer.  If you want to manage the
  identifiers yourself, use insert(const std::size_t identifier, const AtomType& atom).
  You can erase an atom through its identifier by calling
  erase(const std::size_t identifier).  There is no function for moving atoms.
  However, this can be accomplished by removing an atom and re-inserting
  it in a new location.  Note that inserting and erasing atoms
  does not immediately affect the triangulation.  For the sake of efficiency,
  this is done with a separate member function.  Below we insert some
  atoms with specified identifiers and then with automatically generated
  identifiers.  We store the identifiers in a std::vector.

  \verbatim
  std::vector<std::size_t> identifiers;
  molecularSurface.insert(101, Atom(Point(0.0, 0.0, 0.0), 1.0);
  identifiers.push_back(101);
  molecularSurface.insert(102, Atom(Point(1.0, 0.0, 0.0), 1.0);
  identifiers.push_back(102);
  identifiers.push_back(molecularSurface.insert(Atom(Point(2.0, 0.0, 0.0), 1.0));
  identifiers.push_back(molecularSurface.insert(Atom(Point(3.0, 0.0, 0.0), 1.0)); \endverbatim

  You can query if an atom with a specified identifier is in the molecule with
  hasAtom().  You can access a specified atom with getAtom().  Read access
  to the atom identifiers is given through getIdentifiersBeginning() and
  getIdentifiersEnd().  Below we print the identifiers, centers, and radii
  of each of the atoms in a molecule.

  \verbatim
  typedef mst::MolecularSurface<double>::IdentifierConstIterator IdentifierConstIterator;
  for (IdentifierConstIterator i = molecularSurface.getIdentifiersBeginning();
       i != molecularSurface.getIdentifiersEnd(); ++i) {
    assert(molecularSurface.hasAtom(*i));
    const Atom& atom = molecularSurface.getAtom(*i);
    std::cout << "Identifier = " << *i << "\n"
              << "Center = " << atom.getCenter() << "\n"
              << "Radius = " << atom.getRadius() << "\n";
  } \endverbatim

  After you have built the molecule or modified the molecule, and before
  you access the triangulation, you must ask this data structure to update
  the triangle mesh surface.  To update using rubber clipping, use
  updateSurface().  For cut clipping, use updateSurfaceWithCutClipping().
  These functions only update the portion of the surface that has changed.

  After updating the surface, you can access the modified triangles.
  getModifiedTriangleIndices() will output the indices of all of the triangles
  that were modified by the last update to the surface.  A triangle may
  have been added, moved, or removed.  Below we access the triangles
  that are modified by an update operation.

  \verbatim
  std::vector<Atom> atoms;
  // Build atoms.
  ...
  // Insert the atoms.
  for (std::size_t i = 0; i != atoms.size(); ++i) {
    molecularSurface.insert(i, atoms[i]);
  }
  // Update the surface.
  molecularSurface.updateSurface();
  // Get the modified triangles.
  std::vector<std::size_t> modifiedTriangleIndices;
  molecularSurface.getModifiedTriangleIndices(std::back_inserter(modifiedTriangleIndices));
  // Loop over the modified triangles.
  for (std::vector<std::size_t>::const_iterator i = modifiedTriangleIndices.begin();
       i != modifiedTriangleIndices.end(); ++i) {
    // If it is a new or modified triangle.
    if (molecularSurface.isTriangleNonNull(*i)) {
      // Do something.
    }
    // Otherwise the triangle has been removed from the mesh.
    else {
      // Do something.
    }
  } \endverbatim


  The triangles in the surface mesh are stored in an array with holes.
  (See ads::ArrayWithNullHoles if you are interested in the implementation.)
  When a triangle is removed from the surface, it becomes a hole in the
  array.  When a triangle is added to the surface it is inserted into
  the first available hole.  If there are no holes, it is appended to
  the end of the array.  You can get the total size of the array with
  getTrianglesSize().  This includes both the valid triangles and the
  null triangles (holes).  You can use getNullTrianglesSize() and
  getNonNullTrianglesSize() to determine the number of each.  To
  access an element of the triangle array, use getTriangle().
  If you access a hole in the array, it will return a triangle in which
  all of the coordinates are at the origin.  This null triangle is the
  flag for a hole.  Instead of checking the triangle yourself, it is better
  to use the accessors isTriangleNull() and isTriangleNonNull().

  If you want to collect all of the non-null triangles, use
  getTriangleVertices(PointOutputIterator out).  This will write the vertices
  of the non-null triangles to the output iterator.  The code below
  shows examples of this.

  \verbatim
  std::vector<Point> triangleVertices;
  molecularSurface.getTriangleVertices(std::back_inserter(triangleVertices)); \endverbatim

  \verbatim
  Point* triangleVertices = new Point[3 * molecularSurface.getNonNullTrianglesSize()];
  molecularSurface.getTriangleVertices(triangleVertices);
  ...
  delete[] triangleVertices; \endverbatim

  Recall that each triangle in the surface lies on the visible portion of the
  surface of a single atom.  (This is key in efficiently implementing the
  dynamic capability.)  For a specified triangle, you may want to access
  the atom to which it belongs.  This can be done with
  getAtomIdentifierForTriangle() or getAtomForTriangle().
*/
template<typename T>
class MolecularSurface
{
  //
  // Public types.
  //

public:

  //! The number type.
  typedef T Number;
  //! An atom.
  typedef geom::Ball<Number, 3> AtomType;
  //! A Cartesian point.
  typedef std::array<Number, 3> Point;
  //! A triangle is a 2-simplex of points.
  typedef std::array<Point, 3> Triangle;
  //! A const iterator over atom identifiers.
  typedef typename Molecule<Number>::IdentifierConstIterator
  IdentifierConstIterator;

  //
  // Private types.
  //

private:

  typedef std::vector<std::size_t> IndexContainer;
  typedef std::map<std::size_t, IndexContainer> AtomTriangleMap;

  typedef std::set<std::size_t> IndexSet;

  //! An atom identifier and a triangle.
  typedef std::pair<std::size_t, Triangle> IdentifierAndTriangle;
  //! The array of triangles (along with the atom identifiers to which they belong).
  typedef ads::ArrayWithNullHoles<IdentifierAndTriangle> TriangleArray;

  typedef geom::IndSimpSetIncAdj<3, 2, Number> IndexedMesh;

  //! A set of atom identifiers.
  typedef std::set<std::size_t> IdentifierSet;

  //
  // Data
  //

private:

  //! The geometry of the molecule.
  Molecule<Number> _molecule;
  //! The triangulation of the surface.
  TriangleArray _surface;
  //! The triangle indices for each atom.
  AtomTriangleMap _atomTriangleIndices;
  //! The atoms clipped by each atom.
  std::vector<IdentifierSet> _clippedAtoms;
  //! The atoms that have been affected by inserting or erasing atoms.
  IdentifierSet _affectedAtoms;
  //! The triangles that have been modified by inserting or erasing atoms.
  IndexSet _modifiedTriangles;
  //! The slope of the maximum edge length function.
  Number _edgeLengthSlope;
  //! The offset of the maximum edge length function.
  Number _edgeLengthOffset;
  //! The uniform refinement level.  (Non-negative if we are using it.)
  int _refinementLevel;
  //! The maximum stretch factor for rubber clipping.
  Number _maximumStretchFactor;
  //! The epsilon for cut clipping.  Used to identify if a point is on the clipping surface.
  Number _epsilonForCutClipping;
  //! Whether we interpret the triangle edges as circular for rubber clipping.
  bool _areUsingCircularEdges;
  //! The minimum allowed area for a triangle (as a fraction of the target area).
  /*!
    Triangles whose area is too small are discarded.
  */
  Number _minimumAllowedArea;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{

  //! Constructor.  Empty molecule.
  /*!
    \param edgeLengthSlope The slope of the maximum allowed edge length
    function.
    \param edgeLengthOffset The offset of the maximum allowed edge length
    function.
    \param refinementLevel The refinement level for tesselating atoms.  If
    you don't specify a refinement level (or give a negative value for this
    parameter) the maximum allowed edge length function will be used in
    tesselating atoms.
    \param nullTriangle The representation for a null triangle.  You will
    probably want to go with the default value here.

    The maximum allowed edge length function is
    edgeLengthSlope * r + edgeLengthOffset, where r is the radius of the
    atom being tesselated.

    If no arguments are specified, this is the default constructor.
  */
  MolecularSurface(const Number edgeLengthSlope = 0.0,
                   const Number edgeLengthOffset = 1.0,
                   const int refinementLevel = -1,
                   const Triangle nullTriangle =
                   {{{{0, 0, 0}},
                       {{0, 0, 0}},
                       {{0, 0, 0}}}}) :
    _molecule(),
    _surface(IdentifierAndTriangle(-1, nullTriangle)),
    _atomTriangleIndices(),
    _clippedAtoms(),
    _affectedAtoms(),
    _modifiedTriangles(),
    _edgeLengthSlope(edgeLengthSlope),
    _edgeLengthOffset(edgeLengthOffset),
    _refinementLevel(refinementLevel),
    _maximumStretchFactor(0), // By default allow infinite stretching.
    _epsilonForCutClipping(0),
    _areUsingCircularEdges(false),
    _minimumAllowedArea(0) {}

  //! Copy constructor.  Deep copy.
  MolecularSurface(const MolecularSurface& other) :
    _molecule(other._molecule),
    _surface(other._surface),
    _atomTriangleIndices(other._atomTriangleIndices),
    _clippedAtoms(other._clippedAtoms),
    _affectedAtoms(other._affectedAtoms),
    _modifiedTriangles(other._modifiedTriangles),
    _edgeLengthSlope(other._edgeLengthSlope),
    _edgeLengthOffset(other._edgeLengthOffset),
    _refinementLevel(other._refinementLevel),
    _maximumStretchFactor(other._maximumStretchFactor),
    _epsilonForCutClipping(other._epsilonForCutClipping),
    _areUsingCircularEdges(other._areUsingCircularEdges),
    _minimumAllowedArea(other._minimumAllowedArea) {}

  //! Assignment operator.  Deep copy.
  MolecularSurface&
  operator=(const MolecularSurface& other)
  {
    // Avoid assignment to self.
    if (&other != this) {
      _molecule = other._molecule;
      _surface = other._surface;
      _atomTriangleIndices = other._atomTriangleIndices;
      _clippedAtoms = other._clippedAtoms;
      _affectedAtoms = other._affectedAtoms;
      _modifiedTriangles = other._modifiedTriangles;
      _edgeLengthSlope = other._edgeLengthSlope;
      _edgeLengthOffset = other._edgeLengthOffset;
      _refinementLevel = other._refinementLevel;
      _maximumStretchFactor = other._maximumStretchFactor;
      _epsilonForCutClipping = other._epsilonForCutClipping;
      _areUsingCircularEdges = other._areUsingCircularEdges;
      _minimumAllowedArea = other._minimumAllowedArea;
    }
    // Return *this so assignments can chain.
    return *this;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return true if this molecule has the specified atom.
  bool
  hasAtom(const std::size_t identifier) const
  {
    return _molecule.hasAtom(identifier);
  }

  //! Return the specified atom.
  const AtomType&
  getAtom(std::size_t identifier) const
  {
    return _molecule.getAtom(identifier);
  }

  //! Get the beginning of the range of atom identifiers.
  IdentifierConstIterator
  getIdentifiersBeginning() const
  {
    return _molecule.getIdentifiersBeginning();
  }

  //! Get the end of the range of atom identifiers.
  IdentifierConstIterator
  getIdentifiersEnd() const
  {
    return _molecule.getIdentifiersEnd();
  }

  //! Get the modified triangle indices.
  template<typename IntOutputIterator>
  void
  getModifiedTriangleIndices(IntOutputIterator out) const;

  //! Get the slope of the maximum edge length function.
  Number
  getEdgeLengthSlope() const
  {
    return _edgeLengthSlope;
  }

  //! Get the offset of the maximum edge length function.
  Number
  getEdgeLengthOffset() const
  {
    return _edgeLengthOffset;
  }

  //! Get the refinement level.
  int
  getRefinementLevel() const
  {
    return _refinementLevel;
  }

  //! Get the maximum stretch factor for rubber clipping.
  Number
  getMaximumStretchFactor() const
  {
    return _maximumStretchFactor;
  }

  //! Get the epsilon for cut clipping.  This is used to determine if a vertex is on the clipping surface.
  Number
  getEpsilonForCutClipping() const
  {
    return _epsilonForCutClipping;
  }

  //! Get whether we interpret the triangle edges as circular for rubber clipping.
  bool
  getAreUsingCircularEdges() const
  {
    return _areUsingCircularEdges;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Triangle accessors.
  //@{

  //! Return the number of triangles (non-null and null triangles combined).
  std::size_t
  getTrianglesSize() const
  {
    return _surface.size();
  }

  //! Return the number of null triangles.
  std::size_t
  getNullTrianglesSize() const
  {
    return _surface.sizeNull();
  }

  //! Return the number of non-null triangles.
  std::size_t
  getNonNullTrianglesSize() const
  {
    return _surface.sizeNonNull();
  }

  //! Return true if the specified triangle is null.
  bool
  isTriangleNull(const std::size_t index) const
  {
    return _surface.isNull(index);
  }

  //! Return true if the specified triangle is non-null.
  bool
  isTriangleNonNull(const std::size_t index) const
  {
    return _surface.isNonNull(index);
  }

  //! Return the specified triangle.
  const Triangle&
  getTriangle(const std::size_t index) const
  {
    return _surface.get(index).second;
  }

  //! Return the atom identifier for the specified triangle.
  std::size_t
  getAtomIdentifierForTriangle(const std::size_t index) const
  {
    return _surface.get(index).first;
  }

  //! Return the atom for the specified triangle.
  /*!
    The computational complexity is \f$O(\log N)\f$ where \f$N\f$ is the
    number of atoms.  This is because it searches for the atom in the
    mst::Molecule data structure.
  */
  const AtomType&
  getAtomForTriangle(const std::size_t index) const
  {
    return getAtom(_surface.get(index).first);
  }

  //! Get each of the triangles in the surface.
  /*!
    The output is a range of vertices.  Triples of vertices define the
    triangles.
  */
  template<typename PointOutputIterator>
  void
  getTriangleVertices(PointOutputIterator out);

  //! Build the mesh for the specified triangle.  Use rubber clipping.
  /*!
    This does not affect the stored triangle surface.  This function is
    provided for debugging purposes.
  */
  void
  buildMeshUsingRubberClipping(std::size_t atomIdentifier,
                               geom::IndSimpSetIncAdj<3, 2, Number>* mesh)
  const;

  //! Return the minimum allowed area for a triangle (as a fraction of the target area).
  Number
  getMinimumAllowedArea() const
  {
    return _minimumAllowedArea;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Insert an atom.  Return the atom's identifier.
  std::size_t
  insert(const AtomType& atom)
  {
    const std::size_t identifier = _molecule.getUnusedIdentifier();
    insert(identifier, atom);
    return identifier;
  }

  //! Insert an atom with a specified identifier.
  void
  insert(const std::size_t identifier, const AtomType& atom)
  {
    // Insert into the molecule.
    _molecule.insert(identifier, atom);
    // Update the set of affected atoms.
    _affectedAtoms.insert(identifier);
    insertAtomsThatWillBeClipped(identifier);
    // If necessary, insert an empty container for the clipped atoms.
    while (identifier >= _clippedAtoms.size()) {
      _clippedAtoms.push_back(IdentifierSet());
    }
  }

  //! Erase an atom.
  /*!
    \note The atom must be present in the molecule.  This is checked with an
    \c assert().  You can use hasAtom() to determine if the atom is present.
  */
  void
  erase(const std::size_t identifier)
  {
    // Erase from the molecule.
    _molecule.erase(identifier);
    // Update the set of affected atoms.
    insertAtomsThatWereClipped(identifier);
    _affectedAtoms.erase(identifier);
    // Clear the clipping atoms.
    _clippedAtoms[identifier].clear();
  }

  //! Update the surface.  Update the set of modified triangles.
  void
  updateSurface();

  //! Update the surface using the bucket of triangles approach.  Update the set of modified triangles.
  void
  updateSurfaceUsingBot();

  //! Update the surface using cut clipping.  Update the set of modified triangles.
  void
  updateSurfaceWithCutClipping();

  //! Update the surface using rubber clipping.  Update the set of modified triangles.
  void
  updateSurfaceWithRubberClipping();

  //! Set the slope for the maximum edge length function.
  void
  setEdgeLengthSlope(const Number edgeLengthSlope)
  {
    _edgeLengthSlope = edgeLengthSlope;
    // Set the refinement level to an invalid value.
    _refinementLevel = -1;
  }

  //! Set the offset for the maximum edge length function.
  void
  setEdgeLengthOffset(const Number edgeLengthOffset)
  {
    _edgeLengthOffset = edgeLengthOffset;
    // Set the refinement level to an invalid value.
    _refinementLevel = -1;
  }

  //! Set the refinement level.
  void
  setRefinementLevel(const int refinementLevel)
  {
    _refinementLevel = refinementLevel;
  }

  //! Set the maximum stretch factor for rubber clipping.
  void
  setMaximumStretchFactor(const Number maximumStretchFactor)
  {
    _maximumStretchFactor = maximumStretchFactor;
  }


  //! Set the epsilon for cut clipping.  This is used to determine if a vertex is on the clipping surface.
  void
  setEpsilonForCutClipping(const Number epsilonForCutClipping)
  {
    _epsilonForCutClipping = epsilonForCutClipping;
  }

  //! Set whether we interpret the triangle edges as circular for rubber clipping.
  void
  setAreUsingCircularEdges(const bool areUsingCircularEdges)
  {
    _areUsingCircularEdges = areUsingCircularEdges;
  }

  //! Set the minimum allowed area for a triangle (as a fraction of the target area).
  void
  setMinimumAllowedArea(const Number minimumAllowedArea)
  {
    _minimumAllowedArea = minimumAllowedArea;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Information.
  //@{

  //! Print information about the triangulation.
  void
  printInformation(std::ostream& out) const;

  //! Compute triangulation error statistics.
  void
  computeErrorStatistics(Number* minimumRadiusDeviation,
                         Number* maximumRadiusDeviation,
                         Number* meanRadiusDeviation,
                         Number* minimumPenetration,
                         Number* maximumPenetration,
                         Number* meanPenetration) const;

  //@}

  //--------------------------------------------------------------------------
  // Private member functions.
  //--------------------------------------------------------------------------

private:

  //! Clear the triangulation for the specified atom.
  /*!
    Update the set of modified triangles.
  */
  void
  clearAtom(std::size_t identifier);

  //! Triangulate the visible surface for the specified atom.
  /*!
    Update the set of modified triangles.
  */
  void
  triangulateAtom(std::size_t identifier);

  //! Triangulate the visible surface for the specified atom with a bucket of triangles.
  /*!
    Update the set of modified triangles.
  */
  void
  triangulateAtomWithBot(std::size_t identifier);

  //! Triangulate the visible surface for the specified atom with cut clipping.
  /*!
    Update the set of modified triangles.
  */
  void
  triangulateAtomWithCutClipping(std::size_t identifier);

  //! Triangulate the visible surface for the specified atom with rubber clipping.
  /*!
    Update the set of modified triangles.
  */
  void
  triangulateAtomWithRubberClipping(std::size_t identifier);

  //! Insert triangles for the specified atom.
  /*!
    Update the set of modified triangles.
  */
  template<typename TriangleInputIterator>
  void
  insertTriangles(std::size_t identifier, TriangleInputIterator beginning,
                  TriangleInputIterator end, Number targetEdgeLength);

  //! Add the atom identifiers that the specified atom will clip to the affected set.
  /*!
    This function is called when inserting the specified atom.
  */
  void
  insertAtomsThatWillBeClipped(std::size_t identifier);

  //! Add the atom identifiers that clipped the specified atom to the affected set.
  /*!
    This function is called when erasing the specified atom.
  */
  void
  insertAtomsThatWereClipped(std::size_t identifier);

  //! Return true if the atom clips the mesh of the specified atom.
  bool
  doesClipMesh(std::size_t identifier, const AtomType& atom) const;

};


} // namespace mst
}

#define __mst_MolecularSurface_ipp__
#include "stlib/mst/MolecularSurface.ipp"
#undef __mst_MolecularSurface_ipp__

#endif
