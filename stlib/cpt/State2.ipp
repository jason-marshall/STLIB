// -*- C++ -*-

#if !defined(__cpt_State2_ipp__)
#error This file is an implementation detail of the class State.
#endif

namespace stlib
{
namespace cpt
{

//! Hold the state for a 2-D closest point transform.
/*!
  Most of the functionality is dimension independent and is thus implemented
  in the base class cpt::StateBase.

  To perform the closest point transform, you must call:
  -# setParameters() to specify the Cartesian domain
  of interest and how far to compute the distance.
  Note that the computational complexity of the alorithm is proportional to
  this maximum distance.  (The number of grid points for which
  the distance is computed is proportional to the maximum distance.)
  If performance is an issue, make sure that the \c maximumDistance
  parameter is no larger than needed.
  -# cpt::StateBase<N,T>::setLattice() to specify the lattice on which the
  grids lie.
  -# cpt::StateBase<N,T>::insertGrid() for each grid on the lattice.
  Here you specify arrays that hold the
  distance, and optionally the gradient of distance, closest point
  and closest face fields.  cpt::StateBase<N,T>::setLattice() must
  be called before any calls to cpt::StateBase<N,T>::insertGrid().
  -# cpt::StateBase<N,T>::setBRep() or
  cpt::StateBase<N,T>::setBRepWithNoClipping()
  to specify the surface.
  The triangle surface is a boundary representation, b-rep, of the
  volume that is inside the surface.  setParameters() must be called
  before calling cpt::StateBase<N,T>::setBRep().
  -# computeClosestPointTransform() to compute the the CPT on the specified grids.

  To compute the CPT for a new b-rep, call:
  -# cpt::StateBase<N,T>::setBRep().
  -# computeClosestPointTransform().

  To compute the CPT for a different set of grids, call the following:
  -# cpt::StateBase<N,T>::clearGrids() to clear the old grids.
  -# cpt::StateBase<N,T>::setLattice() if the lattice has changed.
  -# cpt::StateBase<N,T>::insertGrid() for each new grid on the lattice.
  -# computeClosestPointTransform().
  .
  For AMR grids, one must follow these steps for each each level of the
  AMR grid.  (Each level of the AMR grid is a separate lattice.)
*/
template<typename T>
class State<2, T> :
  public StateBase<2, T>
{
private:

  //
  // Private types.
  //

  typedef StateBase<2, T> Base;

  //
  // Public types.
  //

public:

  //! The number type.
  typedef typename Base::Number Number;
  //! A point in 2-D.
  typedef typename Base::Point Point;
  //! An indexed face in 2-D.
  typedef typename Base::IndexedFace IndexedFace;
  //! An extent in 2-D.
  typedef typename Base::SizeList SizeList;
  //! An index in 2-D.
  typedef typename Base::IndexList IndexList;
  //! An index range in 2-D.
  typedef typename Base::Range Range;
  //! A bounding box.
  typedef typename Base::BBox BBox;
  //! The grid geometry.
  typedef typename Base::Lattice Lattice;
  //! The grid.
  typedef typename Base::Grid Grid;
  //! The b-rep.
  typedef typename Base::BRep BRep;

  //
  // Member data.
  //

private:

  bool _areUsingLocalClipping;
  std::size_t _globalClippingMethod;
  std::size_t _decimationFactor;

  //
  // Not implemented.
  //

private:

  //! The copy constructor is not implemented.
  State(const State&);

  //! The assignment operator is not implemented.
  State&
  operator=(const State&);

  //
  // Private using.
  //

  // Member data.
  using Base::_brep;
  using Base::_hasCptBeenComputed;
  using Base::_lattice;
  using Base::_grids;
  using Base::_maximumDistance;

  // Accessors.
  using Base::hasBRepBeenSet;
  using Base::initializeGrids;

public:

  //
  // Using.
  //

  // Accessors.

  //! Return the number of grids.
  using Base::getNumberOfGrids;
  //! Get the specified grid.
  using Base::getGrid;
  //! Return the domain that contains all grids.
  using Base::getDomain;
  //! Return how far the distance is being computed.
  using Base::getMaximumDistance;

  // Manipulators.

  // Don't use the base class function.  We will override it.
  //using Base::setParameters;
  //! Set the lattice geometry.
  using Base::setLattice;
  //! Add a grid for the closest point transform.
  using Base::insertGrid;
  //! Clear the grids.
  using Base::clearGrids;
  //! Compute the closest point transform for signed distance.
  using Base::computeClosestPointTransformUsingBBox;
  //! Compute the closest point transform for signed distance.
  using Base::computeClosestPointTransformUsingBruteForce;
  //! Compute the closest point transform for unsigned distance.
  using Base::computeClosestPointTransformUnsignedUsingBBox;
  //! Compute the closest point transform for unsigned distance.
  using Base::computeClosestPointTransformUnsignedUsingBruteForce;
  //! Set the b-rep.
  using Base::setBRepWithNoClipping;
  //! Set the b-rep.  Clip the mesh.
  using Base::setBRep;
  //! Flood fill the signed distance.
  using Base::floodFillAtBoundary;
  //! Flood fill the signed distance.  Determine the sign of the distance on all grids.
  using Base::floodFillDetermineSign;
  //! Flood fill the unsigned distance.
  using Base::floodFillUnsigned;
  //! Check the grids.
  using Base::areGridsValid;
  //! Check the grids.
  using Base::areGridsValidUnsigned;

  //-------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  State() :
    Base(),
    _areUsingLocalClipping(false),
    _globalClippingMethod(0),
    _decimationFactor(0) {}

  //! Destructor.
  ~State() {}

  //@}
  //-------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  //! Set parameters for the closest point transform.
  /*!
    This function must be called at least once before calls to
    setBRep() or computeClosestPointTransform().

    \param domain is the Cartesian domain that contains all grids.
    (All grids on all lattices.)  This domain (expanded by the max distance)
    will be used to clip the b-rep in any subsequent calls to setBRep().

    \param maximumDistance
    The distance will be computed up to maximumDistance away from the curve.

    \param areUsingLocalClipping indicates if adjacent faces should be used to
    clip the characteristic face polygons.

    \param globalClippingMethod indicates if/how a decimated mesh should be used
    to clip the characteristic polyhedra.  0 corresponds to no global clipping;
    1 corresponds to limited global clipping; 2 corresponds to full global
    clipping.

    \param decimationFactor is the factor used to decimate the mesh for
    use in global clipping.

    The distance for grid points whose distance is larger than
    maximumDistance will be set to \c std::numeric_limits<Number>::max().
    Each component of the closest point of these far away
    points will be set to \c std::numeric_limits<Number>::max().
    The closest face of far away points will be set to
    std::numeric_limits<std::size_t>::max().
  */
  void
  setParameters(const BBox& domain, const Number maximumDistance,
                const bool areUsingLocalClipping = false,
                const std::size_t globalClippingMethod = 0,
                const std::size_t decimationFactor = 1)
  {
    Base::setParameters(domain, maximumDistance);
    _areUsingLocalClipping = areUsingLocalClipping;
    _globalClippingMethod = globalClippingMethod;
    _decimationFactor = decimationFactor;
  }

  //! Set parameters for the closest point transform.
  /*!
    This function must be called at least once before calls to
    setBRep() or computeClosestPointTransform().

    \param domainLowerData The lower corner of the Cartesian domain that
    contains all grids.
    (All grids on all lattices.)  This domain (expanded by the max distance)
    will be used to clip the b-rep in any subsequent calls to setBRep().

    \param domainUpperData The upper corner of the Cartesian domain that
    contains all grids.

    \param maximumDistance
    The distance will be computed up to maximumDistance away from the curve.

    \param areUsingLocalClipping indicates if adjacent faces should be used to
    clip the characteristic face polygons.

    \param globalClippingMethod indicates if/how a decimated mesh should be used
    to clip the characteristic polyhedra.  0 corresponds to no global clipping;
    1 corresponds to limited global clipping; 2 corresponds to full global
    clipping.

    \param decimationFactor is the factor used to decimate the mesh for
    use in global clipping.

    The distance for grid points whose distance is larger than
    maximumDistance will be set to \c std::numeric_limits<Number>::max().
    Each component of the closest point of these far away
    points will be set to \c std::numeric_limits<Number>::max().
    The closest face of far away points will be set to
    std::numeric_limits<std::size_t>::max().
  */
  void
  setParameters(const Number* domainLowerData, const Number* domainUpperData,
                const Number maximumDistance,
                const bool areUsingLocalClipping = false,
                const std::size_t globalClippingMethod = 0,
                const std::size_t decimationFactor = 1)
  {
    const Point domainLower = ext::copy_array<Point>(domainLowerData);
    const Point domainUpper = ext::copy_array<Point>(domainUpperData);
    setParameters(BBox(domainLower, domainUpper), maximumDistance,
                  areUsingLocalClipping,
                  globalClippingMethod, decimationFactor);
  }

  //! Compute the closest point transform.
  /*!
    Compute the signed distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    The algorithm uses polygon scan conversion to determine which grid
    points are close to the shapes comprising the surface.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransform();

  //! Compute the closest point transform.
  /*!
    Compute the unsigned distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    The algorithm uses polygon scan conversion to determine which grid
    points are close to the shapes comprising the surface.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUnsigned();

  // CONTINUE
#if 0
  //! Flood fill the distance.
  /*!
    The distance is flood filled.  If any of the distances are known,
    set the unknown distances to +- farAway.  If no distances are
    known, calculate the sign of the distance from the supplied mesh and
    set all distances to + farAway.
  */
  template<typename PointConstIterator, typename IndexConstIterator>
  void
  floodFill(const Number farAway,
            PointConstIterator verticesBeginning,
            PointConstIterator verticesEnd,
            IndexConstIterator facesBeginning,
            IndexConstIterator facesEnd);
#endif

  // CONTINUE
#if 0
  //! Flood fill the distance.
  /*!
    The distance is flood filled.  If any of the distances are known,
    set the unknown distances to +- farAway.  If no distances are
    known, calculate the sign of the distance from the supplied mesh and
    set all distances to + farAway.

    \param verticesSize is the number of vertices.
    \param vertices is a const pointer to the beginning of the vertices.
    \param facesSize is the number of faces.
    \param faces is a const pointer to the beginning of the faces.
  */
  void
  flood_fill(const Number farAway,
             const std::size_t verticesSize,
             const void* vertices,
             const std::size_t facesSize,
             const void* faces);
#endif

  //-------------------------------------------------------------------------
  //! \name Inside/outside.
  //@{

  //! Determine which grid points are inside the solid (non-positive distance).
  /*!
    Use this interface if you repeatedly determine which points are inside.
    The distance array is needed for determining the sign of the distance.
    With this interface, the distance array is not allocated anew for each
    function call.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsInside(const BBox& domain,
                        container::MultiArrayRef<bool, 2>* areInside,
                        container::MultiArrayRef<Number, 2>* distance);

  //! Wrapper for the above function.
  /*!
    \param lower The lower corner of the Cartesian domain.
    \param upper The upper corner of the Cartesian domain.
    \param extents The array extents.
    \param areInside Boolean array is set to whether each grid point is
    inside the solid.
    \param distance Number array that is used in the computation.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsInside(const Point& lower,
                        const Point& upper,
                        const SizeList& extents,
                        bool* areInside,
                        Number* distance);

  //! Determine which grid points are inside the solid (non-positive distance).
  /*!
    Use this interface if you only determine which points are inside one time.
    This function allocates a distance array and calls the function above.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsInside(const BBox& domain,
                        container::MultiArrayRef<bool, 2>* areInside)
  {
    container::MultiArray<Number, 2> distance(areInside->extents(),
        areInside->bases());
    return determinePointsInside(domain, areInside, &distance);
  }

  //! Wrapper for the above function.
  /*!
    \param lower The lower corner of the Cartesian domain.
    \param upper The upper corner of the Cartesian domain.
    \param extents The array extents.
    \param areInside Boolean array is set to whether each grid point is
    inside the solid.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsInside(const Point& lower,
                        const Point& upper,
                        const SizeList& extents,
                        bool* areInside);


  //! Determine which grid points are outside the solid (positive distance).
  /*!
    Use this interface if you repeatedly determine which points are outside.
    The distance array is needed for determining the sign of the distance.
    With this interface, the distance array is not allocated anew for each
    function call.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsOutside(const BBox& domain,
                         container::MultiArrayRef<bool, 2>* areOutside,
                         container::MultiArrayRef<Number, 2>* distance)
  {
    // Determine the points inside.
    const std::pair<std::size_t, std::size_t> counts =
      determinePointsInside(domain, areOutside, distance);
    // Switch the sign of the distance.
    for (typename container::MultiArrayRef<bool, 2>::iterator
         i = areOutside->begin(); i != areOutside->end(); ++i) {
      *i = ! *i;
    }
    for (typename container::MultiArrayRef<Number, 2>::iterator
         i = distance->begin(); i != distance->end(); ++i) {
      *i = -*i;
    }
    return counts;
  }

  //! Wrapper for the above function.
  /*!
    \param lower The lower corner of the Cartesian domain.
    \param upper The upper corner of the Cartesian domain.
    \param extents The array extents.
    \param areOutside Boolean array is set to whether each grid point is
    outside the solid.
    \param distance Number array that is used in the computation.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsOutside(const Point& lower,
                         const Point& upper,
                         const SizeList& extents,
                         bool* areOutside,
                         Number* distance);

  //! Determine which grid points are outside the solid (positive distance).
  /*!
    Use this interface if you only determine which points are outside one time.
    This function allocates a distance array and calls the function above.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsOutside(const BBox& domain,
                         container::MultiArrayRef<bool, 2>* areOutside)
  {
    const std::pair<std::size_t, std::size_t> counts =
      determinePointsInside(domain, areOutside);
    // Switch the inside and outside.
    for (typename container::MultiArrayRef<bool, 2>::iterator
         i = areOutside->begin(); i != areOutside->end(); ++i) {
      *i = ! *i;
    }
    return counts;
  }

  //! Wrapper for the above function.
  /*!
    \param lower The lower corner of the Cartesian domain.
    \param upper The upper corner of the Cartesian domain.
    \param extents The array extents.
    \param areOutside Boolean array is set to whether each grid point is
    outside the solid.
  */
  std::pair<std::size_t, std::size_t>
  determinePointsOutside(const Point& lower,
                         const Point& upper,
                         const SizeList& extents,
                         bool* areOutside);

  //@}
  //-------------------------------------------------------------------------
  //! \name I/O.
  //@{

  //! Display information about the state of the closest point transform.
  void
  displayInformation(std::ostream& out);

  //@}

private:

  //
  // Private member functions.
  //

  void
  decimateVertices(std::vector<Point>* decimatedVertices);
};




template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
computeClosestPointTransform()
{
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the grids.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Choose the vertices used in global clipping.
  std::vector<Point> globalPoints;
  if (_globalClippingMethod) {
    decimateVertices(&globalPoints);
  }

  // Compute the closest point transform.
  return _brep.computeClosestPoint(_lattice, &_grids, _maximumDistance,
                                   _areUsingLocalClipping,
                                   _globalClippingMethod, globalPoints);
}


template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
computeClosestPointTransformUnsigned()
{
  // Make sure everything is set.
  assert(getNumberOfGrids() > 0 && hasBRepBeenSet());

  // Initialize the distance array.
  initializeGrids();

  // Signify that the cpt has been computed.
  _hasCptBeenComputed = true;

  // Choose the vertices used in global clipping.
  std::vector<Point> globalPoints;
  if (_globalClippingMethod) {
    decimateVertices(&globalPoints);
  }

  // compute the closest point transform.
  return _brep.computeClosestPointUnsigned(_lattice, &_grids,
         _maximumDistance,
         _areUsingLocalClipping,
         _globalClippingMethod,
         globalPoints);
}


// CONTINUE
#if 0
template<typename T>
template<typename PointConstIterator, typename IndexConstIterator>
inline
void
State<2, T>::
floodFill(const Number farAway,
          PointConstIterator verticesBeginning,
          PointConstIterator verticesEnd,
          IndexConstIterator facesBeginning,
          IndexConstIterator facesEnd)
{
  // Make sure the cpt has been computed first.
  assert(_hasCptBeenComputed);

  // For each grid.
  for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
    // Flood fill the distance grid.
    _grids[n].floodFill(farAway, verticesBeginning, verticesEnd,
                        facesBeginning, facesEnd);
  }
}
#endif


// CONTINUE
#if 0
template<typename T>
inline
void
State<2, T>::
floodFill(const Number farAway,
          const std::size_t verticesSize,
          const void* vertices,
          const std::size_t facesSize,
          const void* faces)
{
  const Point* v = reinterpret_cast<const Point*>(vertices);
  const IndexedFace* f =
    reinterpret_cast<const IndexedFace*>(faces);
  floodFill(farAway, v, v + 2 * verticesSize, f, f + 2 * facesSize);
}
#endif


template<typename T>
inline
void
State<2, T>::
displayInformation(std::ostream& out)
{
  if (_areUsingLocalClipping) {
    out << "Local clipping performed on edge polygons.\n";
  }
  else {
    out << "No local clipping performed on edge polygons.\n";
  }

  if (_globalClippingMethod == 0) {
    out << "No global clipping performed on characteristic polygons.\n";
  }
  else if (_globalClippingMethod == 1) {
    out << "Limited global clipping performed on characteristic polygons.\n"
        << "The decimation factor is " << _decimationFactor << '\n';
  }
  else if (_globalClippingMethod == 2) {
    out << "Full global clipping performed on characteristic polygons.\n"
        << "The decimation factor is " << _decimationFactor << '\n';
  }
  else {
    assert(false);
  }

  Base::displayInformation(out);
}


// Decimate the vertices by the given factor.
// If factor is zero, the output vector is empty.
template<typename T>
inline
void
State<2, T>::
decimateVertices(std::vector<Point>* decimatedVertices)
{
  decimatedVertices->clear();
  const std::size_t verticesSize = _brep.vertices.size();
  if (_decimationFactor >= 1) {
    for (std::size_t i = 0; i < verticesSize; i += _decimationFactor) {
      decimatedVertices->push_back(_brep.vertices[i]);
    }
  }
}




// Determine which grid points are inside the solid (non-positive distance).
template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
determinePointsInside(const BBox& domain,
                      container::MultiArrayRef<bool, 2>* areInside,
                      container::MultiArrayRef<Number, 2>* distance)
{
  // The two arrays must be the same size.
  assert(areInside->extents() == distance->extents() &&
         areInside->bases() == distance->bases());

  // Set up the lattice.
  setLattice(areInside->extents(), domain);
  // Set the maximum distance to slightly more than the maximum grid
  // spacing in any direction.
  setParameters(domain, 1.01 * ext::max(_lattice.getDelta()));
  // Clear any old grids.
  clearGrids();
  // Add the distance grid. Use empty grids for the rest.
  insertGrid(distance->extents(), distance->bases(), distance->data(), 0, 0,
             0);

  // Compute the distance.
  const std::pair<std::size_t, std::size_t> counts =
    computeClosestPointTransform();

  // If no distances were set, we need to determine the sign of the distance.
  if (counts.second == 0) {
    // Make a data structure that supports distance queries.
    geom::ISS_SignedDistance<geom::IndSimpSetIncAdj<2, 1, T> >
    signedDistance(_brep);
    // We can compute the signed distance to any point in the grid; we choose
    // the min corner.
    // If the distance is negative.
    if (signedDistance(domain.lower) <= 0) {
      // Set all of the distances to a negative value.
      std::fill(_grids[0].getDistance().begin(),
                _grids[0].getDistance().end(), -1.0);
    }
    // Else the distance is positive.
    else {
      // Set all of the distances to a positive value.
      std::fill(_grids[0].getDistance().begin(),
                _grids[0].getDistance().end(), 1.0);
    }
  }
  // Else, some of the distances were set.  We can flood fill the unknown
  // distances.
  else {
    // Flood fill the distance.  The far away value can be any positive number.
    floodFillAtBoundary(1.0);
  }

  // Clear the grids so they are not inadvertently used.
  clearGrids();
  // Use the sign of the distance to determine which points are inside.
  typename container::MultiArray<bool, 2>::iterator i = areInside->begin();
  const typename container::MultiArray<bool, 2>::iterator iEnd = areInside->end();
  typename container::MultiArray<Number, 2>::const_iterator j = distance->begin();
  for (; i != iEnd; ++i, ++j) {
    // True iff the distance is non-positive.
    *i = *j <= 0.0;
  }

  return counts;
}




template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
determinePointsInside(const Point& lower,
                      const Point& upper,
                      const SizeList& extents,
                      bool* areInside,
                      Number* distance)
{
  container::MultiArrayRef<bool, 2> areInsideArray(areInside, extents);
  container::MultiArrayRef<Number, 2> distanceArray(distance, extents);
  return determinePointsInside(BBox(lower, upper), &areInsideArray,
                               &distanceArray);
}




template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
determinePointsInside(const Point& lower,
                      const Point& upper,
                      const SizeList& extents,
                      bool* areInside)
{
  container::MultiArrayRef<bool, 2> areInsideArray(areInside, extents);
  return determinePointsInside(BBox(lower, upper), &areInsideArray);
}




template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
determinePointsOutside(const Point& lower,
                       const Point& upper,
                       const SizeList& extents,
                       bool* areOutside,
                       Number* distance)
{
  container::MultiArrayRef<bool, 2> areOutsideArray(areOutside, extents);
  container::MultiArrayRef<Number, 2> distanceArray(distance, extents);
  return determinePointsOutside(BBox(lower, upper), &areOutsideArray,
                                &distanceArray);
}




template<typename T>
inline
std::pair<std::size_t, std::size_t>
State<2, T>::
determinePointsOutside(const Point& lower,
                       const Point& upper,
                       const SizeList& extents,
                       bool* areOutside)
{
  container::MultiArrayRef<bool, 2> areOutsideArray(areOutside, extents);
  return determinePointsOutside(BBox(lower, upper), &areOutsideArray);
}

} // namespace cpt
}
