// -*- C++ -*-

#if !defined(__cpt_StateBase_h__)
#define __cpt_StateBase_h__

// Local
#include "stlib/cpt/Grid.h"
#include "stlib/cpt/BRep.h"

#include "stlib/geom/mesh/iss/distance.h"

#include <iostream>

#include <cmath>

namespace stlib
{
namespace cpt
{

//! Hold the state for a closest point transform.
/*!
  Implements the dimension-independent functionality.
*/
template < std::size_t N, typename T = double >
class StateBase
{
  //
  // Protected types.
  //

protected:

  //! The number type.
  typedef T Number;
  //! A bounding box.
  typedef geom::BBox<Number, N> BBox;
  //! The lattice defines a domain and index extents.
  typedef geom::RegularGrid<N, T> Lattice;
  //! The grid.
  typedef cpt::Grid<N, T> Grid;
  //! The b-rep.
  typedef cpt::BRep<N, T> BRep;

  //! A point in N-D.
  typedef typename Grid::Point Point;
  //! An extent in N-D.
  typedef typename Grid::SizeList SizeList;
  //! A multi-index in N-D.
  typedef typename Grid::IndexList IndexList;
  //! The indices of a face.
  typedef typename Grid::IndexList IndexedFace;
  //! A single index.
  typedef typename Grid::Index Index;
  //! A multi-index range in N-D.
  typedef typename Grid::Range Range;

  //
  // Member data.
  //

protected:

  //! Has the b-rep been set.
  bool _hasBRepBeenSet;
  //! Has the CPT been computed.
  bool _hasCptBeenComputed;
  //! The domain containing all grids for which the CPT will be computed.
  /*! This may be used in clipping the mesh. */
  BBox _domain;
  //! How far (in Cartesian space) to compute the distance.
  Number _maximumDistance;
  //! The lattice.
  Lattice _lattice;
  //! The grids.
  std::vector<Grid> _grids;
  //! The b-rep.
  BRep _brep;

  //
  // Not implemented.
  //

private:

  //! The copy constructor is not implemented.
  StateBase(const StateBase&);

  //! The assignment operator is not implemented.
  StateBase&
  operator=(const StateBase&);

public:

  //-------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{

  //! Default constructor.
  StateBase() :
    _hasBRepBeenSet(false),
    _hasCptBeenComputed(false),
    _domain(),
    _maximumDistance(-1),
    _lattice(),
    _grids(),
    _brep() {}

  //@}
  //-------------------------------------------------------------------------
  //! \name Accessors.
  //@{

  //! Return the number of grids.
  std::size_t
  getNumberOfGrids() const
  {
    return _grids.size();
  }

  //! Get the specified grid.
  const Grid&
  getGrid(const std::size_t n) const
  {
    return _grids[n];
  }

  //! Return true if the b-rep has been set.
  bool
  hasBRepBeenSet() const
  {
    return _hasBRepBeenSet;
  }

  //! Return the domain that contains all grids.
  const BBox&
  getDomain() const
  {
    return _domain;
  }

  //! Return how far the distance is being computed.
  Number
  getMaximumDistance() const
  {
    return _maximumDistance;
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

  // CONTINUE: The max distance should be specified on a per level basis.
  //! Set the parameters for the Closest Point Transform.
  /*!
    This function must be called at least once before calls to
    computeClosestPointTransform().

    \param domain is the Cartesian domain that contains all grids.
    \param maximumDistance
    The distance will be computed up to maximumDistance away from the surface.

    The distance for grid points whose distance is larger than
    maximumDistance will be set to \c std::numeric_limits<Number>::max().
    Each component of the closest point of these far away
    points will be set to \c std::numeric_limits<Number>::max().
    The closest face of far away points will be set to
    std::numeric_limits<std::size_t>::max().
  */
  void
  setParameters(const BBox& domain, Number maximumDistance);

  //! Set parameters for the closest point transform.
  /*!
    This is a wrapper for the above setParameters function.
  */
  void
  setParameters(const Number* domainLowerData, const Number* domainUpperData,
                const Number maximumDistance)
  {
    const Point domainLower = ext::copy_array<Point>(domainLowerData);
    const Point domainUpper = ext::copy_array<Point>(domainUpperData);
    setParameters(BBox(domainLower, domainUpper), maximumDistance);
  }

  //! Set the parameters for the Closest Point Transform.
  /*!
    This calls the first setParameters() function.
    The domain containing all grids is set to all space.  This means that
    clipping the mesh will have no effect.
  */
  void
  setParameters(const Number maximumDistance)
  {
    // The point at infinity.
    const Point Infinity =
      ext::filled_array<Point>(std::numeric_limits<Number>::max());
    // Call setParameters() with all of space as the domain.
    setParameters(BBox(-Infinity, Infinity), maximumDistance);
  }

  // CONTINUE Make sure setLattice is called before adding grids.
  // CONTINUE enable multiple levels.  Add documentation.
  //! Set the grid geometry.
  /*!
    \param extents are the grid geometry extents.
    \param domain is the grid geometry domain.
  */
  void
  setLattice(const SizeList& extents, const BBox& domain);

  //! Set the lattice geometry.
  /*!
    This is a wrapper for the above setLattice function().
  */
  void
  setLattice(const int* extentsData, const Number* lowerData,
             const Number* upperData)
  {
    const SizeList extents = ext::copy_array<SizeList>(extentsData);
    const Point lower = ext::copy_array<Point>(lowerData);
    const Point upper = ext::copy_array<Point>(upperData);
    setLattice(extents, BBox(lower, upper));
  }

  //! Add a grid for the Closest Point Transform.
  /*!
    This function must be called at least once before calls to
    computeClosestPointTransform().

    \param extents The array extents.
    \param bases The index bases for the arrays.
    \param useGradientOfDistance Whether to compute the gradient of the
    distance. The gradient of the distance is
    computed from the geometric primitives and not by differencing the
    distance array. Thus it is accurate to within machine precision.
    \param useClosestPoint Whether to compute the closest points.
    \param useClosestFace Whether to compute the closet face on the mesh.
  */
  void
  insertGrid(const SizeList& extents,
             const IndexList& bases,
             bool useGradientOfDistance,
             bool useClosestPoint,
             bool useClosestFace);

  //! Add a grid for the Closest Point Transform.
  /*!
    \param extents The array extents.
    \param bases The index bases for the arrays.
    For each of the \c gradientOfDistance, \c closestPoint and \c closestFace
    arrays: if the pointer is non-zero, that quantity will be computed.
    \param distance Pointer to the distance array.
    \param gradientOfDistance Pointer to the gradient of distance array.
    \param closestPoint Pointer to the closest point array.
    \param closestFace Pointer to the closest face array.
  */
  void
  insertGrid(const SizeList& extents,
             const IndexList& bases,
             Number* distance,
             Number* gradientOfDistance,
             Number* closestPoint,
             int* closestFace);

  //! Add a grid for the Closest Point Transform.
  /*!
    This is a wrapper for the above insertGrid function().
  */
  void
  insertGrid(const int* lowerBounds, const int* upperBounds, Number* distance,
             Number* gradientOfDistance, Number* closestPoint,
             int* closestFace)
  {
    const IndexList bases = ext::copy_array<IndexList>(lowerBounds);
    std::size_t extentsData[N];
    for (std::size_t i = 0; i < N; ++i) {
      extentsData[i] = upperBounds[i] - lowerBounds[i] + 1;
    }
    const SizeList extents = ext::copy_array<SizeList>(extentsData);
    insertGrid(extents, bases, distance, gradientOfDistance, closestPoint,
               closestFace);
  }

  //! Clear the grids.
  /*!
    If the grids change, call this function and then add all the new grids with
    insertGrid().
  */
  void
  clearGrids()
  {
    _grids.clear();
  }

  //! Compute the closest point transform for signed distance.
  /*!
    Compute the signed distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    This algorithm does not use polyhedron scan conversion.  Instead, it builds
    bounding boxes around the characteristic polyhedra.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().

    \return
    Return the number of points for which the distance was computed and the
    number of points for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUsingBBox();

  //! Compute the closest point transform for signed distance.
  /*!
    Compute the signed distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    This algorithm does not use polyhedron scan conversion or the
    characteristic polyhedra.  Instead, it builds
    bounding boxes around the faces, edges and vertices.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().

    \return
    Return the number of points for which the distance was computed and the
    number of points for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUsingBruteForce();

  //! Compute the closest point transform for signed distance.
  /*!
    Compute the signed distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    This algorithm uses a bounding box tree to store the mesh and
    a lower-upper-bound queries to determine the distance.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().

    \return
    Return the number of points for which the distance was computed and the
    number of points for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUsingTree();

  //! Compute the closest point transform for unsigned distance.
  /*!
    Compute the unsigned distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    This algorithm does not use polyhedron scan conversion.  Instead, it builds
    bounding boxes around the characteristic polyhedra.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().

    \return
    Return the number of points for which the distance was computed and the
    number of points for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUnsignedUsingBBox();

  //! Compute the closest point transform for unsigned distance.
  /*!
    Compute the unsigned distance.  Compute the gradient of the distance, the
    closest face and closest point if their arrays specified in
    insertGrid() are nonzero.

    This algorithm does not use polyhedron scan conversion or the
    characteristic polyhedra.  Instead, it builds
    bounding boxes around the faces, edges and vertices.

    The unknown distances, gradients, and closest points are set to
    \c std::numeric_limits<Number>::max().
    The unknown closest faces are set to
    std::numeric_limits<std::size_t>::max().

    \return
    Return the number of points for which the distance was computed and the
    number of points for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUnsignedUsingBruteForce();

  //! Flood fill the distance.
  /*!
    This function is used to prepare the distance for visualization.
    The signed distance is flood filled.
    If any of the distances are known in a particular grid,
    set the unknown distances to +-farAway.  If no
    distances are known, set all distances to +farAway.
    Thus note that if no points in a particular grid have known distance,
    then the sign of the distance is not determined.
  */
  void
  floodFillAtBoundary(Number farAway);

  //! Flood fill the distance.
  /*!
    This function is used to prepare the distance for visualization.
    The signed distance is flood filled.
    If any of the distances are known in a particular grid,
    set the unknown distances to +-farAway.  If no
    distances are known, determine the correct sign by computing the signed
    distance to the boundary for a single point in the grid.

    \note In order to determine the sign of the distance for far away grids,
    this algorithm needs that portion of the boundary that is closest to
    those grids.  Using setBRep() may clip away the needed portion (or for
    that matter all) of the boundary.  You should use
    setBRepWithNoClipping() before calling this function.  To be on the
    safe side, you should give the entire boundary to
    setBRepWithNoClipping() .
  */
  void
  floodFillDetermineSign(Number farAway);

  //! Flood fill the unsigned distance.
  /*!
    The unsigned distance is flood filled.  Unknown distances are set to
    \c farAway.
  */
  void
  floodFillUnsigned(Number farAway);

  //! Check the grids.
  /*!
    Verify that the distance grids are valid.  The known distances should
    be between +-maximumDistance.  The difference between adjacent grid
    points should not be more than the grid spacing.  Verify that the
    closest point and closest face grids are valid.  Return true if the grids
    are valid.  Return false and print a message to stderr otherwise.
  */
  bool
  areGridsValid();

  //! Check the grids.
  /*!
    Verify that the distance grids are valid.  The known distances should
    be between 0 and maximumDistance.  The difference between adjacent grid
    points should not be more than the grid spacing.  Verify that the
    closest point and closest face grids are valid.  Return true if the grids
    are valid.  Return false and print a message to stderr otherwise.
  */
  bool
  areGridsValidUnsigned();

  //! Set the b-rep.
  /*!
    Do not use the Cartesian domain to clip the mesh.

    Either this function or setBRep() must be called at least
    once before calls to computeClosestPointTransform().

    \param vertices are the locations of the vertices.
    \param faces is a vector of tuples of vertex indices that describe the
    faces.
  */
  void
  setBRepWithNoClipping
  (const std::vector<std::array<Number, N> >& vertices,
   const std::vector<std::array<std::size_t, N> >& faces);

  //! Wrapper for the above function.
  void
  setBRepWithNoClipping(std::size_t numVertices, const Number* vertices,
                        std::size_t numFaces, const int* faces);

  //! Set the b-rep.
  /*!
    Clip the mesh to use only points that affect the cartesian domain.

    Either this function or setBRepWithNoClipping() must be called at least
    once before calls to computeClosestPointTransform().  This version is more
    efficient if the b-rep extends beyond the domain spanned by the grid.

    \param vertices are the locations of the vertices.
    \param faces is a vector of tuples of vertex indices that describe the
    faces.
  */
  void
  setBRep(const std::vector<std::array<Number, N> >& vertices,
          const std::vector<std::array<std::size_t, N> >& faces);

  //! Wrapper for the above function.
  void
  setBRep(std::size_t numVertices, const Number* vertices,
          std::size_t numFaces, const int* faces);

  //@}
  //-------------------------------------------------------------------------
  //! \name I/O.
  //@{

  //! Display information about the state of the closest point transform.
  void
  displayInformation(std::ostream& out) const;

  //@}
  //-------------------------------------------------------------------------
  //! \name Grid operations.
  //@{

protected:

  //! Initialize the grids.
  void
  initializeGrids()
  {
    for (std::size_t n = 0; n != getNumberOfGrids(); ++n) {
      _grids[n].initialize();
    }
  }

  //@}

};

} // namespace cpt
}

#define __cpt_StateBase_ipp__
#include "stlib/cpt/StateBase.ipp"
#undef __cpt_StateBase_ipp__

#endif
