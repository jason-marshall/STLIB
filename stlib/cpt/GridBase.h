// -*- C++ -*-

/*!
  \file GridBase.h
  \brief Implements a class for the grid data.
*/

#if !defined(__cpt_GridBase_h__)
#define __cpt_GridBase_h__

// Arrays.
#include "stlib/container/MultiArray.h"
#include "stlib/container/MultiIndexRangeIterator.h"

// Computational geometry package.
#include "stlib/geom/grid/RegularGrid.h"

#include <iostream>
#include <vector>

namespace stlib
{
namespace cpt
{

USING_STLIB_EXT_ARRAY;

//! Base class for Grid.
/*!
  Implements common functionality for grids of different dimensions.
  Stores the distance, gradient of distance, closest point, and closest
  face arrays.
*/
template<std::size_t N, typename T>
class GridBase
{
  //
  // Private types.
  //

private:

  //
  // Protected types.
  //

protected:

  //! The number type.
  typedef T Number;
  //! A point in N-D.
  typedef std::array<Number, N> Point;
  //! An extent in N-D.
  typedef typename container::MultiIndexTypes<N>::SizeList SizeList;
  //! A multi-index in N-D.
  typedef typename container::MultiIndexTypes<N>::IndexList IndexList;
  //! A single index.
  typedef typename container::MultiIndexTypes<N>::Index Index;
  //! A multi-index range in N-D.
  typedef container::MultiIndexRange<N> Range;
  //! An index range iterator.
  typedef container::MultiIndexRangeIterator<N> MultiIndexRangeIterator;
  //! A lattice.
  typedef geom::RegularGrid<N, Number> Lattice;

  //
  // Data.
  //

private:

  //! Array of distances.
  container::MultiArray<Number, N> _distance;
  //! Array of gradient of distance.
  container::MultiArray<Point, N> _gradientOfDistance;
  //! Array of closest points.
  container::MultiArray<Point, N> _closestPoint;
  //! Array of closest faces.
  container::MultiArray<std::size_t, N> _closestFace;

  //! External distance array.
  Number* _distanceExternal;
  //! External gradient of distance array.
  Number* _gradientOfDistanceExternal;
  //! External closest point array.
  Number* _closestPointExternal;
  //! External closest face array.
  int* _closestFaceExternal;

  //-------------------------------------------------------------------------
  //! \name Constructors, etc.
  //@{
protected:

  //! Default constructor. Empty arrays.
  GridBase() :
    _distance(),
    _gradientOfDistance(),
    _closestPoint(),
    _closestFace(),
    _distanceExternal(0),
    _gradientOfDistanceExternal(0),
    _closestPointExternal(0),
    _closestFaceExternal(0) {}

  //! Copy constructor.  Reference the arrays.
  GridBase(const GridBase& other) :
    _distance(other._distance),
    _gradientOfDistance(other._gradientOfDistance),
    _closestPoint(other._closestPoint),
    _closestFace(other._closestFace),
    _distanceExternal(other._distanceExternal),
    _gradientOfDistanceExternal(other._gradientOfDistanceExternal),
    _closestPointExternal(other._closestPointExternal),
    _closestFaceExternal(other._closestFaceExternal) {}

  //! Construct from the grids.
  GridBase(const SizeList& extents, const IndexList& bases,
           const bool useGradientOfDistance,
           const bool useClosestPoint,
           const bool useClosestFace) :
    _distance(extents, bases),
    _gradientOfDistance(useGradientOfDistance ? extents :
                        ext::filled_array<SizeList>(0), bases),
    _closestPoint(useClosestPoint ? extents :
                  ext::filled_array<SizeList>(0), bases),
    _closestFace(useClosestFace ? extents :
                 ext::filled_array<SizeList>(0), bases),
    // Don't use the external arrays.
    _distanceExternal(0),
    _gradientOfDistanceExternal(0),
    _closestPointExternal(0),
    _closestFaceExternal(0) {}

  //! Rebuild.
  void
  rebuild(const SizeList& extents, const IndexList& bases,
          const bool useGradientOfDistance,
          const bool useClosestPoint,
          const bool useClosestFace)
  {
    _distance.rebuild(extents, bases);
    _gradientOfDistance.rebuild(useGradientOfDistance ? extents :
                                ext::filled_array<SizeList>(0), bases);
    _closestPoint.rebuild(useClosestPoint ? extents :
                          ext::filled_array<SizeList>(0), bases);
    _closestFace.rebuild(useClosestFace ? extents :
                         ext::filled_array<SizeList>(0), bases);
    // Don't use the external arrays.
    _distanceExternal = 0;
    _gradientOfDistanceExternal = 0;
    _closestPointExternal = 0;
    _closestFaceExternal = 0;
  }

  //! Construct from the grids.
  GridBase(const SizeList& extents, const IndexList& bases,
           Number* distance, Number* gradientOfDistance,
           Number* closestPoint, int* closestFace) :
    _distance(extents, bases),
    _gradientOfDistance(gradientOfDistance ? extents :
                        ext::filled_array<SizeList>(0), bases),
    _closestPoint(closestPoint ? extents :
                  ext::filled_array<SizeList>(0), bases),
    _closestFace(closestFace ? extents :
                 ext::filled_array<SizeList>(0), bases),
    _distanceExternal(distance),
    _gradientOfDistanceExternal(gradientOfDistance),
    _closestPointExternal(closestPoint),
    _closestFaceExternal(closestFace) {}

  //! Rebuild.
  void
  rebuild(const SizeList& extents, const IndexList& bases,
          Number* distance, Number* gradientOfDistance,
          Number* closestPoint, int* closestFace)
  {
    _distance.rebuild(extents, bases);
    _gradientOfDistance.rebuild(gradientOfDistance ? extents :
                                ext::filled_array<SizeList>(0), bases);
    _closestPoint.rebuild(closestPoint ? extents :
                          ext::filled_array<SizeList>(0), bases);
    _closestFace.rebuild(closestFace ? extents :
                         ext::filled_array<SizeList>(0), bases);
    _distanceExternal = distance;
    _gradientOfDistanceExternal = gradientOfDistance;
    _closestPointExternal = closestPoint;
    _closestFaceExternal = closestFace;
  }

  //! Assignment operator.
  GridBase&
  operator=(const GridBase& other);

  //@}

  //-------------------------------------------------------------------------
  //! \name Accessors.
  //@{

public:

  //! Return the grid extents.
  const SizeList&
  getExtents() const
  {
    return _distance.extents();
  }

  //! Return the grid ranges.
  Range
  getRanges() const
  {
    return _distance.range();
  }

  //! Return true if the grids are empty.
  bool
  isEmpty() const
  {
    return _distance.empty();
  }

  //! Return a const reference to the distance grid.
  const container::MultiArray<Number, N>&
  getDistance() const
  {
    return _distance;
  }

  //! Return a const reference to the gradient of the distance grid.
  const container::MultiArray<Point, N>&
  getGradientOfDistance() const
  {
    return _gradientOfDistance;
  }

  //! Return a const reference to the closest point grid.
  const container::MultiArray<Point, N>&
  getClosestPoint() const
  {
    return _closestPoint;
  }

  //! Return a const reference to the closest face grid.
  const container::MultiArray<std::size_t, N>&
  getClosestFace() const
  {
    return _closestFace;
  }

  //! Is the gradient of the distance being computed?
  bool
  isGradientOfDistanceBeingComputed() const
  {
    return ! _gradientOfDistance.empty();
  }

  //! Is the closest point being computed?
  bool
  isClosestPointBeingComputed() const
  {
    return ! _closestPoint.empty();
  }

  //! Is the closest face being computed?
  bool
  isClosestFaceBeingComputed() const
  {
    return ! _closestFace.empty();
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Manipulators.
  //@{

public:

  //! Return a reference to the distance grid.
  container::MultiArray<Number, N>&
  getDistance()
  {
    return _distance;
  }

  //! Return a reference to the gradient of the distance grid.
  container::MultiArray<Point, N>&
  getGradientOfDistance()
  {
    return _gradientOfDistance;
  }

  //! Return a reference to the closest point grid.
  container::MultiArray<Point, N>&
  getClosestPoint()
  {
    return _closestPoint;
  }

  //! Return a reference to the closest face grid.
  container::MultiArray<std::size_t, N>&
  getClosestFace()
  {
    return _closestFace;
  }

  //@}
  //-------------------------------------------------------------------------
  //! \name Mathematical operations.
  //@{

public:

  //! Calculate the signed distance, closest point, etc. for the specified grid points.
  /*!
    \return the number of distances computed and the number of distances set.
  */
  template<class Component>
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransform(const std::vector<IndexList>& indices,
                               const std::vector<Point>& positions,
                               const Component& component,
                               Number maximumDistance);

  //! Calculate the unsigned distance, closest point, etc. for the specified grid points.
  template<class Component>
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUnsigned(const std::vector<IndexList>& indices,
                                       const std::vector<Point>& positions,
                                       const Component& component,
                                       Number maximumDistance);

  //! Calculate the signed distance, closest point, etc. for the specified grid points.
  /*!
    In computing distances, use the versions that check with the
    characteristics of the component.

    \return the number of distances computed and the number of distances set.
  */
  template<class Component>
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransform(const Lattice& lattice,
                               const Range& indexRangeInLattice,
                               const Component& component,
                               Number maximumDistance);

  //! Calculate the unsigned distance, closest point, etc. for the specified grid points.
  /*!
    In computing distances, use the versions that check with the
    characteristics of the component.

    \return the number of distances computed and the number of distances set.
  */
  template<class Component>
  std::pair<std::size_t, std::size_t>
  computeClosestPointTransformUnsigned(const Lattice& lattice,
                                       const Range& indexRangeInLattice,
                                       const Component& component,
                                       Number maximumDistance);

  //! Initialize the grids.
  /*!
    Set all the distances to \c std::numeric_limits<Number>::max() in
    preparation for the distance to be
    computed.  Set the gradient of the distance and the closest points to
    std::numeric_limits<Number>::max().  Set the closest faces to
    std::numeric_limits<std::size_t>::max().
  */
  void
  initialize();

  //! Flood fill the unsigned distance.
  /*!
    If there are any points with known distance then return true and set
    the unknown distances to farAway.  Otherwise set all the distances
    to farAway and return false.
  */
  bool
  floodFillUnsigned(const Number farAway);

protected:

  //! Copy the distance into the external array (if it has been specified).
  void
  copyToDistanceExternal() const;

  //! Copy the distance, etc. into the external arrays.
  /*!
    Only the fields that are being computed will be copied.
  */
  void
  copyToExternal() const;

  //@}
  //-------------------------------------------------------------------------
  //! \name File I/O.
  //@{

public:

  void
  put(std::ostream& out) const;

  void
  displayInformation(std::ostream& out) const;

  std::size_t
  countKnownDistances(const Number maximumDistance) const;

  void
  computeMinimumAndMaximumDistances(const Number maximumDistance,
                                    Number* minimum, Number* maximum) const;

  //@}
};

} // namespace cpt
}

//
// Equality operators
//

//! Return true if the grids are equal.
/*! \relates cpt::GridBase */
template<std::size_t N, typename T>
bool
operator==(const stlib::cpt::GridBase<N, T>& a,
           const stlib::cpt::GridBase<N, T>& b);

//! Return true if the grids are not equal.
/*! \relates stlib::cpt::GridBase */
template<std::size_t N, typename T>
inline
bool
operator!=(const stlib::cpt::GridBase<N, T>& a,
           const stlib::cpt::GridBase<N, T>& b)
{
  return !(a == b);
}

#define __GridBase_ipp__
#include "stlib/cpt/GridBase.ipp"
#undef __GridBase_ipp__

#endif
