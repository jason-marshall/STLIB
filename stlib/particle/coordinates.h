// -*- C++ -*-

/*!
  \file particle/coordinates.h
  \brief Convert N-D floating-point coordinates to discrete coordinates.
*/

#if !defined(__particle_coordinates_h__)
#define __particle_coordinates_h__

#include "stlib/particle/types.h"
#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace particle
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Convert N-D floating-point coordinates to discrete coordinates.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.
  \param _Periodic Whether the domain is periodic.

  \par
  The domain for the floating-point coordinates is an axis-aligned box.
  Fitted to this is an array of cells whose extent is <em>2<sup>N</sup></em>
  in each dimension, where <em>N</em> is the number of levels of refinement
  (from a single cell).
*/
template<typename _Float, std::size_t _Dimension, bool _Periodic>
class DiscreteCoordinates
{
  //
  // Types.
  //
public:

  //! A discrete coordinate.
  typedef IntegerTypes::DiscreteCoordinate DiscreteCoordinate;
  //! A cartesian point.
  typedef typename TemplatedTypes<_Float, _Dimension>::Point Point;
  //! A discrete point with integer coordinates.
  typedef typename TemplatedTypes<_Float, _Dimension>::DiscretePoint
  DiscretePoint;

  //
  // Member data.
  //
private:

  //! The lower corner of the Cartesian domain.
  Point _lowerCorner;
  //! The lengths of the Cartesian domain.
  Point _lengths;
  //! The number of levels of refinement.
  /*! Determine the number of levels that we can have with the interleaved
    code. Leave one of the bits so that there is a number past the largest
    valid code. This makes iteration easier. */
  std::size_t _numLevels;
  //! The number of cells in each dimension that are actually used.
  /*! The extents are no greater than 2<sup>_numLevels</sup>. */
  DiscretePoint _cellExtents;
  //! Cell extents represented in floating-point format.
  /*! This allows us to avoid casting to a float for certain operations. */
  Point _cellExtentsFloat;
  //! The scaling factors in transforming to cells are the extents divided by the lengths.
  Point _scaling;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Default constructor invalidates the data members.
  /*! You must call initialize() before using this data structure. */
  DiscreteCoordinates();

  //! Construct from a domain and the cell lengths.
  /*! This constructor just calls initialize(). */
  DiscreteCoordinates(const geom::BBox<_Float, _Dimension>& domain,
                      const _Float cellLength_)
  {
    initialize(domain, cellLength_);
  }

  //! Initialize from a domain and the cell lengths.
  /*! For best performance for plain domains, all points should lie in the
    interior of the box. */
  void
  initialize(const geom::BBox<_Float, _Dimension>& domain,
             const _Float cellLength_)
  {
    _initialize(domain, cellLength_, std::integral_constant<bool, _Periodic>());
  }

  //! Set the number of levels.
  /*! This operation is only used for coarsening trees for visualization.
    It may change the domain in order to accomodate the requested cell
    sizes. */
  void
  setLevels(std::size_t numLevels);

private:

  //! Initialize for a plain domain.
  void
  _initialize(geom::BBox<_Float, _Dimension> domain, _Float cellLength_,
              std::false_type /*Periodic*/);

  //! Initialize for a periodic domain.
  void
  _initialize(geom::BBox<_Float, _Dimension> domain, _Float cellLength_,
              std::true_type /*Periodic*/);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the lower corner of the Cartesian domain.
  const Point&
  lowerCorner() const
  {
    return _lowerCorner;
  }

  //! The lengths of the Cartesian domain.
  const Point&
  lengths() const
  {
    return _lengths;
  }

  //! Return the lengths of a cell.
  Point
  cellLengths() const
  {
    return _lengths / _cellExtentsFloat;
  }

  //! Return the number of levels of refinement.
  std::size_t
  numLevels() const
  {
    return _numLevels;
  }

  //! Return the cell extents in each dimension.
  const DiscretePoint&
  cellExtents() const
  {
    return _cellExtents;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Calculate the discrete coordinates.
  DiscretePoint
  discretize(const Point& p) const;

  //! Return the index of the highest level whose cell length matches or exceeds the specified length.
  std::size_t
  level(_Float length) const;

  //@}
};

} // namespace particle
}

#define __particle_coordinates_tcc__
#include "stlib/particle/coordinates.tcc"
#undef __particle_coordinates_tcc__

#endif
