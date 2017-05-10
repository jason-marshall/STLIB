// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunctionRegularGrid.h
  \brief Functor for interpolation on a regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunctionRegularGrid_h__)
#define __numerical_interpolation_InterpolatingFunctionRegularGrid_h__

#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGridBase.h"

namespace stlib
{
namespace numerical
{


//! Functor for interpolation on a regular grid.
/*!
  This class allocates memory for the grid. It inherits all of its functionality
  from the base class numerical::InterpolatingFunctionRegularGridBase.

  See the
  \ref interpolation_InterpolatingFunctionRegularGrid "interpolating functor"
  page for documentation of this class.
*/
template < typename _T, std::size_t _Dimension, std::size_t _Order,
           bool _IsPeriodic = false >
class InterpolatingFunctionRegularGrid :
  public InterpolatingFunctionRegularGridBase<_T, _Dimension, _Order,
  _IsPeriodic>
{
  //
  // Private types.
  //
private:
  typedef InterpolatingFunctionRegularGridBase<_T, _Dimension, _Order,
          _IsPeriodic> Base;

  //
  // Public types.
  //

public:
  //! The argument type is a Cartesian point.
  typedef typename Base::argument_type argument_type;
  //! The result type is the field.
  typedef typename Base::result_type result_type;
  //! The value type.
  typedef typename Base::Value Value;
  //! A Cartesian point.
  typedef typename Base::Point Point;
  //! A bounding box.
  typedef typename Base::BBox BBox;
  //! The (multi) size type.
  typedef typename Base::SizeList SizeList;
  //! The (multi) index type.
  typedef typename Base::IndexList IndexList;
  //! The single index type.
  typedef typename Base::Index Index;

  //! The %array of function values.
  typedef container::MultiArray<Value, _Dimension> Grid;

  //
  // Data.
  //
private:
  //! The %array of function values.
  Grid _grid;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the %array of function values and the Cartesian domain.
  /*!
    \param grid The %array of function values.
    \param domain The Cartesian domain.
  */
  InterpolatingFunctionRegularGrid(const Grid& grid, const BBox& domain) :
    Base(grid.extents(), domain),
    _grid(grid)
  {
    Base::initialize(&_grid);
  }

  //! Copy constructor.
  InterpolatingFunctionRegularGrid
  (const InterpolatingFunctionRegularGrid& other) :
    Base(other),
    _grid(other._grid)
  {
    Base::initialize(&_grid);
  }

  //! Assignment operator.
  const InterpolatingFunctionRegularGrid&
  operator=(const InterpolatingFunctionRegularGrid& other)
  {
    if (this != &other) {
      Base::operator=(other);
      _grid = other._grid;
      Base::initialize(&_grid);
    }
    return *this;
  }


  // The default destructor is fine.

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  using Base::interpolate;
  using Base::operator();
  using Base::snapToGrid;

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  //! Return a const reference to the grid of function values.
  const Grid&
  grid() const
  {
    return _grid;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //! @{
public:

  //! Return a reference to the grid of function values.
  Grid&
  grid()
  {
    return _grid;
  }

  //! @}
};

} // namespace numerical
}

#endif
