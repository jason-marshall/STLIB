// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunctionRegularGridReference.h
  \brief Functor for interpolation on a regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunctionRegularGridReference_h__)
#define __numerical_interpolation_InterpolatingFunctionRegularGridReference_h__

#include "stlib/numerical/interpolation/InterpolatingFunctionRegularGridBase.h"

namespace stlib
{
namespace numerical
{


//! Functor for interpolation on a regular grid.
/*!
  \param _T The value type.
  \param _Dimension The space dimension.
  \param _Order The interpolation order.
  \param _IsPeriodic True if the grid data is periodic. By default this is
  false.

  This class references externally allocated memory for the grid. It inherits
  all of its functionality from the base class
  numerical::InterpolatingFunctionRegularGridBase.

  See the
  \ref interpolation_InterpolatingFunctionRegularGrid "interpolating functor"
  page for documentation of this class.
*/
template < typename _T, std::size_t _Dimension, std::size_t _Order,
           bool _IsPeriodic = false >
class InterpolatingFunctionRegularGridReference :
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
  //! The constant reference %array of function values.
  typedef typename Base::GridConstRef GridConstRef;

  //
  // Data.
  //
private:
  //! The %array of function values.
  GridConstRef _grid;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Construct from the grid extents and the Cartesian domain.
  /*!
    \param extents The grid extents.
    \param domain The Cartesian domain.

    \note This constructor does not initialize the grid. You must use
    setData() to reference grid data before using the interpolation functions.
  */
  InterpolatingFunctionRegularGridReference(const SizeList& extents,
      const BBox& domain) :
    Base(extents, domain),
    _grid(0, extents)
  {
    Base::initialize(&_grid);
  }

  // The default copy constructor and destructor are fine.

private:

  // The assignment operator is not implemented.
  const InterpolatingFunctionRegularGridReference&
  operator=(const InterpolatingFunctionRegularGridReference&);

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{
public:

  using Base::interpolate;
  using Base::operator();
  using Base::snapToGrid;

  //! Set the grid data.
  void
  setData(const Value* data)
  {
    _grid.setData(data);
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{
public:

  //! Return a const reference to the grid of function values.
  const GridConstRef&
  grid() const
  {
    return _grid;
  }

  //! @}
};

} // namespace numerical
}

#endif
