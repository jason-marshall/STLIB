// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunction1DRegularGrid.h
  \brief Functor for linear interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunction1DRegularGrid_h__)
#define __numerical_interpolation_InterpolatingFunction1DRegularGrid_h__

#include "stlib/numerical/interpolation/InterpolatingFunction1DRegularGridBase.h"

namespace stlib
{
namespace numerical
{

//! Functor for linear interpolation on a 1-D regular grid.
/*!
  \param _T The value type for the grid.
  \param _Order The interpolation order may be 1 or 3.
*/
template<typename _T, std::size_t _Order>
class InterpolatingFunction1DRegularGrid :
  public InterpolatingFunction1DRegularGridBase<_T, _Order>
{
  //
  // Types.
  //
private:

  //! The functor type.
  typedef InterpolatingFunction1DRegularGridBase<_T, _Order> Base;

public:

  //! The argument type is \c double.
  typedef typename Base::argument_type argument_type;
  //! The result type is the value type for the grid.
  typedef typename Base::result_type result_type;
  //! The value type.
  typedef typename Base::Value Value;

  //
  // Using member data.
  //
protected:

  using Base::_grid;
  using Base::_numberOfCells;
  using Base::_lowerCorner;
  using Base::_inverseWidth;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{
public:

  //! Default constructor. Invalid data.
  InterpolatingFunction1DRegularGrid() :
    Base(1, 0, 1)
  {
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param begin The first element in the grid.
    \param end One past the last element in the grid.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.
  */
  template<typename _ForwardIterator>
  InterpolatingFunction1DRegularGrid(_ForwardIterator begin,
                                     _ForwardIterator end,
                                     double lower, double upper) :
    Base(std::distance(begin, end), lower, upper)
  {
    allocate();
    Base::setGridValues(begin, end);
  }

  //! Copy constructor.
  InterpolatingFunction1DRegularGrid
  (const InterpolatingFunction1DRegularGrid& other) :
    Base(other)
  {
    allocate();
    Base::copyGrid(other);
  }

  //! Assignment operator.
  InterpolatingFunction1DRegularGrid&
  operator=(const InterpolatingFunction1DRegularGrid& other)
  {
    if (this != &other) {
      // Re-allocate memory if necessary.
      if (other._numberOfCells != _numberOfCells) {
        destroy();
        _numberOfCells = other._numberOfCells;
        allocate();
      }
      // Copy the grid values and other member variables.
      Base::copy(other);
    }
    return *this;
  }

  ~InterpolatingFunction1DRegularGrid()
  {
    destroy();
  }

private:

  void
  allocate()
  {
    _grid = new Value[_numberOfCells + 1 + 2 * Base::GhostWidth]
    + Base::GhostWidth;
  }

  void
  destroy()
  {
    if (_grid) {
      _grid -= Base::GhostWidth;
      delete[] _grid;
    }
  }

  //! @}
};

} // namespace numerical
}

#endif
