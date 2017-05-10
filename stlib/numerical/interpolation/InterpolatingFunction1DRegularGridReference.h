// -*- C++ -*-

/*!
  \file numerical/interpolation/InterpolatingFunction1DRegularGridReference.h
  \brief Functor for linear interpolation on a 1-D regular grid.
*/

#if !defined(__numerical_interpolation_InterpolatingFunction1DRegularGridReference_h__)
#define __numerical_interpolation_InterpolatingFunction1DRegularGridReference_h__

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
class InterpolatingFunction1DRegularGridReference :
  public InterpolatingFunction1DRegularGridBase<_T, _Order>
{
  //
  // Types.
  //
private:

  //! The functor type.
  typedef InterpolatingFunction1DRegularGridBase<_T, _Order> Base;

public:

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
  /*! \name Constructors etc.
    We use the default destructor. The memory for the grid is not deleted.
  */
  //! @{
public:

  //! Default constructor. Invalid data.
  InterpolatingFunction1DRegularGridReference() :
    Base(1, 0, 1)
  {
  }

  //! Construct from the grid and the Cartesian domain.
  /*!
    \param grid Pointer to the grid data.
    \param gridSize The grid size.
    \param lower The lower bound of the Cartesian domain.
    \param upper The upper bound of the Cartesian domain.

    The grid memory will be referenced, not copied.
  */
  InterpolatingFunction1DRegularGridReference
  (Value* grid, const std::size_t gridSize, const double lower,
   const double upper) :
    Base(gridSize, lower, upper)
  {
    _grid = grid;
    Base::setGhost();
  }

  //! Copy constructor. Shallow copy. Reference the same memory.
  InterpolatingFunction1DRegularGridReference
  (const InterpolatingFunction1DRegularGridReference& other) :
    Base(other)
  {
  }

  //! Assignment operator. Deep copy. The grids must be the same size.
  InterpolatingFunction1DRegularGridReference&
  operator=(const InterpolatingFunction1DRegularGridReference& other)
  {
    if (this != &other) {
      Base::copy(other);
    }
    return *this;
  }

  //! @}
};

} // namespace numerical
}

#endif
