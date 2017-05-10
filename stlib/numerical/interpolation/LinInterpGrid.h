// -*- C++ -*-

/*!
  \file numerical/interpolation/LinInterpGrid.h
  \brief Functor for linear interpolation on a regular grid.
*/

#if !defined(__numerical_interpolation_linear_h__)
#define __numerical_interpolation_linear_h__

#include "stlib/container/MultiArray.h"
#include "stlib/geom/grid/RegularGrid.h"

#include <boost/call_traits.hpp>

#include <functional>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Functor for linear interpolation on a regular grid.
/*!
  \param N is the space dimension.
  \param F is the field type.  By default it is double.
  \param T is the number type.  By default it is double.
*/
template < std::size_t N, typename F = double, typename T = double >
class LinInterpGrid :
  // The following is complicated, but the argument type is a Cartesian point
  // and the return type is the field.
  public std::unary_function <
  const typename geom::RegularGrid<N, T>::Point&,
  typename boost::call_traits<F>::param_type>
{
  //
  // Private types.
  //

private:

  //! The base class.
  typedef std::unary_function <
  const typename geom::RegularGrid<N, T>::Point&,
        typename boost::call_traits<F>::param_type>
        base_type;

  //
  // Public types.
  //

public:

  //! The argument type is a Cartesian point.
  typedef typename base_type::argument_type argument_type;
  //! The result type is the field.
  typedef typename base_type::result_type result_type;

  //! The number type.
  typedef T Number;
  //! The field type.
  typedef F Field;

  //! A regular grid.
  typedef geom::RegularGrid<N, Number> Grid;
  //! The field array.
  typedef container::MultiArray<Field, N> FieldArray;

  //! A Cartesian point.
  typedef typename Grid::Point Point;
  //! A bounding box.
  typedef typename Grid::BBox BBox;

  //! The (multi) index type.
  typedef typename FieldArray::IndexList IndexList;
  //! The single index type.
  typedef typename FieldArray::Index Index;

  //
  // Data.
  //

private:

  //! The field array.
  FieldArray _fields;
  //! The Cartesian grid.
  Grid _grid;

public:

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //! @{

  //! Default constructor.
  LinInterpGrid() :
    _fields(),
    _grid() {}

  //! Construct from the field array and the Cartesian domain.
  /*!
    \param fields is the array of fields.
    \param domain is the Cartesian domain.
  */
  LinInterpGrid(const container::MultiArray<Field, N>& fields,
                const BBox& domain) :
    _fields(fields),
    _grid(_fields.extents(), domain) {}

  //! Build from the field array and the Cartesian domain.
  /*!
    \param fields is the array of fields.
    \param domain is the Cartesian domain.
  */
  void
  build(const container::MultiArray<Field, N>& fields, const BBox& domain)
  {
    _fields = fields;
    _grid = Grid(_fields.extents(), domain);
  }

  //! Copy constructor.
  LinInterpGrid(const LinInterpGrid& x) :
    _fields(x._fields),
    _grid(x._grid) {}

  //! Assignment operator.
  LinInterpGrid&
  operator=(const LinInterpGrid& x)
  {
    if (this != &x) {
      _fields = x._fields;
      _grid = x._grid;
    }
    return *this;
  }

  //! Destructor.  Deletes memory only if it was allocated internally.
  ~LinInterpGrid() {}

  //! @}
  //--------------------------------------------------------------------------
  //! \name Interpolation.
  //! @{

  //! Interpolate the field at the specified point.
  result_type
  operator()(argument_type x) const;

  //! @}
  //--------------------------------------------------------------------------
  //! \name Static member functions.
  //! @{

  //! Return the dimension of the space.
  static
  int
  space_dimension()
  {
    return N;
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //! @{

  //! Return the field array.
  const FieldArray&
  fields() const
  {
    return _fields;
  }

  //! Return the Cartesian domain.
  const BBox&
  domain() const
  {
    return _grid.domain();
  }

  //! @}
  //--------------------------------------------------------------------------
  //! \name Manipulators
  //! @{

  //! Return the field array.
  /*!
    \warning Don't resize the fields array using this accessors.  Use the
    resize() member function instead.
  */
  FieldArray&
  fields()
  {
    return _fields;
  }

  //! Set the Cartesian domain.
  void
  set_domain(const BBox& domain);

  //! Resize the fields array.
  void
  resize(const IndexList& extents);

  //! @}
};

} // namespace numerical
}

#define __numerical_interpolation_LinInterpGrid_ipp__
#include "stlib/numerical/interpolation/LinInterpGrid.ipp"
#undef __numerical_interpolation_LinInterpGrid_ipp__

#endif
