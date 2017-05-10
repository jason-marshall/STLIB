// -*- C++ -*-

/*!
  \file interp_extrap.h
  \brief Interpolation/extrapolation for grids.
*/

#if !defined(__numerical_grid_interp_extrap_interp_extrap_h__)
#define __numerical_grid_interp_extrap_interp_extrap_h__

#include "stlib/container/Array.h"
#include "stlib/container/MultiArray.h"
#include "stlib/geom/grid/RegularGrid.h"

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Grid interpolation/extrapolation for a set of points.
/*!
  \param values is the vector of interpolated/extrapolated fields.

  \param positions is the vector of points in at which to perform
  interpolation/extrapolation.
  \code
  point_0_x point_0_y point_0_z
  point_1_x point_1_y point_1_z
  ...
  \endcode

  \param defaultValues are the default values for the fields.  If there are
  no nearby grid points with known values for a given point, the field
  values will be set to these default values.  The "nearby" grid points are
  the \f$ 4^N \f$ grid points that surround the point.

  \param extents are the extents of the grid.

  \param lowerCorner is the lower corner of the Cartesian domain of the grid.

  \param upperCorner is the upper corner of the Cartesian domain of the grid.

  \param distance is the distance array.  It has size
  \code
  extents[0] * ... * extents[N-1]
  \endcode
  The field values are known at grid
  points with non-negative distance and unknown at grid points with negative
  distance.  The first coordinate varies fastest.  For an X by Y grid in 2-D
  the layout is
  \code
  distance(0,0) distance(1,0) ... distance(X-1,0)
  distance(0,1) distance(1,1) ... distance(X-1,1)
  ...
  distance(0,Y-1) distance(1,Y-1) ... distance(X-1,Y-1)
  \endcode

  \param fields is the array of the vector fields.  The \c M fields for
  for a grid point are contiguous in memory.

  Template Parameters:
  - \c N is the dimension.  (1-D, 2-D and 3-D is supported.)
  - \c M is the number of fields.
  - \c T is the number type.

  Explicitly specify the dimension and the number of fields in the function
  call as these cannot be deduced from the arguments,
  i.e. use \c gridInterpExtrap<3,2>(...) for a 3-D problem with 2 fields.

  Example usage: 3-D with one field.  (N = 3, M = 1, T = double)
  \code
  const int num_points = ...;
  double values[ num_points ];
  double positions[ 3 * num_points ];
  // Set the positions.
  ...
  const double default_values[1] = { 0 };

  // A 100 x 100 x 100 grid.
  const int extents[3] = { 100, 100, 100 };
  // The grid spans the unit cube.
  const double domain[6] = { 0, 0, 0, 1, 1, 1 };

  double distance[1000000];
  double fields[1000000];
  // Set the distance and field values.
  ...
  numerical::gridInterpExtrap<3,1>(num_points, values, positions, default_values,
                                      extents, domain, distance, fields);
  \endcode
*/
template<std::size_t N, std::size_t M, typename T>
void
gridInterpExtrap(std::vector<std::array<T, M> >* values,
                 const std::vector<std::array<T, N> >& positions,
                 const std::array<T, M>& defaultValues,
                 const std::array<std::size_t, N>& extents,
                 const std::array<T, N>& lowerCorner,
                 const std::array<T, N>& upperCorner,
                 const T* distance,
                 const T* fields);

//! Wrapper for the above function.
/*!
  \note You must explicitly specify N and M.
*/
template<std::size_t N, std::size_t M, typename T>
inline
void
gridInterpExtrap(const std::size_t size,
                 T* valuesData, // array of M-tuples
                 const T* positionsData, // array of N-tuples
                 const T* defaultValuesData, // M-tuple
                 const int* extentsData, // N-tuple
                 const T* lowerCornerData, // N-tuple
                 const T* upperCornerData, // N-tuple
                 const T* distance,
                 const T* fields)
{
  // Copy from the input.
  std::vector<std::array<T, M> > values(size);
  {
    T* v = valuesData;
    for (std::size_t i = 0; i != values.size(); ++i) {
      for (std::size_t m = 0; m != M; ++m) {
        values[i][m] = *v++;
      }
    }
  }
  std::vector<std::array<T, N> > positions(size);
  for (std::size_t i = 0; i != positions.size(); ++i) {
    for (std::size_t n = 0; n != N; ++n) {
      positions[i][n] = *positionsData++;
    }
  }
  const std::array<T, M> defaultValues =
    ext::copy_array<std::array<T, M> >(defaultValuesData);
  const std::array<std::size_t, N> extents =
    ext::copy_array<std::array<std::size_t, N> >(extentsData);
  const std::array<T, N> lowerCorner =
    ext::copy_array<std::array<T, N> >(lowerCornerData);
  const std::array<T, N> upperCorner =
    ext::copy_array<std::array<T, N> >(upperCornerData);
  // Interpolate.
  gridInterpExtrap(&values, positions, defaultValues, extents,
                   lowerCorner,	upperCorner, distance, fields);
  // Copy to the output.
  for (std::size_t i = 0; i != values.size(); ++i) {
    for (std::size_t m = 0; m != M; ++m) {
      *valuesData++ = values[i][m];
    }
  }
}

} // namespace numerical
}

#define __numerical_interp_extrap_ipp__
#include "stlib/numerical/grid_interp_extrap/interp_extrap.ipp"
#undef __numerical_interp_extrap_ipp__

#endif
