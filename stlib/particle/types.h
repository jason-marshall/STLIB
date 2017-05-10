// -*- C++ -*-

/*!
  \file particle/types.h
  \brief Data types for particle simulations.
*/

#if !defined(__particle_types_h__)
#define __particle_types_h__

#include <array>

namespace stlib
{
namespace particle
{

//! Integer data types for particle simulations.
struct IntegerTypes {
  //! The unsigned integer type for holding a code.
  typedef unsigned long long Code;
  //! A discrete coordinate.
  typedef unsigned DiscreteCoordinate;
};

//! Data types for particle simulations.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.
*/
template<typename _Float, std::size_t _Dimension>
struct TemplatedTypes {
  //! A cartesian point.
  typedef std::array<_Float, _Dimension> Point;
  //! A discrete point with integer coordinates.
  typedef std::array<IntegerTypes::DiscreteCoordinate, _Dimension>
  DiscretePoint;
};

} // namespace particle
}

#endif
