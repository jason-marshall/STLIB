// -*- C++ -*-

/*!
  \file Grid.h
  \brief Implements a class for the grid data.
*/

#if !defined(__cpt_Grid_h__)
#define __cpt_Grid_h__

// Local
#include "stlib/cpt/GridBase.h"
#include "stlib/cpt/Vertex.h"
#include "stlib/cpt/Face.h"

namespace stlib
{
namespace cpt
{


//! A class to hold the grid data.
template < std::size_t N, typename T = double >
class Grid;


//
// File I/O
//


//! Print the grid.
/*! \relates Grid */
template<std::size_t N, typename T>
inline
std::ostream&
operator<<(std::ostream& out, const Grid<N, T>& g)
{
  g.put(out);
  return out;
}


//
// Equality operators
//


//! Return true if the grids are equal.
/*! \relates Grid */
template<std::size_t N, typename T>
inline
bool
operator==(const Grid<N, T>& a, const Grid<N, T>& b)
{
  return (static_cast<const GridBase<N, T>&>(a) ==
          static_cast<const GridBase<N, T>&>(b));
}


//! Return true if the grids are not equal.
/*! \relates Grid */
template<std::size_t N, typename T>
inline
bool
operator!=(const Grid<N, T>& a, const Grid<N, T>& b)
{
  return !(a == b);
}


} // namespace cpt
}

#define __Grid1_ipp__
#include "stlib/cpt/Grid1.ipp"
#undef __Grid1_ipp__

#define __Grid2_ipp__
#include "stlib/cpt/Grid2.ipp"
#undef __Grid2_ipp__

#define __Grid3_ipp__
#include "stlib/cpt/Grid3.ipp"
#undef __Grid3_ipp__

#endif
