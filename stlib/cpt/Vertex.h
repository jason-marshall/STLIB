// -*- C++ -*-

#if !defined(__cpt_Vertex_h__)
#define __cpt_Vertex_h__

#include "stlib/ads/algorithm/sign.h"

#include "stlib/ext/array.h"

#include "stlib/geom/grid/RegularGrid.h"
#include "stlib/geom/kernel/Point.h"
#include "stlib/geom/polytope/IndexedEdgePolyhedron.h"
#include "stlib/geom/polytope/ScanConversionPolyhedron.h"

#include <vector>

#include <cmath>

namespace stlib
{
namespace cpt
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

template < std::size_t N, typename T = double >
class Vertex;


//! Equality operator
/*! \relates Vertex */
template<std::size_t N, typename T>
bool
operator==(const Vertex<N, T>& a, const Vertex<N, T>& b);


//! Inequality operator
/*! \relates Vertex */
template<std::size_t N, typename T>
inline
bool
operator!=(const Vertex<N, T>& a, const Vertex<N, T>& b)
{
  return !(a == b);
}


} // namespace cpt
}

#define __cpt_Vertex2_ipp__
#include "stlib/cpt/Vertex2.ipp"
#undef __cpt_Vertex2_ipp__

#define __cpt_Vertex3_ipp__
#include "stlib/cpt/Vertex3.ipp"
#undef __cpt_Vertex3_ipp__

#endif
