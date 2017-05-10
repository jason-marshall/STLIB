// -*- C++ -*-

/*!
  \file Face.h
  \brief Class for a face on a b-rep.
*/

#if !defined(__cpt_Face_h__)
#define __cpt_Face_h__

#include "stlib/ads/algorithm/min_max.h"

#include "stlib/ext/array.h"

#include "stlib/geom/grid/RegularGrid.h"
#include "stlib/geom/kernel/Hyperplane.h"
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
class Face;

//! Equality operator
/*! \relates Face<N,T> */
template<std::size_t N, typename T>
bool
operator==(const Face<N, T>& a, const Face<N, T>& b);

//! Inequality operator
/*! \relates Face */
template<std::size_t N, typename T>
inline
bool
operator!=(const Face<N, T>& a, const Face<N, T>& b)
{
  return !(a == b);
}

} // namespace cpt
}

#define __cpt_Face1_ipp__
#include "stlib/cpt/Face1.ipp"
#undef __cpt_Face1_ipp__

#define __cpt_Face2_ipp__
#include "stlib/cpt/Face2.ipp"
#undef __cpt_Face2_ipp__

#define __cpt_Face3_ipp__
#include "stlib/cpt/Face3.ipp"
#undef __cpt_Face3_ipp__

#endif
