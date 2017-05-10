// -*- C++ -*-

/*!
  \file distinct_points.h
  \brief From a set of points, generate an indexed set of distinct points.
*/

#if !defined(__geom_mesh_iss_distinct_points_h__)
#define __geom_mesh_iss_distinct_points_h__

#include "stlib/geom/mesh/iss/IndSimpSet.h"

#include "stlib/geom/kernel/BBox.h"
#include "stlib/geom/orq/CellArray.h"
#include "stlib/ads/functor/Dereference.h"

#include <iostream>
#include <vector>

#include <cassert>

namespace stlib
{
namespace geom {


//-----------------------------------------------------------------------------
/*! \defgroup iss_distinct_points Identify distinct points and remove duplicate points. */
//@{

//! From a set of points, generate an indexed set of distinct points.
/*!
  \param pointsBeginning is the beginning of a range of points.
  \param pointsEnd is the end of a range of points.
  \param distinctPointsOutput  The distinct points will be written to this
  iterator.
  \param indicesOutput  For each input point, there is an index into the
  container of distinct points.
  \param minDistance is the minimum distance separating distinct points.

  Template parameters:
  - \c N is the space dimension.
  - \c PtForIter is a forward iterator for Cartesian points.
  - \c PtOutIter is an output iterator for Cartesian points.
  - \c IntOutIter in an output iterator for integers.
  - \c T is the number type.
*/
template < std::size_t N, typename PtForIter, typename PtOutIter, typename IntOutIter,
         typename T >
void
buildDistinctPoints(PtForIter pointsBeginning, PtForIter pointsEnd,
                    PtOutIter distinctPointsOutput,
                    IntOutIter indicesOutput,
                    const T minDistance);


//! From a set of points, generate an indexed set of distinct points.
/*!
  \param pointsBeginning is the beginning of a range of points.
  \param pointsEnd is the end of a range of points.
  \param distinctPoints  The distinct points will be written to this iterator.
  \param indices  For each input point, there is an index into the
  container of distinct points.

  Template parameters:
  - \c N is the space dimension.
  - \c PtForIter is a forward iterator for Cartesian points.
  - \c PtOutIter is an output iterator for Cartesian points.
  - \c IntOutIter in an output iterator for integers.

  This function chooses an appropriate minimum distance and then calls
  the above buildDistinctPoints() function.
*/
template<std::size_t N, typename PtForIter, typename PtOutIter, typename IntOutIter>
void
buildDistinctPoints(PtForIter pointsBeginning, PtForIter pointsEnd,
                    PtOutIter distinctPoints, IntOutIter indices);


//! Remove duplicate vertices.
template<std::size_t N, std::size_t M, typename T>
void
removeDuplicateVertices(IndSimpSet<N, M, T>* x, T minDistance);


//! Remove duplicate vertices.
/*!
  This function chooses an appropriate minimum distance and then calls
  the above removeDuplicateVertices() function.
*/
template<std::size_t N, std::size_t M, typename T>
void
removeDuplicateVertices(IndSimpSet<N, M, T>* x);

//@}

} // namespace geom
}

#define __geom_mesh_iss_distinct_points_ipp__
#include "stlib/geom/mesh/iss/distinct_points.ipp"
#undef __geom_mesh_iss_distinct_points_ipp__

#endif
