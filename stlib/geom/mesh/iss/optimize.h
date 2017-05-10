// -*- C++ -*-

/*!
  \file geom/mesh/iss/optimize.h
  \brief Implements file I/O operations for IndSimpSet.
*/

#if !defined(__geom_mesh_iss_optimize_h__)
#define __geom_mesh_iss_optimize_h__

#include "stlib/geom/mesh/iss/IndSimpSetIncAdj.h"
#include "stlib/geom/mesh/iss/accessors.h"
#include "stlib/geom/mesh/iss/geometry.h"
#include "stlib/geom/mesh/iss/PointsOnManifold.h"

#include "stlib/geom/mesh/simplex/ComplexWithFreeVertexOnManifold.h"
#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include "stlib/geom/kernel/ParametrizedLine.h"
#include "stlib/geom/kernel/ParametrizedPlane.h"

#include "stlib/ads/functor/Identity.h"
#include "stlib/ads/iterator/IntIterator.h"

#include "stlib/numerical/optimization/staticDimension/QuasiNewton.h"
#include "stlib/numerical/optimization/staticDimension/PenaltyQuasiNewton.h"
#include "stlib/numerical/optimization/staticDimension/Simplex.h"

namespace stlib
{
namespace geom {

//-----------------------------------------------------------------------------
/*! \defgroup iss_optimize Optimization
  These functions perform geometric optimization of vertex positions.
  One can sweep over all vertices or a set of vertices.  One can apply
  either unconstrained optimization or optimization subject to a constant
  content constraint.
*/
//@{



//-------------------------------------------------------------------------
// Interior
//-------------------------------------------------------------------------


//! Optimize the position of the interior vertices.
/*!
  Make \c numSweeps optimization sweeps over the interior vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T >
void
geometricOptimizeInterior
(IndSimpSetIncAdj<N, N, T>* mesh, std::size_t numSweeps = 1);



//! Make \c numSweeps optimization sweeps over the given interior vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter >
void
geometricOptimizeInterior(IndSimpSetIncAdj<N, N, T>* mesh,
                          IntForIter begin, IntForIter end,
                          std::size_t numSweeps = 1);


//! Optimize the position of the interior vertices.
/*!
  Make \c numSweeps optimization sweeps over the interior vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T>
void
geometricOptimizeInteriorUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh, const std::size_t numSweeps = 1) {
   geometricOptimizeInterior<SimplexModMeanRatio>(mesh, numSweeps);
}


//! Optimize the position of the interior vertices.
/*!
  Make \c numSweeps optimization sweeps over the interior vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T>
void
geometricOptimizeInteriorUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 const std::size_t numSweeps = 1) {
   geometricOptimizeInterior<SimplexModCondNum>(mesh, numSweeps);
}




//-------------------------------------------------------------------------
// Boundary
//-------------------------------------------------------------------------


//! Optimize the position of all boundary vertices.
/*!
  Make \c numSweeps optimization sweeps over the boundary vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         std::size_t SD >
void
geometricOptimizeBoundary(IndSimpSetIncAdj<N, N, T>* mesh,
                          PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
                          std::size_t numSweeps = 1);



//! Make \c numSweeps optimization sweeps over the given boundary vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         typename T,
         typename IntForIter, std::size_t SD >
void
geometricOptimizeBoundary(IndSimpSetIncAdj<2, 2, T>* mesh,
                          IntForIter begin, IntForIter end,
                          PointsOnManifold<2, 1, SD, T>* boundaryManifold,
                          std::size_t numSweeps = 1);


//! Optimize the position of all boundary vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeBoundaryUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeBoundary<SimplexModMeanRatio>
   (mesh, boundaryManifold, numSweeps);
}


//! Optimize the position of all boundary vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeBoundaryUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeBoundary<SimplexModCondNum>
   (mesh, boundaryManifold, numSweeps);
}




//-------------------------------------------------------------------------
// Mixed
//-------------------------------------------------------------------------



//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the boundary vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.  If specified (nonzero),
  the boundary vertices will be optimized.  If not, they will not be altered.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T, std::size_t SD >
void
geometricOptimize(IndSimpSetIncAdj<N, N, T>* mesh,
                  PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
                  std::size_t numSweeps = 1);



//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.  If specified (nonzero),
  the boundary vertices will be optimized.  If not, they will not be altered.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModMeanRatio>(mesh, boundaryManifold, numSweeps);
}


//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.  If specified (nonzero),
  the boundary vertices will be optimized.  If not, they will not be altered.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModCondNum>(mesh, boundaryManifold, numSweeps);
}




//-------------------------------------------------------------------------
// Other methods.
//-------------------------------------------------------------------------



//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T, std::size_t SD >
void
geometricOptimizeWithBoundaryCondition
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 std::size_t numSweeps = 1);



//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeWithBoundaryConditionUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithBoundaryCondition<SimplexModMeanRatio>
   (mesh, boundaryManifold, numSweeps);
}


//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T, std::size_t SD>
void
geometricOptimizeWithBoundaryConditionUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithBoundaryCondition<SimplexModCondNum>
   (mesh, boundaryManifold, numSweeps);
}










//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         class BoundaryCondition >
void
geometricOptimizeWithCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                               const BoundaryCondition& condition,
                               std::size_t numSweeps = 1);

//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         class BoundaryCondition >
inline
void
geometricOptimizeWithConditionUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 const BoundaryCondition& condition,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithCondition<SimplexModMeanRatio>
   (mesh, condition, numSweeps);
}

//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         class BoundaryCondition >
inline
void
geometricOptimizeWithConditionUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 const BoundaryCondition& condition,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithCondition<SimplexModCondNum>
   (mesh, condition, numSweeps);
}




//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the specified quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T >
inline
void
geometricOptimize(IndSimpSetIncAdj<N, N, T>* mesh,
                  const std::size_t numSweeps = 1) {
   typedef typename IndSimpSetIncAdj<N, N, T>::Vertex Vertex;
   geometricOptimizeWithCondition<QF>(mesh, ads::identity<Vertex>(), numSweeps);
}


//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified mean ratio quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T>
inline
void
geometricOptimizeUsingMeanRatio(IndSimpSetIncAdj<N, N, T>* mesh,
                                const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModMeanRatio>(mesh, numSweeps);
}

//! Optimize the position of all vertices.
/*!
  Make \c numSweeps optimization sweeps over the vertices with
  the modified condition number quality function.

  \param mesh is the indexed simplex set.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template<std::size_t N, typename T>
inline
void
geometricOptimizeUsingConditionNumber(IndSimpSetIncAdj<N, N, T>* mesh,
                                      const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModCondNum>(mesh, numSweeps);
}











//! Make \c numSweeps optimization sweeps over the given vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param boundaryManifold is the boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, std::size_t SD >
void
geometricOptimizeWithBoundaryCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                                       IntForIter begin, IntForIter end,
                                       PointsOnManifold < N, N - 1, SD, T > *
                                       boundaryManifold,
                                       std::size_t numSweeps = 1);


//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param boundaryManifold The boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeWithBoundaryConditionUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithBoundaryCondition<SimplexModMeanRatio>
   (mesh, begin, end, boundaryManifold, numSweeps);
}


//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param boundaryManifold The boundary manifold.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, std::size_t SD >
inline
void
geometricOptimizeWithBoundaryConditionUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 PointsOnManifold < N, N - 1, SD, T > * boundaryManifold,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithBoundaryCondition<SimplexModCondNum>
   (mesh, begin, end, boundaryManifold, numSweeps);
}










//! Make \c numSweeps optimization sweeps over the given vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
void
geometricOptimizeWithCondition(IndSimpSetIncAdj<N, N, T>* mesh,
                               IntForIter begin, IntForIter end,
                               const BoundaryCondition& condition,
                               std::size_t numSweeps = 1);

//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithConditionUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 std::size_t numSweeps = 1) {
   geometricOptimizeWithCondition<SimplexModMeanRatio>(mesh, begin, end,
         condition, numSweeps);
}

//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified condition number quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithConditionUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithCondition<SimplexModCondNum>(mesh, begin, end,
         condition, numSweeps);
}






//! Make \c numSweeps optimization sweeps over the given vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter >
void
geometricOptimize(IndSimpSetIncAdj<N, N, T>* mesh,
                  IntForIter begin, IntForIter end, const std::size_t numSweeps = 1) {
   typedef typename IndSimpSetIncAdj<N, N, T>::Vertex Vertex;
   geometricOptimizeWithCondition<QF>(mesh, begin, end, ads::identity<Vertex>(),
                                      numSweeps);
}

//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeUsingMeanRatio(IndSimpSetIncAdj<N, N, T>* mesh,
                                IntForIter begin, IntForIter end,
                                const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModMeanRatio>(mesh, begin, end, numSweeps);
}

//! Optimize the position of a set of vertices.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified condition number quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeUsingConditionNumber(IndSimpSetIncAdj<N, N, T>* mesh,
                                      IntForIter begin, IntForIter end,
                                      const std::size_t numSweeps = 1) {
   geometricOptimize<SimplexModCondNum>(mesh, begin, end, numSweeps);
}






//! Make \c numSweeps constrained optimization sweeps over the given vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
void
geometricOptimizeWithConditionConstrained
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 const T maxConstraintError,
 std::size_t numSweeps = 1);

//! Optimize the position of a set of vertices subject to a constant content constraint.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithConditionConstrainedUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 const T maxConstraintError,
 std::size_t numSweeps = 1) {
   geometricOptimizeWithConditionConstrained<SimplexModMeanRatio>
   (mesh, begin, end, condition, maxConstraintError, numSweeps);
}

//! Optimize the position of a set of vertices subject to a constant content constraint.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified condition number quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param condition is a transformation that is applied after the optimization.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter, class BoundaryCondition >
inline
void
geometricOptimizeWithConditionConstrainedUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const BoundaryCondition& condition,
 const T maxConstraintError,
 const std::size_t numSweeps = 1) {
   geometricOptimizeWithConditionConstrained<SimplexModCondNum>
   (mesh, begin, end, condition, maxConstraintError, numSweeps);
}







//! Make \c numSweeps constrained optimization sweeps over the given vertices with the quality function given as a template parameter.
/*!
  \c QF is a simplex quality functor.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < template<std::size_t, typename> class QF,
         std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeConstrained(IndSimpSetIncAdj<N, N, T>* mesh,
                             IntForIter begin, IntForIter end,
                             const T maxConstraintError,
                             const std::size_t numSweeps = 1) {
   typedef typename IndSimpSetIncAdj<N, N, T>::Vertex Vertex;
   geometricOptimizeWithConditionConstrained<QF>
   (mesh, begin, end, ads::identity<Vertex>(), maxConstraintError, numSweeps);
}

//! Optimize the position of a set of vertices subject to a constant content constraint.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified mean ratio quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeConstrainedUsingMeanRatio
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const T maxConstraintError,
 const std::size_t numSweeps = 1) {
   geometricOptimizeConstrained<SimplexModMeanRatio>
   (mesh, begin, end, maxConstraintError, numSweeps);
}

//! Optimize the position of a set of vertices subject to a constant content constraint.
/*!
  Make \c numSweeps optimization sweeps over the given vertices with
  the modified condition number quality function.
  \c IntForIter is a const iterator on vertex indices.

  \param mesh is the indexed simplex set.
  \param begin is the beginning of the vertex indices.
  \param end is the end of the vertex indices.
  \param maxConstraintError is the maximum allowed constraint error.
  \param numSweeps is the number of sweeps performed over the vertices.
  By default it is 1.
*/
template < std::size_t N, typename T,
         typename IntForIter >
inline
void
geometricOptimizeConstrainedUsingConditionNumber
(IndSimpSetIncAdj<N, N, T>* mesh,
 IntForIter begin, IntForIter end,
 const T maxConstraintError,
 const std::size_t numSweeps = 1) {
   geometricOptimizeConstrained<SimplexModCondNum>
   (mesh, begin, end, maxConstraintError, numSweeps);
}

//@}

} // namespace geom
}

#define __geom_mesh_iss_optimize_ipp__
#include "stlib/geom/mesh/iss/optimize.ipp"
#undef __geom_mesh_iss_optimize_ipp__

#endif
