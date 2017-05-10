// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/coarsen.h
  \brief Functions to coarsen the cells in a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_coarsen_h__)
#define __geom_mesh_simplicial_coarsen_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"
#include "stlib/geom/mesh/simplicial/geometry.h"
#include "stlib/geom/mesh/simplicial/topology.h"
#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/set.h"

#include "stlib/geom/mesh/iss/PointsOnManifold.h"

#include "stlib/numerical/constants.h"

// CONTINUE: I use the following for debugging.
#if 0
#include "file_io.h"
#include <sstream>
#include <iomanip>
#endif

namespace stlib
{
namespace geom {

//! Coarsen the mesh using the minimum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
  edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
  the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
  the resulting elements is not less than the initial quality times this value.
  \param manifold The manifold data structure.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
  default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.

  Only edges which are a common minimum edge are collapsed.  Also, an edge may
  not be collapsed if its endpoints are mirror nodes.  (Doing so would tangle
  the mesh.) An interior edge may not be collapsed if both of its incident
  nodes lay on the boundary.  (Doing so would change the topology of the mesh.)

  A boundary edge may not be collapsed if both of its nodes
  are corner features.  (Doing so would change the corner features.)
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t SD,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<2, 2, T, Node, Cell, Cont>* mesh,
        const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        PointsOnManifold<2, 1, SD, T>* manifold,
        std::size_t maxSweeps = 0);


//! Coarsen the mesh using the min edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
  edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
  the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
  the resulting elements is not less than the initial quality times this value.
  \param cornerDeviation Any boundary node whose angle deviates
  from \f$ \pi \f$ more than this value will be considered a corner feature.
  If not specified, no vertices will be considered corner features.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
  default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<2, 2, T, Node, Cell, Cont>* mesh, const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        T cornerDeviation = -1, std::size_t maxSweeps = 0);






//! Coarsen the mesh using the min edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
  edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
  the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
  the resulting elements is not less than the initial quality times this value.
  \param manifold The manifold data structure.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
  default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.

  Only edges which are a common minimum edge are collapsed.  Also, an edge may
  not be collapsed if its endpoints are mirror nodes.  (Doing so would tangle
  the mesh.) An interior edge may not be collapsed if both of its incident
  nodes lay on the boundary.  (Doing so would change the topology of the mesh.)
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t SD,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<3, 2, T, Node, Cell, Cont>* mesh,
        const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        PointsOnManifold<3, 2, SD, T>* manifold,
        std::size_t maxSweeps = 0);


//! Coarsen the mesh using the min edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
    edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
  the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
  the resulting elements is not less than the initial quality times this value.
  \param maxDihedralAngleDeviation The maximum dihedral angle deviation
    (from straight) for a surface feature.  The rest are edge features.
    If not specified, all interior edges will be set as surface features.
  \param maxSolidAngleDeviation Solid angles that deviate
    more than this value (from \f$2 \pi\f$) are corner features.
    If not specified, this criterion will not be used to identify corners.
  \param maxBoundaryAngleDeviation
    If the angle deviation (from \f$\pi\f$) between two boundary edges
    exceeds this value, it will be set as a corner feature.  If not specified,
    this criterion will not be used to identify corners on the boundary.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
    default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<3, 2, T, Node, Cell, Cont>* mesh, const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        T maxDihedralAngleDeviation = -1,
        T maxSolidAngleDeviation = -1,
        T maxBoundaryAngleDeviation = -1,
        std::size_t maxSweeps = 0);






//! Coarsen the mesh using the min edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
  edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
  the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
  the resulting elements is not less than the initial quality times this value.
  \param manifold The manifold data structure.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
  default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.

  Only edges which are a common minimum edge are collapsed.  Also, an edge may
  not be collapsed if its endpoints are mirror nodes.  (Doing so would tangle
  the mesh.) An interior edge may not be collapsed if both of its incident
  nodes lay on the boundary.  (Doing so would change the topology of the mesh.)
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t SD,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<3, 3, T, Node, Cell, Cont>* mesh,
        const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        PointsOnManifold<3, 2, SD, T>* manifold,
        std::size_t maxSweeps = 0);


//! Coarsen the mesh using the min edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The minimum edge length functor.  The algorithm will try to collapse
    edges below this threshold.
  \param minimumAllowedQuality A collapse is only allowed if the quality of
    the resulting elements is not less than this value.
  \param qualityFactor A collapse is only allowed if the quality of
    the resulting elements is not less than the initial quality times this
    value.
  \param maxDihedralAngleDeviation The maximum dihedral angle deviation
    (from straight) for a surface feature.  The rest are edge features.
    If not specified, all interior edges will be set as surface features.
  \param maxSolidAngleDeviation Solid angles that deviate
    more than this value (from \f$2 \pi\f$) are corner features.
    If not specified, this criterion will not be used to identify corners.
  \param maxBoundaryAngleDeviation
    If the angle deviation (from \f$\pi\f$) between two boundary edges
    exceeds this value, it will be set as a corner feature.  If not specified,
    this criterion will not be used to identify corners on the boundary.
  \param maxSweeps The maximum number of sweeps over the vertices.  By
    default, there is no limit on the number of sweeps.

  \return the number of edges collapsed.
*/
template < class QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<3, 3, T, Node, Cell, Cont>* mesh, const MinEdgeLength& f,
        T minimumAllowedQuality, T qualityFactor,
        T maxDihedralAngleDeviation = -1,
        T maxSolidAngleDeviation = -1,
        T maxBoundaryAngleDeviation = -1,
        std::size_t maxSweeps = 0);












//! Collapse edges to remove the specified cells.
/*!
  \param mesh The simplicial mesh.
  \param begin The beginning of a range of cells.
  \param end The end of a range of cells.

  \return the number of edges collapsed.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename IntInIter >
std::size_t
coarsen(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
        IntInIter begin, IntInIter end);






} // namespace geom
}

#define __geom_mesh_simplicial_coarsenN_ipp__
#include "stlib/geom/mesh/simplicial/coarsenN.ipp"
#undef __geom_mesh_simplicial_coarsenN_ipp__

#define __geom_mesh_simplicial_coarsen2_ipp__
#include "stlib/geom/mesh/simplicial/coarsen2.ipp"
#undef __geom_mesh_simplicial_coarsen2_ipp__

#define __geom_mesh_simplicial_coarsen3_ipp__
#include "stlib/geom/mesh/simplicial/coarsen3.ipp"
#undef __geom_mesh_simplicial_coarsen3_ipp__

#define __geom_mesh_simplicial_coarsen_ipp__
#include "stlib/geom/mesh/simplicial/coarsen.ipp"
#undef __geom_mesh_simplicial_coarsen_ipp__

#endif
