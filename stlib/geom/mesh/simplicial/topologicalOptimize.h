// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/topologicalOptimize.h
  \brief Functions to topologicalOptimize the cells in a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_topologicalOptimize_h__)
#define __geom_mesh_simplicial_topologicalOptimize_h__

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"
#include "stlib/geom/mesh/simplicial/EdgeRemoval.h"
#include "stlib/geom/mesh/simplicial/FaceRemoval.h"
#include "stlib/geom/mesh/simplicial/set.h"

#include "stlib/geom/mesh/iss/PointsOnManifold.h"

#include "stlib/ads/algorithm/skipElements.h"

namespace stlib
{
namespace geom {


//! Use edge and face removal to optimize the mesh.
/*!
  \param mesh The simplicial mesh.
  \param manifold The manifold data structure.  We pass this by const
  reference, because the topological optimization will not change the
  manifold data structure.
  \param edgeRemovalOperations Multi-set to record the edge removal operations.
  \param faceRemovalOperations Multi-set to record the face removal operations.
  \param maximumSteps The maximum allowed number of steps.

  Use the specified metric.

  \return The number of edge and face removal operations.
*/
template < class _QualityMetric,
         typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t SD >
std::size_t
topologicalOptimize
(SimpMeshRed<3, 3, T, Node, Cell, Cont>* mesh,
 const PointsOnManifold<3, 2, SD, T>* manifold,
 std::multiset<std::pair<std::size_t, std::size_t> >* edgeRemovalOperations = 0,
 std::multiset<std::pair<std::size_t, std::size_t> >* faceRemovalOperations = 0,
 std::size_t maximumSteps = std::numeric_limits<std::size_t>::max());



//! Use edge and face removal to optimize the mesh.
/*!
  \param mesh The simplicial mesh.
  \param manifold The manifold data structure.
  \param edgeRemovalOperations Multi-set to record the edge removal operations.
  \param faceRemovalOperations Multi-set to record the face removal operations.
  \param maximumSteps The maximum allowed number of steps.

  Use the modified mean ratio metric.

  \return The number of edge and face removal operations.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD >
inline
std::size_t
topologicalOptimizeUsingMeanRatio
(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
 const PointsOnManifold<N, MM, SD, T>* manifold,
 std::multiset<std::pair<std::size_t, std::size_t> >* edgeRemovalOperations = 0,
 std::multiset<std::pair<std::size_t, std::size_t> >* faceRemovalOperations = 0,
 const std::size_t maximumSteps = std::numeric_limits<std::size_t>::max()) {
   return topologicalOptimize<SimplexModMeanRatio<N, T> >(mesh, manifold,
          edgeRemovalOperations,
          faceRemovalOperations,
          maximumSteps);
}



//! Use edge and face removal to optimize the mesh.
/*!
  \param mesh The simplicial mesh.
  \param manifold The manifold data structure.
  \param edgeRemovalOperations Multi-set to record the edge removal operations.
  \param faceRemovalOperations Multi-set to record the face removal operations.
  \param maximumSteps The maximum allowed number of steps.

  Use the modified mean ratio metric.

  \return The number of edge and face removal operations.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD >
inline
std::size_t
topologicalOptimizeUsingConditionNumber
(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
 const PointsOnManifold<N, MM, SD, T>* manifold,
 std::multiset<std::pair<std::size_t, std::size_t> >* edgeRemovalOperations = 0,
 std::multiset<std::pair<std::size_t, std::size_t> >* faceRemovalOperations = 0,
 const std::size_t maximumSteps = std::numeric_limits<std::size_t>::max()) {
   return topologicalOptimize<SimplexModCondNum<N, T> >(mesh, manifold,
          edgeRemovalOperations,
          faceRemovalOperations,
          maximumSteps);
}


} // namespace geom
}

#define __geom_mesh_simplicial_topologicalOptimize3_ipp__
#include "stlib/geom/mesh/simplicial/topologicalOptimize3.ipp"
#undef __geom_mesh_simplicial_topologicalOptimize3_ipp__


#endif
