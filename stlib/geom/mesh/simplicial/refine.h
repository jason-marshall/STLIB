// -*- C++ -*-

/*!
  \file geom/mesh/simplicial/refine.h
  \brief Functions to refine the cells in a SimpMeshRed.
*/

#if !defined(__geom_mesh_simplicial_refine_h__)
#define __geom_mesh_simplicial_refine_h__

#include "stlib/geom/mesh/simplicial/coarsen.h"
#include "stlib/geom/mesh/simplicial/inc_opt.h"
#include "stlib/geom/mesh/simplicial/insert.h"
#include "stlib/geom/mesh/simplicial/set.h"

#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/PointsOnManifold.h"

#include "stlib/ads/functor/constant.h"

#include <set>

// CONTINUE: I use the following for debugging.
#if 0
#include "file_io.h"
#include <sstream>
#include <iomanip>
#endif

namespace stlib
{
namespace geom {

// CONTINUE: Refining is really slow when the max edge length is relatively
// small.  This is due to deep recursion.  Fix this problem.

//! Refine the mesh using the maximum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param manifold The manifold data structure.
  \param f The maximum edge length functor.  The algorithm will split
  edges above this threshold.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD,
         class MaxEdgeLength >
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
       PointsOnManifold<N, MM, SD, T>* manifold,
       const MaxEdgeLength& f);



//! Refine the mesh using the maximum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The maximum edge length functor.  The algorithm will split
  edges above this threshold.

  This simply calls the above function with a null pointer for the
  boundary manifold data structure.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MaxEdgeLength >
inline
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh, const MaxEdgeLength& f) {
   PointsOnManifold < N, M - 1, 1, T > * null = 0;
   // Call the above function with a null boundary manifold pointer.
   return refine(mesh, null, f);
}




//! Refine the mesh by splitting the specified cells.
/*!
  \param mesh The simplicial mesh.
  \param manifold The boundary manifold data structure.  By default it is null.
  \param begin The beginning of a range of cell indices.
  \param end The end of a range of cell indices.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD,
         typename IntInputIterator >
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
       PointsOnManifold<N, MM, SD, T>* manifold,
       IntInputIterator begin, IntInputIterator end);



//! Refine the mesh by splitting the specified cells.
/*!
  \param mesh The simplicial mesh.
  \param begin The beginning of a range of cell indices.
  \param end The end of a range of cell indices.

  This simply calls the above function with a null pointer for the
  boundary manifold data structure.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename IntInputIterator >
inline
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
       IntInputIterator begin, IntInputIterator end) {
   PointsOnManifold < N, M - 1, 1, T > * null = 0;
   // Call the above function with a null boundary manifold pointer.
   return refine(mesh, null, begin, end);
}



//! Refine the specified cells using the maximum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param manifold The manifold data structure.
  \param begin The beginning of a range of cell indices.
  \param end The end of a range of cell indices.
  \param f The maximum edge length functor.  The algorithm will split
  edges above this threshold.

  This function will refine the cells whose edge lengths exceed the
  maximum specified edge length.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD,
         typename IntInputIterator,
         class MaxEdgeLength >
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
       PointsOnManifold<N, MM, SD, T>* manifold,
       IntInputIterator begin, IntInputIterator end,
       const MaxEdgeLength& f);




//! Refine the specified cells using the maximum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param begin The beginning of a range of cell indices.
  \param end The end of a range of cell indices.
  \param f The maximum edge length functor.  The algorithm will split
  edges above this threshold.

  This simply calls the above function with a null pointer for the
  boundary manifold data structure.

  \return the number of edges split.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename IntInputIterator,
         class MaxEdgeLength >
inline
std::size_t
refine(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
       IntInputIterator begin, IntInputIterator end,
       const MaxEdgeLength& f) {
   PointsOnManifold < N, M - 1, 1, T > * null = 0;
   // Call the above function with a null boundary manifold pointer.
   return refine(mesh, null, begin, end, f);
}





//! Refine the mesh to better fit the boundary.
/*!
  \return the number of edges split.
*/
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class ISS >
std::size_t
refineBoundary(SimpMeshRed<2, 2, T, Node, Cell, Cont>* x,
               const ISS& boundary,
               T maxAngle,
               T minEdgeLength,
               std::size_t maxSweeps = 10);


//! Refine the mesh to better fit the boundary.
/*!
  \return the number of edges split.
*/
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class Distance,
         class ClosestPoint,
         class MaxAngle,
         class MinEdgeLength >
std::size_t
refineBoundary(SimpMeshRed<2, 2, T, Node, Cell, Cont>* x,
               const Distance& distance,
               const ClosestPoint& closestPoint,
               const MaxAngle& maxAngle,
               const MinEdgeLength& minEdgeLength,
               std::size_t maxSweeps = 10);


//! Refine and adjust the mesh using the maximum edge length function.
/*!
  \param mesh The simplicial mesh.
  \param f The maximum edge length functor.  The algorithm will split
  edges above this threshold.

  When an edge is split, incidence optimization and Laplacian smoothing
  is performed to adjust the mesh.

  \return the number of edges split.
*/
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MaxEdgeLength >
std::size_t
refineAdjust(SimpMeshRed<2, 2, T, Node, Cell, Cont>* mesh, const MaxEdgeLength& f);


} // namespace geom
}

#define __geom_mesh_simplicial_refine2_ipp__
#include "stlib/geom/mesh/simplicial/refine2.ipp"
#undef __geom_mesh_simplicial_refine2_ipp__

#define __geom_mesh_simplicial_refine3_ipp__
#include "stlib/geom/mesh/simplicial/refine3.ipp"
#undef __geom_mesh_simplicial_refine3_ipp__

#define __geom_mesh_simplicial_refine_ipp__
#include "stlib/geom/mesh/simplicial/refine.ipp"
#undef __geom_mesh_simplicial_refine_ipp__

#endif
