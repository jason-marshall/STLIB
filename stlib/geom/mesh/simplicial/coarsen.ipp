// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_coarsen_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


// CONTINUE: REMOVE
#if 0
// Coarsen the mesh using the min edge length function.
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
inline
std::size_t
coarsen(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh, const MinEdgeLength& f,
        const std::size_t maxSweeps) {
   std::size_t c;
   std::size_t count = 0;
   std::size_t sweep = 0;
   do {
      c = geom::coarsenSweep(mesh, f);
      count += c;
      ++sweep;
   }
   while (c != 0 && sweep != maxSweeps);
   return count;
}
#endif



template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t SD,
         class MinEdgeLength >
std::size_t
coarsen(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
        PointsOnManifold < N, M - 1, SD, T > * manifold,
        const MinEdgeLength& f,
        const std::size_t maxSweeps) {
   std::size_t c;
   std::size_t count = 0;
   std::size_t sweep = 0;
   do {
      c = geom::coarsenSweep(mesh, manifold, f);
      count += c;
      ++sweep;
   }
   while (c != 0 && sweep != maxSweeps);
   return count;
}



// CONTINUE: I think I should get rid of this.
#if 0
//! Coarsen the mesh using the min edge length function.
/*!
  Do not collapse edges that have both nodes on the boundary.  This
  preserves the boundary.

  \return the number of edges collapsed.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
inline
std::size_t
coarsenInterior(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                const MinEdgeLength& f, const std::size_t maxSweeps) {
   std::size_t c;
   std::size_t count = 0;
   std::size_t sweep = 0;
   do {
      c = geom::coarsenInteriorSweep(mesh, f);
      count += c;
   }
   while (c != 0 && sweep != maxSweeps);
   return count;
}
#endif


// CONTINUE: I think I should get rid of this.
#if 0
//! Coarsen the mesh using the min edge length function.
/*!
  Only collapse edges that have both nodes on the boundary.

  \return the number of edges collapsed.
*/
template < std::size_t N, std::size_t M, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class MinEdgeLength >
inline
std::size_t
coarsenBoundary(SimpMeshRed<N, M, T, Node, Cell, Cont>* mesh,
                const MinEdgeLength& f, const std::size_t maxSweeps) {
   std::size_t c;
   std::size_t count = 0;
   std::size_t sweep = 0;
   do {
      c = geom::coarsenBoundarySweep(mesh, f);
      count += c;
   }
   while (c != 0 && sweep != maxSweeps);
   return count;
}
#endif

} // namespace geom
}
