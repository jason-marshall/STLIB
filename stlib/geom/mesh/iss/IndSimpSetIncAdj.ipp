// -*- C++ -*-

#if !defined(__geom_IndSimpSetIncAdj_ipp__)
#error This file is an implementation detail of the class IndSimpSetIncAdj.
#endif

namespace stlib
{
namespace geom {

template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
bool
IndSimpSetIncAdj<SpaceD, _M, _T>::
isVertexOnBoundary(const std::size_t vertexIndex) const {
   std::size_t si, m, n;
   // For each simplex incident to this vertex.
   for (IncidenceConstIterator sii = incident.begin(vertexIndex);
         sii != incident.end(vertexIndex); ++sii) {
      // The simplex index.
      si = *sii;
      // The number of the vertex in the simplex.
      n = ext::index(Base::indexedSimplices[si], vertexIndex);
      // For each face incident to the vertex.
      for (m = 0; m != M + 1; ++m) {
         if (m != n) {
            // If the face is on the boundary.
            if (adjacent[si][m] == std::numeric_limits<std::size_t>::max()) {
               return true;
            }
         }
      }
   }
   return false;
}

} // namespace geom
}
