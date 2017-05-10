// -*- C++ -*-

#if !defined(__geom_IndSimpSet_ipp__)
#error This file is an implementation detail of the class IndSimpSet.
#endif

namespace stlib
{
namespace geom {

template<std::size_t SpaceD, std::size_t _M, typename _T>
inline
void
IndSimpSet<SpaceD, _M, _T>::
convertFromIdentifiersToIndices
(const std::vector<std::size_t>& vertexIdentifiers) {
   assert(vertices.size() == vertexIdentifiers.size());

   //
   // Make the array of indexed simplices.
   //
   // Mapping from vertex identifiers to vertex indices.
   std::unordered_map<std::size_t, std::size_t> identifierToIndex;
   for (std::size_t i = 0; i != vertexIdentifiers.size(); ++i) {
      identifierToIndex[vertexIdentifiers[i]] = i;
   }
   // Convert to simplices of vertex indices.
   for (std::size_t i = 0; i != indexedSimplices.size(); ++i) {
      for (std::size_t m = 0; m != M + 1; ++m) {
         indexedSimplices[i][m] = identifierToIndex[indexedSimplices[i][m]];
      }
   }
}

} // namespace geom
}
