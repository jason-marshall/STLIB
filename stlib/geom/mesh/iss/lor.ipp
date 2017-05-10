// -*- C++ -*-

#if !defined(__geom_mesh_iss_lor_ipp__)
#error This file is an implementation detail or lor.
#endif

namespace stlib
{
namespace geom {

//
// IndSimpSet
//

// Reorder the vertices.
template<std::size_t N, std::size_t M, typename T>
inline
void
orderVertices(IndSimpSet<N, M, T>* mesh, 
              const std::vector<std::size_t>& ranking,
              const std::vector<std::size_t>& mapping) {
   typedef IndSimpSet<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;
   
   // Update the simplex indices.
   for (std::size_t i = 0; i != mesh->indexedSimplices.size(); ++i) {
      for (std::size_t j = 0; j != M + 1; ++j) {
         mesh->indexedSimplices[i][j] = mapping[mesh->indexedSimplices[i][j]];
      }
   }
   // Make an ordered vector of vertices.
   std::vector<Vertex> tmp(ranking.size());
   for (std::size_t i = 0; i != tmp.size(); ++i) {
      tmp[i] = mesh->vertices[ranking[i]];
   }
   // Swap the vectors.
   mesh->vertices.swap(tmp);
}

// Reorder the vertices.
template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrderVertices(IndSimpSet<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::mortonOrder<_Integer>(mesh->vertices, &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrderVertices(IndSimpSet<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::axisOrder(mesh->vertices, &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrderVertices(IndSimpSet<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::randomOrder(mesh->vertices.size(), &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

// Reorder the simplices.
template<std::size_t N, std::size_t M, typename T>
inline
void
orderSimplices(IndSimpSet<N, M, T>* mesh,
               const std::vector<std::size_t>& ranking) {
   typedef IndSimpSet<N, M, T> Iss;
   typedef typename Iss::IndexedSimplex IndexedSimplex;

   // Make an ordered vector of simplices.
   std::vector<IndexedSimplex> tmp(ranking.size());
   for (std::size_t i = 0; i != tmp.size(); ++i) {
      tmp[i] = mesh->indexedSimplices[ranking[i]];
   }
   // Swap the vectors.
   mesh->indexedSimplices.swap(tmp);
}

// Reorder the simplices.
template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrderSimplices(IndSimpSet<N, M, T>* mesh) {
   typedef IndSimpSet<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;

   // Make a vector of the centroids.
   std::vector<Vertex> centroids(mesh->indexedSimplices.size());
   for (std::size_t i = 0; i != centroids.size(); ++i) {
      getCentroid(*mesh, i, &centroids[i]);
   }
   // Determine the order for the simplices.
   std::vector<std::size_t> ranking;
   lorg::mortonOrder<_Integer>(centroids, &ranking);
   orderSimplices(mesh, ranking);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrderSimplices(IndSimpSet<N, M, T>* mesh) {
   typedef IndSimpSet<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;

   // Make a vector of the centroids.
   std::vector<Vertex> centroids(mesh->indexedSimplices.size());
   for (std::size_t i = 0; i != centroids.size(); ++i) {
      getCentroid(*mesh, i, &centroids[i]);
   }
   // Determine the order for the simplices.
   std::vector<std::size_t> ranking;
   lorg::axisOrder(centroids, &ranking);
   orderSimplices(mesh, ranking);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrderSimplices(IndSimpSet<N, M, T>* mesh) {
   std::vector<std::size_t> ranking;
   lorg::randomOrder(mesh->indexedSimplices.size(), &ranking);
   orderSimplices(mesh, ranking);
}

template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrder(IndSimpSet<N, M, T>* mesh) {
   mortonOrderVertices<_Integer>(mesh);
   mortonOrderSimplices<_Integer>(mesh);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrder(IndSimpSet<N, M, T>* mesh) {
   axisOrderVertices(mesh);
   axisOrderSimplices(mesh);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrder(IndSimpSet<N, M, T>* mesh) {
   randomOrderVertices(mesh);
   randomOrderSimplices(mesh);
}

//
// IndSimpSetIncAdj
//

// Reorder the vertices.
template<std::size_t N, std::size_t M, typename T>
inline
void
orderVertices(IndSimpSetIncAdj<N, M, T>* mesh,
              const std::vector<std::size_t>& ranking,
              const std::vector<std::size_t>& mapping) {
   typedef IndSimpSetIncAdj<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;
   typedef typename Iss::IncidenceContainer IncidenceContainer;
   
   // Update the simplex indices.
   for (std::size_t i = 0; i != mesh->indexedSimplices.size(); ++i) {
      for (std::size_t j = 0; j != M + 1; ++j) {
         mesh->indexedSimplices[i][j] = mapping[mesh->indexedSimplices[i][j]];
      }
   }
   // Order the vertices.
   {
      // Make an ordered vector of vertices.
      std::vector<Vertex> tmp(ranking.size());
      for (std::size_t i = 0; i != tmp.size(); ++i) {
         tmp[i] = mesh->vertices[ranking[i]];
      }
      // Swap the vectors.
      mesh->vertices.swap(tmp);
   }
   // Order the incident simplices.
   {
      IncidenceContainer incident;
      {
         // The number of incident simplices for each vertex with the new 
         // ordering.
         std::vector<std::size_t> sizes(mesh->vertices.size());
         for (std::size_t i = 0; i != sizes.size(); ++i) {
            sizes[i] = mesh->incident.size(ranking[i]);
         }
         // Use the sizes get the right shape.
         incident.rebuild(sizes.begin(), sizes.end());
      }
      for (std::size_t i = 0; i != incident.getNumberOfArrays(); ++i) {
         std::copy(mesh->incident.begin(ranking[i]),
                   mesh->incident.end(ranking[i]),
                   incident.begin(i));
      }
      // Swap the packed arrays.
      mesh->incident.swap(incident);
   }
}

// Reorder the vertices.
template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrderVertices(IndSimpSetIncAdj<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::mortonOrder<_Integer>(mesh->vertices, &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrderVertices(IndSimpSetIncAdj<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::axisOrder(mesh->vertices, &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrderVertices(IndSimpSetIncAdj<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::randomOrder(mesh->vertices.size(), &ranking, &mapping);
   orderVertices(mesh, ranking, mapping);
}

// Reorder the simplices.
template<std::size_t N, std::size_t M, typename T>
inline
void
orderSimplices(IndSimpSetIncAdj<N, M, T>* mesh,
               const std::vector<std::size_t>& ranking,
               const std::vector<std::size_t>& mapping) {
   typedef IndSimpSetIncAdj<N, M, T> Iss;
   typedef typename Iss::IndexedSimplex IndexedSimplex;
   typedef typename Iss::AdjacencyContainer AdjacencyContainer;

   // Map the indices for the incident simplices.
   for (std::size_t i = 0; i != mesh->incident.size(); ++i) {
      mesh->incident[i] = mapping[mesh->incident[i]];
   }
   // The incident simplex indices for each vertex are sorted.
   for (std::size_t i = 0; i != mesh->incident.getNumberOfArrays(); ++i) {
      std::sort(mesh->incident.begin(i), mesh->incident.end(i));
   }
   // Map the indices for the adjacent simplices.
   for (std::size_t i = 0; i != mesh->adjacent.size(); ++i) {
      for (std::size_t j = 0; j != M + 1; ++j) {
         // Note that std::numeric_limits<std::size_t>::max() denotes a
         // boundary (no adjacent simplex).
         if (mesh->adjacent[i][j] < mesh->indexedSimplices.size()) {
            mesh->adjacent[i][j] = mapping[mesh->adjacent[i][j]];
         }
      }
   }

   // Order the indexed simplices.
   {
      // Make an ordered vector of simplices.
      std::vector<IndexedSimplex> tmp(ranking.size());
      for (std::size_t i = 0; i != tmp.size(); ++i) {
         tmp[i] = mesh->indexedSimplices[ranking[i]];
      }
      // Swap the vectors.
      mesh->indexedSimplices.swap(tmp);
   }
   // Order the adjacent simplices.
   {
      AdjacencyContainer adjacent(mesh->adjacent.size());
      for (std::size_t i = 0; i != adjacent.size(); ++i) {
         adjacent[i] = mesh->adjacent[ranking[i]];
      }
      mesh->adjacent.swap(adjacent);
   }
}

// Reorder the simplices.
template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrderSimplices(IndSimpSetIncAdj<N, M, T>* mesh) {
   typedef IndSimpSetIncAdj<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;

   // Make a vector of the centroids.
   std::vector<Vertex> centroids(mesh->indexedSimplices.size());
   for (std::size_t i = 0; i != centroids.size(); ++i) {
      getCentroid(*mesh, i, &centroids[i]);
   }
   // Determine the order for the simplices.
   std::vector<std::size_t> ranking, mapping;
   lorg::mortonOrder<_Integer>(centroids, &ranking, &mapping);
   orderSimplices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrderSimplices(IndSimpSetIncAdj<N, M, T>* mesh) {
   typedef IndSimpSetIncAdj<N, M, T> Iss;
   typedef typename Iss::Vertex Vertex;

   // Make a vector of the centroids.
   std::vector<Vertex> centroids(mesh->indexedSimplices.size());
   for (std::size_t i = 0; i != centroids.size(); ++i) {
      getCentroid(*mesh, i, &centroids[i]);
   }
   // Determine the order for the simplices.
   std::vector<std::size_t> ranking, mapping;
   lorg::axisOrder(centroids, &ranking, &mapping);
   orderSimplices(mesh, ranking, mapping);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrderSimplices(IndSimpSetIncAdj<N, M, T>* mesh) {
   std::vector<std::size_t> ranking, mapping;
   lorg::randomOrder(mesh->indexedSimplices.size(), &ranking, &mapping);
   orderSimplices(mesh, ranking, mapping);
}

template<typename _Integer, std::size_t N, std::size_t M, typename T>
inline
void
mortonOrder(IndSimpSetIncAdj<N, M, T>* mesh) {
   mortonOrderVertices<_Integer>(mesh);
   mortonOrderSimplices<_Integer>(mesh);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
axisOrder(IndSimpSetIncAdj<N, M, T>* mesh) {
   axisOrderVertices(mesh);
   axisOrderSimplices(mesh);
}

template<std::size_t N, std::size_t M, typename T>
inline
void
randomOrder(IndSimpSetIncAdj<N, M, T>* mesh) {
   randomOrderVertices(mesh);
   randomOrderSimplices(mesh);
}


} // namespace geom
}
