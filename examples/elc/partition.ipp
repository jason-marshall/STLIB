// -*- C++ -*-

#if !defined(__elc_test_partition_ipp__)
#error This file is an implementation detail.
#endif


template < std::size_t N, typename T, typename Vertex, typename IndexedElement1,
         typename IndexedElement2 >
inline
void
partitionMesh(const std::size_t numberOfProcessors, const std::size_t rank,
              const std::vector<Vertex>& vertices,
              const std::vector<IndexedElement1>& elements,
              std::vector<int>* localIdentifiers,
              std::vector<Vertex>* localVertices,
              std::vector<IndexedElement2>* localElements) {
   assert(numberOfProcessors > 0);
   assert(0 <= rank && rank < numberOfProcessors);

   // Get the centroids of the elements.
   std::vector<Vertex> centroids(elements.size());
   {
      Vertex x;
      for (std::size_t i = 0; i != elements.size(); ++i) {
         std::fill(x.begin(), x.end(), 0.0);
         for (std::size_t m = 0; m != N; ++m) {
            x += vertices[elements[i][m]];
         }
         x /= T(N);
         centroids[i] = x;
      }
   }

   // Make an array of element identifiers.
   std::vector<int> elementIdentifiers(elements.size());
   for (std::size_t i = 0; i != elementIdentifiers.size(); ++i) {
      elementIdentifiers[i] = i;
   }

   // Partition.
   std::vector<int*> identifierPartition(numberOfProcessors + 1);
   concurrent::rcb<N>(numberOfProcessors, elementIdentifiers.size(),
                      &elementIdentifiers[0], &identifierPartition[0],
                      reinterpret_cast<T*>(&centroids[0]));

   // The local element identifiers.
   array::ArrayRef<int>
   localElementIdentifiers(identifierPartition[rank],
                           identifierPartition[rank + 1] -
                           identifierPartition[rank]);

   // The local elements.
   localElements->resize(localElementIdentifiers.size());
   for (std::size_t i = 0; i != localElements->size(); ++i) {
      // Note that these are global vertex indices.
      // REMOVE
      /*
      assert(0 <= localElementIdentifiers[i] &&
         localElementIdentifiers[i] < elements.size());
      */
      const IndexedElement1& e = elements[localElementIdentifiers[i]];
      IndexedElement2& le = (*localElements)[i];
      for (std::size_t m = 0; m != le.size(); ++m) {
         le[m] = e[m];
      }
   }

   //
   // The local vertices.
   //

   // Determine which vertices are used in the local mesh.
   std::vector<bool> used(vertices.size(), false);
   for (std::size_t i = 0; i != localElements->size(); ++i) {
      for (std::size_t n = 0; n != N; ++n) {
         used[(*localElements)[i][n]] = true;
      }
   }
   const std::size_t numberOfVertices =
      std::count(used.begin(), used.end(), true);

   // Make the local vertex and vertex identifier arrays.
   localIdentifiers->resize(numberOfVertices);
   localVertices->resize(numberOfVertices);
   std::vector<int> globalToLocal(used.size(), -1);
   {
      int j = 0;
      // Loop over all vertices.
      for (std::size_t i = 0; i != used.size(); ++i) {
         // If the i_th vertex is used in the local mesh.
         if (used[i]) {
            // Add the positions and identifier to the local mesh.
            (*localVertices)[j] = vertices[i];
            (*localIdentifiers)[j] = i;
            globalToLocal[i] = j;
            ++j;
         }
      }
      assert(j == numberOfVertices);
   }

   // Convert the elements from global to local indices.
   for (std::size_t i = 0; i != localElements->size(); ++i) {
      for (std::size_t n = 0; n != N; ++n) {
         (*localElements)[i][n] = globalToLocal[(*localElements)[i][n]];
      }
   }
}
