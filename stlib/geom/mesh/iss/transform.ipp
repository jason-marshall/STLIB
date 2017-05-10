// -*- C++ -*-

#if !defined(__geom_mesh_iss_transform_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
pack(IndSimpSet<N, M, T>* mesh, IntOutputIterator usedVertexIndices) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::IndexedSimplexConstIterator
   IndexedSimplexConstIterator;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;
   typedef typename ISS::VertexContainer VertexContainer;

   // The vertices that are used.
   std::vector<bool> used(mesh->vertices.size(), false);

   // Loop over the simplices.
   std::size_t m;
   for (IndexedSimplexConstIterator s = mesh->indexedSimplices.begin();
         s != mesh->indexedSimplices.end(); ++s) {
      // Note which vertices are used.
      for (m = 0; m != M + 1; ++m) {
         used[(*s)[m]] = true;
      }
   }

   // The packed vertices.
   VertexContainer
   packedVertices(std::count(used.begin(), used.end(), true));

   //
   // Determine the vertex positions and indices for the new mesh.
   //

   // vertex index.
   std::size_t vi = 0;
   // This array maps the old vertex indices to the packed vertex indices.
   std::vector<std::size_t> indices(mesh->vertices.size(),
                                    std::numeric_limits<std::size_t>::max());
   // Loop over the vertices.
   for (std::size_t n = 0; n != mesh->vertices.size(); ++n) {
      // If the vertex is used in the packed mesh.
      if (used[n]) {
         // Record in the container of the used vertex indices.
         *usedVertexIndices++ = n;
         // Calculate its index in the packed mesh.
         indices[n] = vi;
         // Add the vertex to the packed mesh.
         packedVertices[vi] = mesh->vertices[n];
         ++vi;
      }
   }

   // Swap the old vertices with the packed vertices.
   mesh->vertices.swap(packedVertices);

   // Map the vertex indices from the old mesh to the packed mesh.
   for (IndexedSimplexIterator s = mesh->indexedSimplices.begin();
         s != mesh->indexedSimplices.end(); ++s) {
      for (m = 0; m != M + 1; ++m) {
         (*s)[m] = indices[(*s)[m]];
      }
   }

   // Update any auxilliary topological information.
   mesh->updateTopology();
}



template<std::size_t N, typename T>
inline
void
orientPositive(IndSimpSet<N, N, T>* mesh) {
   typedef IndSimpSet<N, N, T> ISS;
   typedef typename ISS::Simplex Simplex;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;

   Simplex s;
   SimplexJac<N, T> sj;
   // For each simplex.
   for (IndexedSimplexIterator i = mesh->indexedSimplices.begin();
         i != mesh->indexedSimplices.end(); ++i) {
      mesh->getSimplex(i, &s);
      sj.setFunction(s);
      // If the content is negative.
      if (sj.getDeterminant() < 0) {
         // Reverse its orientation.
         reverseOrientation(&*i);
      }
   }

   // Update any auxilliary topological information.
   mesh->updateTopology();
}



template<std::size_t N, std::size_t M, typename T>
inline
void
reverseOrientation(IndSimpSet<N, M, T>* mesh) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::IndexedSimplexIterator IndexedSimplexIterator;

   // For each simplex.
   for (IndexedSimplexIterator i = mesh->indexedSimplices.begin();
         i != mesh->indexedSimplices.end(); ++i) {
      // Reverse the orientation of the indexed simplemesh->
      reverseOrientation(&*i);
   }

   // Update any auxilliary topological information.
   mesh->updateTopology();
}



template<std::size_t N, std::size_t M, typename T>
inline
void
orient(IndSimpSetIncAdj<N, M, T>* mesh) {
   typedef IndSimpSetIncAdj<N, M, T> ISS;
   typedef typename ISS::IndexedSimplexFace IndexedSimplexFace;

   const std::size_t simplicesSize = mesh->indexedSimplices.size();

   // Initially flag every simplex as un-oriented.
   std::vector<bool> isSimplexOriented(simplicesSize, false);

   // The number of oriented faces.
   std::size_t numOriented = 0;

   std::size_t n, m, nu, mu;
   IndexedSimplexFace f, g;
   std::stack<std::size_t> boundary;
   // While not all simplices have been oriented.
   while (numOriented < simplicesSize) {

      // Pick a simplex which currently is un-oriented to have a known
      // orientation.
      for (n = 0; n != simplicesSize && isSimplexOriented[n]; ++n)
         ;
      assert(n != simplicesSize);
      // Set the orientation of this simplemesh->
      isSimplexOriented[n] = true;
      ++numOriented;

      // Add the simplex to the boundary.
      boundary.push(n);

      // Loop until the boundary is empty.
      while (! boundary.empty()) {

         // Get a simplex from the boundary.
         n = boundary.top();
         boundary.pop();

         // For each adjacent simplemesh->
         for (m = 0; m != M + 1; ++m) {
            // The m_th adjacent simplemesh->
            nu = mesh->adjacent[n][m];
            // If this is a boundary face or the adjacent simplex already has
            // known orientation, do nothing.
            if (nu == std::size_t(-1) || isSimplexOriented[nu]) {
               continue;
            }

            mu = mesh->getMirrorIndex(n, m);
            getFace(mesh->indexedSimplices[n], m, &f);
            getFace(mesh->indexedSimplices[nu], mu, &g);
            reverseOrientation(&g);
            if (! haveSameOrientation(f, g)) {
               // Reverse the orientation of the neighbor.
               mesh->reverseOrientation(nu);
               // Add it to the boundary.
               boundary.push(nu);
               // The orientation is now known.
               isSimplexOriented[nu] = true;
               ++numOriented;
            }
         }
      }
   }
}



// Transform each vertex in the range with the specified function.
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class UnaryFunction >
inline
void
transform(IndSimpSet<N, M, T>* mesh,
          IntForIter begin, IntForIter end, const UnaryFunction& f) {
   std::transform
   (ads::constructArrayIndexingIterator(begin, mesh->vertices.begin()),
    ads::constructArrayIndexingIterator(end, mesh->vertices.begin()),
    ads::constructArrayIndexingIterator(begin, mesh->vertices.begin()),
    f);
}



// Transform each vertex in the mesh with the specified function.
template < std::size_t N, std::size_t M, typename T,
         class UnaryFunction >
inline
void
transform(IndSimpSet<N, M, T>* mesh, const UnaryFunction& f) {
   std::transform
   (ads::constructArrayIndexingIterator(ads::constructIntIterator<std::size_t>(0),
                                        mesh->vertices.begin()),
    ads::constructArrayIndexingIterator(ads::constructIntIterator<std::size_t>
                                        (mesh->vertices.size()),
                                        mesh->vertices.begin()),
    ads::constructArrayIndexingIterator(ads::constructIntIterator<std::size_t>(0),
                                        mesh->vertices.begin()),
    f);
}



// Transform each vertex in the range with the closest point in the normal direction.
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class ISS >
inline
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          IntForIter beginning, IntForIter end,
          const ISS_SD_ClosestPointDirection<ISS>& f) {
   for (; beginning != end; ++beginning) {
      applyBoundaryCondition(mesh, f, *beginning);
   }
}



// Transform each vertex in the mesh with the closest point in the normal direction.
template<std::size_t N, std::size_t M, typename T, class ISS>
inline
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          const ISS_SD_ClosestPointDirection<ISS>& f) {
   const std::size_t size = mesh->vertices.size();
   for (std::size_t n = 0; n != size; ++n) {
      boundary_condition(mesh, f, n);
   }
}



// Transform each vertex in the range with the closer point in the normal direction.
template < std::size_t N, std::size_t M, typename T,
         typename IntForIter, class ISS >
inline
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          IntForIter beginning, IntForIter end,
          const ISS_SD_CloserPointDirection<ISS>& f) {
   for (; beginning != end; ++beginning) {
      applyBoundaryCondition(mesh, f, *beginning);
   }
}



// Transform each vertex in the mesh with the closer point in the normal direction.
template<std::size_t N, std::size_t M, typename T, class ISS>
inline
void
transform(IndSimpSetIncAdj<N, M, T>* mesh,
          const ISS_SD_CloserPointDirection<ISS>& f) {
   const std::size_t size = mesh->vertices.size();
   for (std::size_t n = 0; n != size; ++n) {
      applyBoundaryCondition(mesh, f, n);
   }
}



template<std::size_t N, std::size_t M, typename T>
inline
void
removeLowAdjacencies(IndSimpSetIncAdj<N, M, T>* mesh,
                     const std::size_t minRequiredAdjacencies) {
   // Make sure that the min adjacency requirement is in the right range.
   assert(minRequiredAdjacencies <= M + 1);

   std::array < std::size_t, M + 2 > adjacencyCounts;
   std::size_t hi, lo;
   std::vector<std::size_t> ss;
   // Loop until the simplices with low adjacencies are gone.
   do {
      //
      // Get rid of the simplices with low adjacency counts.
      //

      // Get the set of simplices with sufficient adjacencies.
      ss.clear();
      determineSimplicesWithRequiredAdjacencies(*mesh, minRequiredAdjacencies,
            std::back_inserter(ss));
      // If not all simplices have sufficient adjacencies.
      if (ss.size() != mesh->indexedSimplices.size()) {
         // Remove those with low adjacencies.
         IndSimpSetIncAdj<N, M, T> y;
         buildFromSubsetSimplices(*mesh, ss.begin(), ss.end(), &y);
         y.updateTopology();
         mesh->swap(y);
      }

      //
      // Check the adjacency counts.
      //
      countAdjacencies(*mesh, &adjacencyCounts);
      lo = std::accumulate(adjacencyCounts.begin(),
                           adjacencyCounts.begin() + minRequiredAdjacencies,
                           0);
      hi = std::accumulate(adjacencyCounts.begin() + minRequiredAdjacencies,
                           adjacencyCounts.end(),
                           0);
   }
   while (lo != 0 && hi != 0);
}

} // namespace geom
}
