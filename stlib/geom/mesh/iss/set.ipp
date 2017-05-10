// -*- C++ -*-

#if !defined(__geom_mesh_iss_set_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template < std::size_t N, std::size_t M, typename T,
         class LSF, typename IntOutIter >
inline
void
determineVerticesInside(const IndSimpSet<N, M, T>& mesh,
                        const LSF& f, IntOutIter indexIterator) {
   // Loop over the vertices.
   for (std::size_t n = 0; n != mesh.vertices.size(); ++n) {
      // If the vertex is inside the object.
      if (f(mesh.vertices[n]) <= 0) {
         // Insert it into the container.
         *indexIterator++ = n;
      }
   }
}



template < std::size_t N, std::size_t M, typename T,
         class LSF, typename IntOutIter >
inline
void
determineSimplicesInside(const IndSimpSet<N, M, T>& mesh,
                         const LSF& f, IntOutIter indexIterator) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::Vertex Vertex;
   typedef typename ISS::SimplexConstIterator SimplexConstIterator;

   Vertex x;
   SimplexConstIterator s = mesh.getSimplicesBegin();
   // Loop over the simplices.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n, ++s) {
      computeCentroid(*s, &x);
      // If the centroid is inside the object.
      if (f(x) <= 0) {
         // Insert the simplex index into the container.
         *indexIterator++ = n;
      }
   }
}



// Determine the simplices which satisfy the specified condition.
template < std::size_t N, std::size_t M, typename T,
         class UnaryFunction, typename IntOutIter >
inline
void
determineSimplicesThatSatisfyCondition(const IndSimpSet<N, M, T>& mesh,
                                       const UnaryFunction& f,
                                       IntOutIter indexIterator) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::SimplexConstIterator SimplexConstIterator;

   SimplexConstIterator s = mesh.getSimplicesBegin();
   // Loop over the simplices.
   const std::size_t size = mesh.indexedSimplices.size();
   for (std::size_t n = 0; n != size; ++n, ++s) {
      // If the condition is satisfied.
      if (f(*s)) {
         // Insert the simplex index into the container.
         *indexIterator++ = n;
      }
   }
}



// Determine the simplices whose bounding boxes overlap the domain.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutIter >
void
determineOverlappingSimplices(const IndSimpSet<N, M, T>& mesh,
                              const BBox<T, N>& domain,
                              IntOutIter indexIterator) {
   typedef IndSimpSet<N, M, T> ISS;
   typedef typename ISS::Simplex Simplex;

   Simplex s;
   // Loop over the simplices.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      mesh.getSimplex(n, &s);
      // Make a bounding box around the simplex.
      if (doOverlap(domain, specificBBox<BBox<T, N> >(s.begin(), s.end()))) {
         // Insert the simplex index into the container.
         *indexIterator++ = n;
      }
   }
}



template < std::size_t N, std::size_t M, typename T,
         typename IntOutIter >
inline
void
determineSimplicesWithRequiredAdjacencies
(const IndSimpSetIncAdj<N, M, T>& mesh,
 const std::size_t minRequiredAdjacencies, IntOutIter indexIterator) {
   // For each simplex.
   for (std::size_t n = 0; n != mesh.indexedSimplices.size(); ++n) {
      // If this simplex has the minimum required number of adjacencies.
      if (numAdjacent(mesh.adjacent[n]) >= minRequiredAdjacencies) {
         // Add the simplex to the set.
         *indexIterator++ = n;
      }
   }
}



// Add the boundary vertex indices to the set.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutIter >
inline
void
determineInteriorVertices(const IndSimpSetIncAdj<N, M, T>& mesh,
                          IntOutIter indexIterator) {
   // For each vertex.
   for (std::size_t n = 0; n != mesh.vertices.size(); ++n) {
      // If this is a boundary vertex.
      if (! mesh.isVertexOnBoundary(n)) {
         // Add the vertex to the set.
         *indexIterator++ = n;
      }
   }
}



// Add the boundary vertex indices to the set.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutIter >
inline
void
determineBoundaryVertices(const IndSimpSetIncAdj<N, M, T>& mesh,
                          IntOutIter indexIterator) {
   // For each vertex.
   for (std::size_t n = 0; n != mesh.vertices.size(); ++n) {
      // If this is a boundary vertex.
      if (mesh.isVertexOnBoundary(n)) {
         // Add the vertex to the set.
         *indexIterator++ = n;
      }
   }
}



// Add the vertices which are incident to the simplices.
template < std::size_t N, std::size_t M, typename T,
         typename IntInIter, typename IntOutIter >
inline
void
determineIncidentVertices(const IndSimpSet<N, M, T>& mesh,
                          IntInIter simplexIndicesBeginning,
                          IntInIter simplexIndicesEnd,
                          IntOutIter vertexIndicesIterator) {
   std::set<std::size_t> vertexIndices;

   std::size_t m, s;
   // For each indexed simplex in the range.
   for (; simplexIndicesBeginning != simplexIndicesEnd;
         ++simplexIndicesBeginning) {
      // The index of a simplex.
      s = *simplexIndicesBeginning;
      // For each vertex index of the indexed simplex.
      for (m = 0; m != M + 1; ++m) {
         // Add the vertex index to the set.
         vertexIndices.insert(mesh.getIndexedSimplex(s)[m]);
      }
   }

   // Add the vertex indices to the output iterator.
   for (std::set<std::size_t>::const_iterator i = vertexIndices.begin();
         i != vertexIndices.end(); ++i) {
      *vertexIndicesIterator++ = *i;
   }
}



// Add the simplices (simplex indices) which are incident to the vertices.
template < std::size_t N, std::size_t M, typename T,
         typename IntInIter, typename IntOutIter >
inline
void
determineIncidentSimplices(const IndSimpSetIncAdj<N, M, T>& mesh,
                           IntInIter vertexIndicesBeginning,
                           IntInIter vertexIndicesEnd,
                           IntOutIter simplexIndicesIterator) {
   std::set<std::size_t> simplexIndices;

   std::size_t m, size, v;
   // For each vertex in the range.
   for (; vertexIndicesBeginning != vertexIndicesEnd;
         ++vertexIndicesBeginning) {
      // The index of a vertex.
      v = *vertexIndicesBeginning;
      // For each incident simplex index to the vertex.
      size = mesh.getIncidentSize(v);
      for (m = 0; m != size; ++m) {
         // Add the simplex index to the set.
         simplexIndices.insert(mesh.getIncident(v, m));
      }
   }

   // Add the simplex indices to the output iterator.
   for (std::set<std::size_t>::const_iterator i = simplexIndices.begin();
         i != simplexIndices.end(); ++i) {
      *simplexIndicesIterator++ = *i;
   }
}



// Make the complement set of indices.
template<typename IntForIter, typename IntOutIter>
inline
void
determineComplementSetOfIndices(const std::size_t upperBound,
                                IntForIter beginning, IntForIter end,
                                IntOutIter indexIterator) {
   // Loop over all integers in the range.
   for (std::size_t n = 0; n != upperBound; ++n) {
      // If the element is in the set.
      if (beginning != end && *beginning == n) {
         // Skip the element.
         ++beginning;
      }
      // If the element is not in the set.
      else {
         // Add it to b.
         *indexIterator++ = n;
      }
   }
   assert(beginning == end);
}



// Add the simplices in the component with the n_th simpex to the set.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutIter >
inline
void
determineSimplicesInComponent(const IndSimpSetIncAdj<N, M, T>& mesh,
                              const std::size_t index, IntOutIter indexIterator) {
   assert(index < mesh.indexedSimplices.size());

   // The simplices on the boundary of those that have been identified as
   // being in the component.
   std::stack<std::size_t> boundary;
   // The simplex indices in the component.
   std::set<std::size_t> component;
   std::size_t i, m, n;

   boundary.push(index);
   while (! boundary.empty()) {
      // Get a simplex from the boundary.
      n = boundary.top();
      // REMOVE
      //std::cerr << "top = " << n << "\n";
      boundary.pop();
      // Add it to the component set.
      component.insert(n);
      // Add the neighbors that aren't already in the component set to
      // the boundary.
      for (m = 0; m != M + 1; ++m) {
         i = mesh.adjacent[n][m];
         // If there is an adjacent simplex and it's not already in
         // the component set.
         if (i != std::size_t(-1) && component.count(i) == 0) {
            // REMOVE
            //std::cerr << "new adjacent = " << i << "\n";
            boundary.push(i);
         }
      }
   }

   // Copy the simplex indices in the component to the output iterator.
   for (std::set<std::size_t>::const_iterator iter = component.begin();
         iter != component.end(); ++iter) {
      *indexIterator++ = *iter;
   }
}


// Label the components of the mesh.
template<std::size_t N, std::size_t M, typename T>
std::size_t
labelComponents(const IndSimpSetIncAdj<N, M, T>& mesh,
                std::vector<std::size_t>* labels) {
   const std::size_t Unknown = std::numeric_limits<std::size_t>::max();

   labels->resize(mesh.indexedSimplices.size());
   std::fill(labels->begin(), labels->end(), Unknown);

   // The simplices on the boundary of those that have been identified as
   // being in a specified component.
   std::stack<std::size_t> boundary;
   std::size_t index;

   // Loop until all of the components have been labeled.
   std::size_t label;
   for (label = 0; true; ++label) {
      // Find the first unknown simplex.
      for (index = 0; index != labels->size() && (*labels)[index] != Unknown;
            ++index) {
      }
      // If there are no unknown simplices we are done.
      if (index == labels->size()) {
         break;
      }
      (*labels)[index] = label;
      boundary.push(index);
      while (! boundary.empty()) {
         // Get a simplex from the boundary.
         index = boundary.top();
         boundary.pop();
         // Add the neighbors that aren't already in a component to the boundary.
         for (std::size_t m = 0; m != M + 1; ++m) {
            const std::size_t adjacent = mesh.adjacent[index][m];
            // If there is an adjacent simplex and it's not already in
            // the component set.
            if (adjacent != std::size_t(-1) && (*labels)[adjacent] == Unknown) {
               (*labels)[adjacent] = label;
               boundary.push(adjacent);
            }
         }
      }
   }
   return label;
}


// Separate the connected components of the mesh.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator >
inline
void
separateComponents(IndSimpSetIncAdj<N, M, T>* mesh,
                   IntOutputIterator delimiterIterator) {
   separateComponents(mesh, delimiterIterator,
                      ads::constructTrivialOutputIterator());
}


// Separate the connected components of the mesh.
template < std::size_t N, std::size_t M, typename T,
         typename IntOutputIterator1, typename IntOutputIterator2 >
inline
void
separateComponents(IndSimpSetIncAdj<N, M, T>* mesh,
                   IntOutputIterator1 delimiterIterator,
                   IntOutputIterator2 permutationIterator) {
   typedef IndSimpSetIncAdj<N, M, T> ISS;

   // Check for the trivial case of an empty mesh.
   if (mesh->indexedSimplices.size() == 0) {
      return;
   }

   // Simplex indices, grouped by component.
   std::vector<std::size_t> indices;
   indices.reserve(mesh->indexedSimplices.size());
   // Which simplices have been identified as belonging to a particular
   // component.
   std::vector<bool> used(mesh->indexedSimplices.size(), false);

   // The beginning of the first component.
   *delimiterIterator++ = 0;

   std::size_t i;
   std::size_t n = 0;
   std::size_t oldSize, newSize;
   do {
      // Get the first unused simplex.
      while (used[n]) {
         ++n;
      }
      oldSize = indices.size();
      // Get the component.
      determineSimplicesInComponent(*mesh, n, std::back_inserter(indices));
      // Record the end delimiter for this component.
      *delimiterIterator++ = indices.size();

      // Record the simplices that are used in this component.
      newSize = indices.size();
      for (i = oldSize; i != newSize; ++i) {
         used[indices[i]] = true;
      }
   }
   while (indices.size() != mesh->indexedSimplices.size());

   // Separate the components.
   typename ISS::IndexedSimplexContainer
   indexedSimplices(mesh->indexedSimplices);
   for (i = 0; i != mesh->indexedSimplices.size(); ++i) {
      mesh->indexedSimplices[i] = indexedSimplices[indices[i]];
   }

   // Record the permutation.
   std::copy(indices.begin(), indices.end(), permutationIterator);

   // Rebuild the incidences and adjacencies.
   mesh->updateTopology();
}

} // namespace geom
}
