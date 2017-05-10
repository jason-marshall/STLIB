// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_SimpMeshRed_ipp__)
#error This file is an implementation detail of the class SimpMeshRed.
#endif

namespace stlib
{
namespace geom {

template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
template<typename VertInIter, typename SimpInIter>
inline
void
SimpMeshRed<N, M, T, V, C, CR>::
build(VertInIter verticesBeginning, VertInIter verticesEnd,
      SimpInIter simplicesBeginning, SimpInIter simplicesEnd) {
   // Clear the nodes and cells.
   _nodes.clear();
   _cells.clear();

   // Add the vertices.
   Node node;
   for (; verticesBeginning != verticesEnd; ++verticesBeginning) {
      node.setVertex(*verticesBeginning);
      insertNode(node);
   }

   // Make an array of the node iterators.
   std::vector<Node*> nodeHandles(computeNodesSize());
   {
      NodeIterator ni = getNodesBeginning();
      typename std::vector<Node*>::iterator i = nodeHandles.begin();
      for (; ni != getNodesEnd(); ++ni, ++i) {
         *i = &*ni;
      }
   }

   // Add the cells.
   Cell c;
   for (; simplicesBeginning != simplicesEnd; ++simplicesBeginning) {
      for (std::size_t m = 0; m != M + 1; ++m) {
         c.setNode(m, nodeHandles[(*simplicesBeginning)[m]]);
      }
      insertCell(c);
   }

   // Build the cell adjacencies.
   buildCellAdjacencies();
}


//----------------------------------------------------------------------------
// Manipulators.
//----------------------------------------------------------------------------


template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
inline
void
SimpMeshRed<N, M, T, V, C, CR>::
setNodeIdentifiers() const {
   std::size_t n = 0;
   for (NodeConstIterator i = getNodesBeginning(); i != getNodesEnd();
         ++i, ++n) {
      i->setIdentifier(n);
   }
}


template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
inline
void
SimpMeshRed<N, M, T, V, C, CR>::
setCellIdentifiers() const {
   std::size_t n = 0;
   for (CellConstIterator i = getCellsBeginning(); i != getCellsEnd();
         ++i, ++n) {
      i->setIdentifier(n);
   }
}


template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
inline
typename SimpMeshRed<N, M, T, V, C, CR>::NodeIterator
SimpMeshRed<N, M, T, V, C, CR>::
insertNode(const Node& node) {
   // Check if this is the first node that we are inserting.
   const bool isFirstNode = _nodes.empty();

   // Insert a node at the end of the list.
   NodeIterator i = _nodes.insert(_nodes.end(), node);
   // Set the self iterator.
   i->setSelf(i);

   //
   // Set the identifier.
   //

   // If this is the first node inserted into the mesh.
   if (isFirstNode) {
      // The identifier is 0.
      i->setIdentifier(0);
   }
   else {
      NodeIterator prev = i;
      --prev;
      // If we have run out of positive integers.
      if (prev->getIdentifier() == std::numeric_limits<std::size_t>::max()) {
         // CONTINUE: This is problematic.  Something else may rely on the
         // identifiers.
         // Recompute all of the identifiers.
         setNodeIdentifiers();
      }
      else {
         i->setIdentifier(prev->getIdentifier() + 1);
      }
   }

   return i;
}



template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
inline
typename SimpMeshRed<N, M, T, V, C, CR>::CellIterator
SimpMeshRed<N, M, T, V, C, CR>::
insertCell(const Cell& c) {
   // Check if this is the first cell that we are inserting.
   const bool isFirstCell = areCellsEmpty();

   // Insert the cell at the end of the list.
   CellIterator ch = _cells.insert(_cells.end(), c);
   // Set the cell's self iterator.
   ch->setSelf(ch);
   // Set the cell iterators where necessary at the vertices.
   for (std::size_t m = 0; m != M + 1; ++m) {
      // CONTINUE: Should this be an assertion or an if?
#ifdef STLIB_DEBUG
      //assert(ch->getNode(m) != 0);
#endif
      if (ch->getNode(m) != 0) {
         ch->getNode(m)->insertCell(ch);
      }
   }

   //
   // Set the identifier.
   //
   // If this is the first cell inserted into the mesh.
   if (isFirstCell) {
      // The identifier is 0.
      ch->setIdentifier(0);
   }
   else {
      CellIterator prev = ch;
      --prev;
      // If we have run out of positive integers.
      if (prev->getIdentifier() == std::numeric_limits<std::size_t>::max()) {
         // Recompute all of the identifiers.
         setCellIdentifiers();
      }
      else {
         ch->setIdentifier(prev->getIdentifier() + 1);
      }
   }

   return ch;
}


//---------------------------------------------------------------------------
// Private member functions.
//---------------------------------------------------------------------------

/* CONTINUE REMOVE
   template<std::size_t N, std::size_t M, typename T,
   template<class> class V,
   template<class> class C,
   template<class,class> class CR>
   template<class Map>
   inline
   void
   SimpMeshRed<N,M,T,V,C,CR>::
   map_vertex_iterators_to_indices(Map& x) const
   {
   typedef typename Map::value_type value_type;

   x.clear();
   std::size_t i = 0;
   for (vertex_const_iterator vh = vertices_begin(); vh != vertices_end();
   ++vh, ++i) {
   x.insert(value_type(vh, i));
   }
   }
*/

template < std::size_t N, std::size_t M, typename T,
         template<class> class V,
         template<class> class C,
         template<class, class> class CR >
inline
void
SimpMeshRed<N, M, T, V, C, CR>::
buildCellAdjacencies() {
   //
   // First set all the neighbors to a null value.
   //

   // Loop over the cells of the mesh.
   CellIterator ch = getCellsBeginning();
   const CellIterator chEnd = getCellsEnd();
   for (; ch != chEnd; ++ch) {
      // Loop over the nodes.
      for (std::size_t m = 0; m != M + 1; ++m) {
         ch->setNeighbor(m, 0);
      }
   }

   // We can convert a node iterator to an integer with its identifier.
   // Thus we can represent a face with M integers.  We store these
   // integers in sorted order.  Then we can compare faces with the
   // opposite orientation.
   typedef std::array<std::size_t, M> SortedFace;
   // To store the faces in a map data structure, we need less than
   // comparison.  We use comparison of the M-D composite number.
   typedef ads::less_composite<M, SortedFace> FaceComp;
   FaceComp comp;
   comp.set(0);
   // The face map stores cell iterators with the sorted faces as keys.
   typedef std::map<SortedFace, CellIterator, FaceComp> FaceMap;
   typedef typename FaceMap::value_type FaceMapValue;
   typedef typename FaceMap::iterator FaceMapIterator;
   FaceMap faceMap(comp);

   //
   // In the following loop we build one-sided adjacency information.
   //

   typename Cell::Face face;
   SortedFace sortedFace;
   FaceMapIterator fmi;
   // Loop over the cells of the mesh.
   for (ch = getCellsBeginning(); ch != chEnd; ++ch) {
      // Loop over the faces of this cell.
      for (std::size_t fi = 0; fi != M + 1; ++fi) {

         // Get the face composed of node iterators.
         ch->getFace(fi, &face);
         // Convert the node iterators to integers.
         for (std::size_t i = 0; i != M; ++i) {
            sortedFace[i] = face[i]->getIdentifier();
         }
         // Sort the vertex identifiers of the face.
         std::sort(sortedFace.begin(), sortedFace.end());

         // Check if this sorted face has already been encountered.
         fmi = faceMap.find(sortedFace);
         // If it has not been encountered before
         if (fmi == faceMap.end()) {
            // Add it to the map.
            faceMap.insert(FaceMapValue(sortedFace, ch));
         }
         // If it has been encountered before.
         else {
            // Set the adjacency information.
            ch->setNeighbor(fi, &*fmi->second);
            // Remove the face from the map.
            faceMap.erase(fmi);
         }
      }
   }

   //
   // Finally convert the one-sided adjacency links to two-sided links.
   //

   // Neighbor iterators.
   Cell* nh;
   // Loop over the cells of the mesh.
   for (ch = getCellsBeginning(); ch != chEnd; ++ch) {
      // Loop over the faces of this cell.
      for (std::size_t m = 0; m != M + 1; ++m) {
         nh = ch->getNeighbor(m);
         if (nh != 0) {
            ch->getFace(m, &face);
            nh->setNeighbor(nh->getIndex(face), &*ch);
         }
      }
   }
}

} // namespace geom
}
