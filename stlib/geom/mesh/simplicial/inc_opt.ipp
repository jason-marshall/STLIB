// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_inc_opt_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

template<typename T>
inline
T
computeIncidentAngleDeviation(const T actual, const T ideal) {
   return std::abs(ideal / actual - 1);
}



template<class SMR>
inline
typename SMR::Number
computeIdealIncidence(const typename SMR::Node* node,
                      std::integral_constant<std::size_t, 2>
                      /*space_dimension*/) {
   if (node->isOnBoundary()) {
      return computeIncidentCellsAngle<SMR>(node) /
             (numerical::Constants<typename SMR::Number>::Pi() / 3);
   }
   // else
   return 6.0;
}


template<class SMR>
inline
typename SMR::Number
computeIdealIncidence(const typename SMR::Node* node,
                      std::integral_constant<std::size_t, 3>
                      /*space_dimension*/) {
   return computeIncidentCellsAngle<SMR>(node) /
          (numerical::Constants<typename SMR::Number>::Pi() / 3);
}


template<class SMR>
inline
typename SMR::Number
computeIdealIncidence(const typename SMR::Node* node) {
   return computeIdealIncidence<SMR>
     (node, std::integral_constant<std::size_t, SMR::N>());
}


// Works for any space dimension.
template<typename SMR, typename OutputIterator>
inline
bool
flipToOptimizeIncidence(const typename SMR::Face& face,
                        const std::size_t norm, OutputIterator affectedNodes) {
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::Number Number;

   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   const std::size_t M = SMR::M;

   Node* node;
   CellIterator cell;
   std::size_t i;
   std::size_t topIncidence, bottomIncidence, leftIncidence, rightIncidence;
   Number top_ii, bot_ii, left_ii, right_ii;
   Number t1, t2, t3, t4, newDeviation, oldDeviation;

   if (! isOnBoundary<SMR>(face)) {
      // The cell and index that define the face.
      cell = face.first;
      i = face.second;
      //
      // The cell incidence and identifiers for the four nodes.
      //
      node = cell->getNode(i);
      topIncidence = node->getCellsSize();
      top_ii = computeIdealIncidence<SMR>(node);

      node = cell->getNeighbor(i)->getNode(cell->getMirrorIndex(i));
      bottomIncidence = node->getCellsSize();
      bot_ii = computeIdealIncidence<SMR>(node);

      node = cell->getNode((i + 1) % (M + 1));
      leftIncidence = node->getCellsSize();
      left_ii = computeIdealIncidence<SMR>(node);

      node = cell->getNode((i + 2) % (M + 1));
      rightIncidence = node->getCellsSize();
      right_ii = computeIdealIncidence<SMR>(node);

      t1 = computeIncidentAngleDeviation(Number(topIncidence + 1), top_ii);
      t2 = computeIncidentAngleDeviation(Number(bottomIncidence + 1), bot_ii);
      t3 = computeIncidentAngleDeviation(Number(leftIncidence - 1), left_ii);
      t4 = computeIncidentAngleDeviation(Number(rightIncidence - 1), right_ii);
      if (norm == 0) {
         newDeviation = ads::max(t1, t2, t3, t4);
      }
      else if (norm == 1) {
         newDeviation = t1 + t2 + t3 + t4;
      }
      else if (norm == 2) {
         newDeviation = t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4;
      }
      else {
         assert(false);
      }

      t1 = computeIncidentAngleDeviation(Number(topIncidence), top_ii);
      t2 = computeIncidentAngleDeviation(Number(bottomIncidence), bot_ii);
      t3 = computeIncidentAngleDeviation(Number(leftIncidence), left_ii);
      t4 = computeIncidentAngleDeviation(Number(rightIncidence), right_ii);
      if (norm == 0) {
         oldDeviation = ads::max(t1, t2, t3, t4);
      }
      else if (norm == 1) {
         oldDeviation = t1 + t2 + t3 + t4;
      }
      else if (norm == 2) {
         oldDeviation = t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4;
      }
      else {
         assert(false);
      }

      // Use the l2 norm of the difference between the incidence and
      // the desired incidence to determine whether to flip.
      if (newDeviation < oldDeviation *
            (1. - 10. * std::numeric_limits<Number>::epsilon())) {
         // Record the affected nodes.
         *affectedNodes++ = cell->getNode(i);
         *affectedNodes++ = cell->getNeighbor(i)->getNode(cell->getMirrorIndex(i));
         *affectedNodes++ = cell->getNode((i + 1) % (M + 1));
         *affectedNodes++ = cell->getNode((i + 2) % (M + 1));

         // CONTINUE
         /*
         std::cerr << node->vertex() << " "
           << topIncidence << " "
           << top_ii << "\n"
           << node->vertex() << " "
           << bottomIncidence << " "
           << bot_ii << "\n"
           << node->vertex() << " "
           << leftIncidence << " "
           << left_ii << "\n"
           << node->vertex() << " "
           << rightIncidence << " "
           << right_ii << "\n";
         std::cerr << "oldDeviation = " << oldDeviation
           << ", newDeviation = " << newDeviation << "\n";
         std::cerr << cell->getNode(i)->vertex() << "\n";
         std::cerr << cell->getNeighbor(i)->getNode(cell->getMirrorIndex(i))->vertex() << "\n";
         std::cerr << cell->getNode((i+1)%(M+1))->vertex() << "\n";
         std::cerr << cell->getNode((i+2)%(M+1))->vertex() << "\n";
         */

         // Flip the face.
         flip<SMR>(face);

         return true;
      }
   }

   return false;
}



// Works for any space dimension.
template<typename SMR, typename OutputIterator>
inline
bool
flipToOptimizeIncidence(const typename SMR::Face& f,
                        const std::vector<typename SMR::Number>&
                        computeIdealIncidence,
                        const std::size_t norm,
                        OutputIterator affectedNodes) {
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::Number Number;

   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   const std::size_t M = SMR::M;

   Node* node;
   CellIterator cell;
   std::size_t i;
   std::size_t topIncidence, bottomIncidence, leftIncidence, rightIncidence;
   std::size_t top_id, bot_id, left_id, right_id;
   Number t1, t2, t3, t4, newDeviation, oldDeviation;

   if (! isOnBoundary<SMR>(f)) {
      // The cell and index that define the face.
      cell = f.first;
      i = f.second;
      //
      // The cell incidence and identifiers for the four nodes.
      //
      node = cell->getNode(i);
      topIncidence = node->getCellsSize();
      top_id = node->getIdentifier();

      node = cell->getNeighbor(i)->getNode(cell->getMirrorIndex(i));
      bottomIncidence = node->getCellsSize();
      bot_id = node->getIdentifier();

      node = cell->getNode((i + 1) % (M + 1));
      leftIncidence = node->getCellsSize();
      left_id = node->getIdentifier();

      node = cell->getNode((i + 2) % (M + 1));
      rightIncidence = node->getCellsSize();
      right_id = node->getIdentifier();

      // CONTINUE REMOVE;
      /*
      std::cout << "incidences = "
            << topIncidence << " "
            << bottomIncidence << " "
            << leftIncidence << " "
            << rightIncidence << "\n";
      */


      t1 = computeIncidentAngleDeviation(Number(topIncidence + 1),
                                         computeIdealIncidence[top_id]);
      t2 = computeIncidentAngleDeviation(Number(bottomIncidence + 1),
                                         computeIdealIncidence[bot_id]);
      t3 = computeIncidentAngleDeviation(Number(leftIncidence - 1),
                                         computeIdealIncidence[left_id]);
      t4 = computeIncidentAngleDeviation(Number(rightIncidence - 1),
                                         computeIdealIncidence[right_id]);
      if (norm == 0) {
         newDeviation = ads::max(t1, t2, t3, t4);
      }
      else if (norm == 1) {
         newDeviation = t1 + t2 + t3 + t4;
      }
      else if (norm == 2) {
         newDeviation = t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4;
      }
      else {
         assert(false);
      }

      t1 = computeIncidentAngleDeviation(Number(topIncidence),
                                         computeIdealIncidence[top_id]);
      t2 = computeIncidentAngleDeviation(Number(bottomIncidence),
                                         computeIdealIncidence[bot_id]);
      t3 = computeIncidentAngleDeviation(Number(leftIncidence),
                                         computeIdealIncidence[left_id]);
      t4 = computeIncidentAngleDeviation(Number(rightIncidence),
                                         computeIdealIncidence[right_id]);
      if (norm == 0) {
         oldDeviation = ads::max(t1, t2, t3, t4);
      }
      else if (norm == 1) {
         oldDeviation = t1 + t2 + t3 + t4;
      }
      else if (norm == 2) {
         oldDeviation = t1 * t1 + t2 * t2 + t3 * t3 + t4 * t4;
      }
      else {
         assert(false);
      }

      // CONTINUE REMOVE;
      //std::cout << "new = " << newDeviation << " " << "old = " << oldDeviation << "\n";

      // Use the l2 norm of the difference between the incidence and
      // the desired incidence to determine whether to flip.
      if (newDeviation < oldDeviation *
            (1. - 10. * std::numeric_limits<Number>::epsilon())) {
         // Record the affected nodes.
         *affectedNodes++ = cell->getNode(i);
         *affectedNodes++ = cell->getNeighbor(i)->
                            getNode(cell->getMirrorIndex(i));
         *affectedNodes++ = cell->getNode((i + 1) % (M + 1));
         *affectedNodes++ = cell->getNode((i + 2) % (M + 1));

         // Flip the face.
         flip<SMR>(f);

         return true;
      }
   }

   return false;
}



// Wrapper for the above function.
// This is used when it is not necessary to keep track of the affected nodes.
template<typename SMR>
inline
bool
flipToOptimizeIncidence(const typename SMR::Face& f,
                        const std::vector<typename SMR::Number>&
                        computeIdealIncidence,
                        const std::size_t norm) {
   return flipToOptimizeIncidence<SMR>(f, computeIdealIncidence, norm,
                                       ads::constructTrivialOutputIterator());
}



template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
std::size_t
incidenceOptimize(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
                  const std::size_t norm,
                  const std::size_t numSweeps) {
   typedef SimpMeshRed<N, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::NodeConstIterator NodeConstIterator;
   typedef typename SMR::FaceIterator FaceIterator;

   // Calculate the desired incidence for each node.
   mesh->setNodeIdentifiers();
   std::vector<T> idealIncidence(mesh->computeNodesSize());
   for (NodeConstIterator i = mesh->getNodesBeginning();
         i != mesh->getNodesEnd(); ++i) {
      idealIncidence[i->getIdentifier()] = computeIdealIncidence<SMR>(&*i);
   }

   std::size_t c, count = 0;
   std::size_t sweep = 0;
   do {
      c = 0;
      // Loop over the faces.
      for (FaceIterator face = mesh->getFacesBeginning();
            face != mesh->getFacesEnd(); ++face) {
         if (flipToOptimizeIncidence<SMR>(*face, idealIncidence, norm)) {
            ++c;
         }
      }
      count += c;
      ++sweep;
   }
   while (sweep != numSweeps && c != 0);

   return count;
}



template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
incidenceOptimize(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
                  typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::
                  Node* node,
                  typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::
                  NodePointerSet* nodes,
                  const std::size_t norm) {
   typedef SimpMeshRed<N, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::FaceSet FaceSet;

   // Get the faces of the incident cells.
   FaceSet faces;
   determineFacesOfIncidentCells(*mesh, node, &faces);
   // Loop over the faces.
   for (typename FaceSet::const_iterator face = faces.begin();
         face != faces.end(); ++face) {
      // Try a flip on the face.
      // If the face is flipped, the affected nodes will be inserted into
      // the set of nodes.
      if (flipToOptimizeIncidence<SMR>(*face, norm,
                                       std::inserter(*nodes, nodes->end()))) {
         // CONTINUE
         //std::cerr << "flipToOptimizeIncidence() succeeded.\n";
         return;
      }
   }
   // If we reach here, none of the potential flips improved the incidences.
   // Remove the node from the set.
   nodes->erase(node);
}



// Optimize the incidence relationships for the set of nodes.
template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
incidenceOptimize(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
                  typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::
                  NodePointerSet* nodes,
                  const std::size_t norm) {
   while (! nodes->empty()) {
      // CONTINUE
      /*
      std::cerr << "nodes.size() = " << nodes.size() << "\n";
      {
        std::ofstream file("problem.vtu");
        write_vtk_xml(file, *mesh);
      }
      */
      incidenceOptimize(mesh, *nodes->begin(), nodes, norm);
   }
}


template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont >
inline
void
incidenceOptimize(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
                  const typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::
                  Node* node,
                  const std::size_t norm) {
   typedef SimpMeshRed<N, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::NodePointerSet NodePointerSet;

   NodePointerSet nodes;
   nodes.insert(node);
   incidenceOptimize(mesh, node, nodes, norm);
}


} // namespace geom
}
