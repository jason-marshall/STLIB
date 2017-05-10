// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_refine2_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {

//! Return the maximum edge length.
/*!
  Set \c i to the maximum edge index.
*/
template<class SMR>
inline
typename SMR::Number
computeMaximumEdgeLength(const typename SMR::CellConstIterator c,
                         std::size_t* i) {
   typedef typename SMR::Number Number;

   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   std::size_t a = 0, b = 0;
   Number d = c->computeMaximumEdgeLength(&a, &b);
   (*i) = 0;
   if ((*i) == a) {
      ++(*i);
   }
   if ((*i) == b) {
      ++(*i);
   }
   return d;
}


//! Return the maximum edge length.
template<class SMR>
inline
typename SMR::Number
computeMaximumEdgeLength(const typename SMR::CellConstIterator c) {
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   std::size_t a, b;
   return c->computeMaximumEdgeLength(&a, &b);
}



//! Return the maximum edge index.
template<class SMR>
inline
std::size_t
computeMaximumEdgeIndex(const typename SMR::CellConstIterator c) {
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   std::size_t a = 0, b = 0, i = 0;
   c->computeMaximumEdgeLength(&a, &b);
   if (i == a) {
      ++i;
   }
   if (i == b) {
      ++i;
   }
   return i;
}



// Return true if this is a common maximum edge or a boundary edge.
template<class SMR>
inline
bool
isCommonMaximumEdge(const typename SMR::CellConstIterator c, const std::size_t i) {
   typedef typename SMR::Number Number;

   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   return c->isFaceOnBoundary(i) ||
      computeMaximumEdgeLength<SMR>(c->getNeighbor(i)->getSelf()) <=
          geom::computeDistance(c->getNode((i + 1) % 3)->getVertex(),
                                c->getNode((i + 2) % 3)->getVertex()) *
          (1.0 + 10.0 * std::numeric_limits<Number>::epsilon());
}


// Split the cell.
/*
  c is the cell.
  i defines the edge.
  m is the midnode of the edge to be split.
*/
template<std::size_t N, typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont>
inline
typename SimpMeshRed<N, 2, _T, _Node, _Cell, _Cont>::CellIterator
splitCell(SimpMeshRed<N, 2, _T, _Node, _Cell, _Cont>* mesh,
          const typename SimpMeshRed<N, 2, _T, _Node, _Cell, _Cont>::
          CellIterator c,
          const std::size_t i,
          typename SimpMeshRed<N, 2, _T, _Node, _Cell, _Cont>::Node* m) {
   typedef SimpMeshRed<N, 2, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Cell Cell;
   typedef typename SMR::CellIterator CellIterator;

   const std::size_t M = 2;

   // The other two node indices.
   const std::size_t i1 = (i + 1) % (M + 1);
   const std::size_t i2 = (i + 2) % (M + 1);
   // The neighbor that will be adjacent to the new cell.
   Cell* const n1 = c->getNeighbor(i1);

   // Add the new cell.  We know the three nodes and two of the neighbors.
   CellIterator cn = insertCell(mesh,
                                c->getNode(i), m, c->getNode(i2),
                                static_cast<Cell*>(0), n1, &*c);
   if (n1 != 0) {
      n1->setNeighbor(c->getMirrorIndex(i1), &*cn);
   }

   //
   // Fix the topology of the old cell.
   //

   // Remove old node-cell incidence.
   c->getNode(i2)->removeCell(c);
   // Changed the cell-node incidence.
   c->setNode(i2, m);
   // Add the new node-cell incidence.
   m->insertCell(c);
   // Cell adjacency with the new cell.
   c->setNeighbor(i1, &*cn);
   // At this point, we don't know the cell adjacency across the split edge.
   c->setNeighbor(i, 0);

   // Return the new cell.
   return cn;
}



// Register the node in the manifold.
template<typename SMR, std::size_t N, std::size_t M, std::size_t SD, typename T>
inline
void
registerNode(PointsOnManifold<N, M, SD, T>* manifold,
             typename SMR::NodeIterator node) {
   static_assert(N == SMR::N, "Bad dimensions.");

   typedef typename SMR::Vertex Vertex;

   assert(manifold != 0);

   // Register the point in the boundary manifold.
   const Vertex newLocation =
      manifold->insert(node->getIdentifier(), node->getVertex());
   // Update the location.
   node->setVertex(newLocation);
}



// Split the edge.
/*
  Return the new node.
*/
template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class _Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD >
inline
typename SimpMeshRed<N, 2, T, Node, _Cell, Cont>::NodeIterator
splitFace(SimpMeshRed<N, 2, T, Node, _Cell, Cont>* mesh,
          PointsOnManifold<N, MM, SD, T>* manifold,
          typename SimpMeshRed<N, 2, T, Node, _Cell, Cont>::CellIterator a,
          const std::size_t i) {
   // Either the boundary of the mesh or has the same dimensions as the mesh.
   static_assert(MM == 1 || MM == 2, "Dimensions not supported.");

   typedef SimpMeshRed<N, 2, T, Node, _Cell, Cont> SMR;
   typedef typename SMR::Vertex Vertex;
   typedef typename SMR::NodeIterator NodeIterator;
   typedef typename SMR::Cell Cell;

   const std::size_t M = 2;

   //
   // Perform a couple calculations before we muck up the incidence/adjacency
   // information.
   //

   // The neighboring cell.
   Cell* const b = a->getNeighbor(i);
   const std::size_t j = (b != 0) ? a->getMirrorIndex(i) : 0;

   // The mid-node.
   NodeIterator midNode = mesh->insertNode();
   {
      // The mid-point of the edge.
      Vertex midPoint = a->getNode((i + 1) % (M + 1))->getVertex();
      midPoint += a->getNode((i + 2) % (M + 1))->getVertex();
      midPoint *= 0.5;
      midNode->setVertex(midPoint);
   }

   // If a boundary manifold was specified.
   if (manifold != 0) {
      // If the edge is on the boundary.
      if (N == 2) {
         if (b == 0) {
            // Register the midpoint in the boundary manifold.
            registerNode<SMR>(manifold, midNode);
         }
      }
      else {
         // Register the midpoint in the manifold.
         registerNode<SMR>(manifold, midNode);
      }
   }

   //
   // Let the mucking commence.
   //

   // Split the first cell.
   Cell* an = &*splitCell(mesh, a, i, &*midNode);

   // Split the second cell.
   Cell* bn = 0;
   if (b != 0) {
      bn = &*splitCell(mesh, b->getSelf(), j, &*midNode);
   }

   // Fix the cell adjacencies.
   a->setNeighbor(i, bn);
   an->setNeighbor(an->getIndex(a->getNode(i)), b);
   if (b != 0) {
      b->setNeighbor(j, an);
      bn->setNeighbor(bn->getIndex(b->getNode(j)), &*a);
   }

   return midNode;
}




//! Split the cell.
/*!
  Split other cells as necessary to ensure that only shared longest edges are
  split.

  \return the number of edges split.
*/
template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD >
inline
std::size_t
splitCell(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
          PointsOnManifold<N, MM, SD, T>* manifold,
          typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::CellIterator c) {
   // Either the boundary of the mesh or has the same dimensions as the mesh.
   static_assert(MM == 1 || MM == 2, "Dimensions not supported.");

   typedef SimpMeshRed<N, 2, T, Node, Cell, Cont> SMR;

   // Count that we are splitting the longest edge of this cell.
   std::size_t count = 1;

   const std::size_t i = computeMaximumEdgeIndex<SMR>(c);
   // Recursively split the neighbor until the longest edge of c is a shared
   // longest edge.
   while (! isCommonMaximumEdge<SMR>(c, i)) {
      count += splitCell(mesh, manifold, c->getNeighbor(i)->getSelf());
   }
   // Split this cell.
   splitFace(mesh, manifold, c, i);
   // Return the number of edges split.
   return count;
}



//! Split the cell.
/*!
  Split other cells as necessary to ensure that only shared longest edges are
  split.

  \return the number of edges split.
*/
template < std::size_t N, typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         std::size_t MM, std::size_t SD,
         typename CellIterOutIter >
inline
std::size_t
splitCell(SimpMeshRed<N, 2, T, Node, Cell, Cont>* mesh,
          PointsOnManifold<N, MM, SD, T>* manifold,
          typename SimpMeshRed<N, 2, T, Node, Cell, Cont>::CellIterator c,
          CellIterOutIter splitCells) {
   typedef SimpMeshRed<N, 2, T, Node, Cell, Cont> SMR;

   // Count that we are splitting the longest edge of this cell.
   std::size_t count = 1;

   const std::size_t i = computeMaximumEdgeIndex<SMR>(c);
   // Recursively split the neighbor until the longest edge of c is a shared
   // longest edge.
   while (! isCommonMaximumEdge<SMR>(c, i)) {
      count += splitCell(mesh, manifold, c->getNeighbor(i)->getSelf(),
                         splitCells);
   }
   // Record the cells that will be split.
   *splitCells++ = c;
   if (! c->isFaceOnBoundary(i)) {
      *splitCells++ = c->getNeighbor(i)->getSelf();
   }
   // Split this cell.
   splitFace(mesh, manifold, c, i);

   return count;
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
inline
std::size_t
refineBoundary(SimpMeshRed<2, 2, T, Node, Cell, Cont>* mesh,
               const ISS& boundary,
               const T maxAngle, const T minimumEdgeLength,
               const std::size_t maxSweeps) {
   typedef typename SimpMeshRed<2, 2, T, Node, Cell, Cont>::Vertex VT;

   // The signed distance functor.
   ISS_SignedDistance<ISS> distance(boundary);
   // The functor for computing the closest point.
   ISS_SD_ClosestPoint<ISS> closestPoint(distance);

   return refineBoundary(mesh, distance, closestPoint,
                         ads::constructUnaryConstant<VT, T>(maxAngle),
                         ads::constructUnaryConstant<VT, T>(minimumEdgeLength),
                         maxSweeps);
}




//! Refine the mesh to better fit the boundary.
/*!
  \return the number of edges split.
*/
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         class Boundary,
         class MaxAngle,
         class MinEdgeLength >
inline
void
getBoundaryCellsToSplit(SimpMeshRed<2, 2, T, Node, Cell, Cont>& x,
                        const Boundary& boundary,
                        const MaxAngle& maxAngle,
                        const MinEdgeLength& minimumEdgeLength,
                        typename SimpMeshRed<2, 2, T, Node, Cell, Cont>::
                        CellIteratorSet* cells) {
   typedef SimpMeshRed<2, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::Vertex Vertex;

   const std::size_t M = 2;

   // The set uses the identifiers to compare cells.
   // CONTINUE: REMOVE
   //x.setCellIdentifiers();

   cells->clear();

   T length;
   T distance;
   Vertex midPoint;
   // For each cell.
   for (CellIterator c = x.getCellsBeginning(); c != x.getCellsEnd(); ++c) {
      // For each face.
      for (std::size_t m = 0; m != M + 1; ++m) {
         // If this is a boundary face.
         if (c->getNeighbor(m) == 0) {
            // Compute the mid-point of the edge.
            midPoint = c->getNode((m + 1) % (M + 1))->getVertex();
            midPoint += c->getNode((m + 2) % (M + 1))->getVertex();
            midPoint *= 0.5;
            // Check that the edge is not too short.
            length = geom::computeDistance(c->getNode((m + 1) % (M + 1))->getVertex(),
                                           c->getNode((m + 2) % (M + 1))->getVertex());
            if (length >= minimumEdgeLength(midPoint)) {
               // Compute the distance to the boundary and the closest point on
               // the boundary.
               distance = std::abs(boundary(midPoint));
               // If the edge deviates too much from the boundary.
               if (distance > std::tan(maxAngle(midPoint)) * 0.5 * length) {
                  /* CONTINUE REMOVE
                  std::cout << "midPoint = " << midPoint << "\n"
                        << "distance = " << distance << "\n"
                        << "length = " << length << "\n"
                        << "identifier = " << c->getIdentifier() << "\n"
                        << "m = " << m << "\n\n";
                  */
                  // Add the cell to the set.
                  cells->insert(cells->end(), c);
                  // Break out of the loop over the M+1 faces.
                  break;
               }
            }
         }
      }
   }
}




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
inline
std::size_t
refineBoundary(SimpMeshRed<2, 2, T, Node, Cell, Cont>* mesh,
               const Distance& distance,
               const ClosestPoint& closestPoint,
               const MaxAngle& maxAngle,
               const MinEdgeLength& minimumEdgeLength,
               const std::size_t maxSweeps) {
   typedef SimpMeshRed<2, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::CellIteratorSet CellIteratorSet;

   // The set of cells to be split during a sweep.
   CellIteratorSet cells;
   // This stores the cells that are split during a single recursive cell
   // splitting operation.
   std::vector<CellIterator> splitCells;

   std::size_t sweeps = 0;
   std::size_t count = 0;
   std::size_t c;
   do {
      c = 0;

      // Make a set of the cells on the boundary that should be split.
      getBoundaryCellsToSplit(*mesh, distance, maxAngle, minimumEdgeLength,
                              &cells);

      // Iterate until all the cells in the set have been split.
      while (! cells.empty()) {
         // Recursively split a cell.
         c += splitCell(mesh, *cells.begin(), std::back_inserter(splitCells));
         // Remove the cells that were split from the set.
         for (typename std::vector<CellIterator>::const_iterator i =
                  splitCells.begin(); i != splitCells.end(); ++i) {
            cells.erase(*i);
         }
         splitCells.clear();
      }

      // Move the boundary of the mesh to the specified boundary.
      transformBoundary(mesh, closestPoint);

      ++sweeps;
      count += c;
   }
   while (c != 0 && sweeps != maxSweeps);

   return count;
}





template<typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont,
         class _MaxEdgeLength>
inline
std::size_t
refineAdjust(SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont>* mesh,
             const _MaxEdgeLength& f) {
   typedef SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Vertex Vertex;
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::NodePointerSet NodePointerSet;

   std::size_t cnt, count = 0;
   Vertex x;
   std::size_t i;
   Node* node;
   NodePointerSet neighbors;

   do {
      cnt = 0;
      // Loop over the cells.
      for (CellIterator c = mesh->getCellsBeginning(); c != mesh->getCellsEnd();
            ++c) {
         c->getCentroid(&x);
         if (computeMaximumEdgeLength<SMR>(c, &i) > f(x)) {
            // Split the edge.
            node = splitFace(mesh, c, i);
            ++cnt;
            // Optimize the incidence relationships.
            // CONTINUE: This is very inefficient.
            incidenceOptimize(mesh);
            // Laplacian smoothing.
            determineNeighbors(*mesh, node, 2, &neighbors);
            applyLaplacian(mesh, neighbors.begin(), neighbors.end());
            applyLaplacian(mesh, neighbors.begin(), neighbors.end());
            applyLaplacian(mesh, neighbors.begin(), neighbors.end());
         }
      }
      count += cnt;
   }
   while (cnt != 0);
   return count;
}





template<typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont,
         class _BoundaryCondition,
         class _MinEdgeLength,
         class _MaxEdgeLength>
inline
std::pair<std::size_t, std::size_t>
optimizeWithSubspaceMethod(SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont>* mesh,
                           const _BoundaryCondition& boundaryCondition,
                           const _T minAngle,
                           const _MinEdgeLength& minEdgeLength,
                           const _MaxEdgeLength& maxEdgeLength,
                           const std::size_t norm = 0,
                           std::size_t numSweeps = 1) {
   typedef SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Vertex Vertex;
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::NodePointerSet NodePointerSet;

   const std::size_t M = 2;

   assert(minAngle >= 0);

   std::size_t collapseCount = 0, splitCount = 0;
   Vertex x;
   std::size_t i;
   Node* node;
   NodePointerSet neighbors, incidenceOptimizeNodes;
   CellIterator nextValidCell;
   bool isBoundaryModified = false;

   // Make the functor for determining which nodes can be moved.
   geom::IsNotSharpAngle<SMR> isMovable(minAngle);

   while (numSweeps-- != 0) {
      // Loop over the cells.
      CellIterator last = mesh->getCellsEnd();
      --last;
      std::size_t maxId = 0;
      if (last != mesh->getCellsEnd()) {
         maxId = last->getIdentifier();
      }
      for (CellIterator c = mesh->getCellsBeginning();
            c != mesh->getCellsEnd() && c->getIdentifier() <= maxId;) {
         isBoundaryModified = false;
         c->getCentroid(&x);
         // The last condition prevents collapsing an interior edge with two
         // boundary nodes.  Doing this could change the topology of the mesh.
         if (computeMinimumEdgeLength<SMR>(c, &i) < minEdgeLength(x) &&
               isCommonMinimumEdge<SMR>(c, i) &&
               ((c->isFaceOnBoundary(i) &&
                 isMovable(c->getNode((i + 1) % (M + 1))) &&
                 isMovable(c->getNode((i + 2) % (M + 1)))) ||
                doesEdgeHaveAnInteriorNode<SMR>(c, i))) {
            //std::cerr << c->getNode((i+1)%(M+1))->getVertex() << " "
            //<< c->getNode((i+2)%(M+1))->getVertex() << "\n";
            if (c->isFaceOnBoundary(i)) {
               isBoundaryModified = true;
            }
            nextValidCell = getNextValidCellAfterCollapse<SMR>(c, i);
            // The edge will be collapsed to a node.
            node = c->getNode((i + 1) % (M + 1));
            collapse(mesh, c, i);
            // Move to the next valid cell.
            c = nextValidCell;
            ++collapseCount;

            incidenceOptimizeNodes.clear();
            if (isBoundaryModified) {
               // CONTINUE
               //std::cerr << "Laplacian on boundary after collapse.\n";
               applyLaplacian(mesh, boundaryCondition, minAngle, 4);
               //std::cerr << "Done.\n";
               determineBoundaryNeighbors<SMR>(node, &incidenceOptimizeNodes);
            }
            incidenceOptimizeNodes.insert(node);

            // CONTINUE: These are very inefficient.
            // Optimize the incidence relationships.
            //incidenceOptimize(mesh);
            incidenceOptimize(mesh, &incidenceOptimizeNodes, norm);
            // Laplacian smoothing on interior nodes.
            determineNeighbors(*mesh, node, 3, &neighbors);
            applyLaplacian(mesh, neighbors.begin(), neighbors.end(), 4);
         }
         else if (computeMaximumEdgeLength<SMR>(c, &i) > maxEdgeLength(x) &&
                  isCommonMaximumEdge<SMR>(c, i)) {
            if (c->isFaceOnBoundary(i)) {
               isBoundaryModified = true;
            }
            // Split the edge.
            PointsOnManifold<2, 1, 1, _T>* dummy = 0;
            node = &*splitFace(mesh, dummy, c, i);
            ++c;
            ++splitCount;
            // CONTINUE: These are very inefficient.

            incidenceOptimizeNodes.clear();
            if (isBoundaryModified) {
               applyLaplacian(mesh, boundaryCondition, minAngle, 4);
               determineBoundaryNeighbors<SMR>(node, &incidenceOptimizeNodes);
            }
            incidenceOptimizeNodes.insert(node);

            // Optimize the incidence relationships.
            //incidenceOptimize(mesh);
            incidenceOptimize(mesh, &incidenceOptimizeNodes, norm);
            // Laplacian smoothing.
            determineNeighbors(*mesh, node, 3, &neighbors);
            applyLaplacian(mesh, neighbors.begin(), neighbors.end(), 4);
         }
         else {
            ++c;
         }
      }
   }

   return std::make_pair(collapseCount, splitCount);
}





template<typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont,
         class _LevelSet,
         class _MinEdgeLength,
         class _MaxEdgeLength>
inline
std::pair<std::size_t, std::size_t>
optimizeWithSubspaceMethod(SimpMeshRed<3, 2, _T, _Node, _Cell, _Cont>* mesh,
                           const _LevelSet& levelSet,
                           const _MinEdgeLength& minEdgeLength,
                           const _MaxEdgeLength& maxEdgeLength,
                           const std::size_t norm = 0,
                           std::size_t numSweeps = 1) {
   typedef SimpMeshRed<3, 2, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Vertex Vertex;
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::NodePointerSet NodePointerSet;

   const std::size_t M = 2;

   std::size_t collapseCount = 0, splitCount = 0;
   Vertex x;
   std::size_t i;
   Node* node;
   NodePointerSet neighbors, incidenceOptimizeNodes;
   CellIterator nextValidCell;

   while (numSweeps-- != 0) {
      // Loop over the cells.
      CellIterator last = mesh->getCellsEnd();
      --last;
      std::size_t maxId = 0;
      if (last != mesh->getCellsEnd()) {
         maxId = last->getIdentifier();
      }
      for (CellIterator c = mesh->getCellsBeginning();
            c != mesh->getCellsEnd() && c->getIdentifier() <= maxId;) {
         c->getCentroid(&x);
         // The last condition prevents collapsing an interior edge with two
         // boundary nodes.  Doing this could change the topology of the mesh.
         if (computeMinimumEdgeLength<SMR>(c, &i) < minEdgeLength(x) &&
               isCommonMinimumEdge<SMR>(c, i) &&
               doesEdgeHaveAnInteriorNode<SMR>(c, i)) {
            nextValidCell = getNextValidCellAfterCollapse<SMR>(c, i);
            // The edge will be collapsed to a node.
            node = c->getNode((i + 1) % (M + 1));
            collapse(mesh, c, i);
            // Move to the next valid cell.
            c = nextValidCell;
#if 0
            // REMOVE
            {
               std::cerr << "collapse " << node->getVertex() << "\n";
               std::ofstream file("problem.txt");
               writeAscii(file, *mesh);
            }
#endif
            ++collapseCount;

            incidenceOptimizeNodes.clear();
            incidenceOptimizeNodes.insert(node);

            // Laplacian smoothing on the node.
            determineNeighbors(*mesh, node, 1, &neighbors);
            applyLaplacian(mesh, levelSet, neighbors.begin(), neighbors.end(), 1);
            // Optimize the incidence relationships.
            incidenceOptimize(mesh, &incidenceOptimizeNodes, norm);
            // Laplacian smoothing on interior nodes.
            determineNeighbors(*mesh, node, 3, &neighbors);
            applyLaplacian(mesh, levelSet, neighbors.begin(), neighbors.end(), 4);
         }
         else if (computeMaximumEdgeLength<SMR>(c, &i) > maxEdgeLength(x) &&
                  isCommonMaximumEdge<SMR>(c, i)) {
            // Split the edge.
            PointsOnManifold<3, 1, 1, _T>* dummy = 0;
            node = &*splitFace(mesh, dummy, c, i);
#if 0
            // REMOVE
            {
               std::cerr << "split " << node->getVertex() << "\n";
               std::ofstream file("problem.txt");
               writeAscii(file, *mesh);
            }
#endif
            ++c;
            ++splitCount;

            incidenceOptimizeNodes.clear();
            incidenceOptimizeNodes.insert(node);

            // Laplacian smoothing on the node.
            determineNeighbors(*mesh, node, 1, &neighbors);
            applyLaplacian(mesh, levelSet, neighbors.begin(), neighbors.end(), 1);
            // Optimize the incidence relationships.
            incidenceOptimize(mesh, &incidenceOptimizeNodes, norm);
            // Laplacian smoothing.
            determineNeighbors(*mesh, node, 3, &neighbors);
            applyLaplacian(mesh, levelSet, neighbors.begin(), neighbors.end(), 4);
         }
         else {
            ++c;
         }
      }
   }

   return std::make_pair(collapseCount, splitCount);
}





template<typename _T,
         template<class> class _Node,
         template<class> class _Cell,
         template<class, class> class _Cont>
inline
std::size_t
refineLengthRatio(SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont>* mesh,
                  const _T maxRatio) {
   typedef SimpMeshRed<2, 2, _T, _Node, _Cell, _Cont> SMR;
   typedef typename SMR::Node Node;
   typedef typename SMR::CellIterator CellIterator;
   typedef typename SMR::NodePointerSet NodePointerSet;

   std::size_t count = 0;
   std::size_t i;
   Node* node;
   NodePointerSet neighbors;

   // Loop over the cells.
   CellIterator last = mesh->getCellsEnd();
   --last;
   std::size_t maxId = 0;
   if (last != mesh->getCellsEnd()) {
      maxId = last->getIdentifier();
   }
   for (CellIterator c = mesh->getCellsBeginning();
         c != mesh->getCellsEnd() && c->getIdentifier() <= maxId; ++c) {
      if (computeMaximumEdgeLength<SMR>(c, &i) >
            maxRatio * computeMinimumEdgeLength<SMR>(c)) {
         // Split the edge.
         PointsOnManifold<2, 1, 1, _T>* dummy = 0;
         node = splitFace(mesh, dummy, c, i);
         ++count;
         // Optimize the incidence relationships.
         // CONTINUE: This is very inefficient.
         incidenceOptimize(mesh);
         // Laplacian smoothing.
         determineNeighbors(*mesh, node, 3, neighbors);
         applyLaplacian(mesh, neighbors.begin(), neighbors.end());
         applyLaplacian(mesh, neighbors.begin(), neighbors.end());
         applyLaplacian(mesh, neighbors.begin(), neighbors.end());
         applyLaplacian(mesh, neighbors.begin(), neighbors.end());
      }
   }

   return count;
}

} // namespace geom
}
