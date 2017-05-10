// -*- C++ -*-

#if !defined(__geom_mesh_simplicial_geometry_ipp__)
#error This file is an implementation detail.
#endif

namespace stlib
{
namespace geom {


template<class SMR>
inline
typename SMR::Number
computeIncidentCellsAngle
(const typename SMR::Node* node,
 std::integral_constant<std::size_t, 2> /*space dimension*/,
 std::integral_constant<std::size_t, 2> /*simplex dimension*/) {
   typedef typename SMR::Node::CellIncidentToNodeConstIterator
   CellIncidentToNodeConstIterator;
   typedef typename SMR::Vertex Vertex;

   static_assert(SMR::N == 2, "The space dimension must be 2.");
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");
   const std::size_t M = 2;

#ifdef STLIB_DEBUG
   assert(node->isOnBoundary());
#endif

   //
   // Get the two two tangent vectors to the neighboring faces, a and b.
   //

#ifdef STLIB_DEBUG
   std::size_t aCount = 0, bCount = 0;
#endif
   std::size_t i;
   // Initialize to avoid a warning.
   Vertex a = ext::filled_array<Vertex>(0);
   Vertex b = ext::filled_array<Vertex>(0);
   // For each incident cell.
   for (CellIncidentToNodeConstIterator c = node->getCellsBeginning();
         c != node->getCellsEnd(); ++c) {
      i = c->getIndex(node);

      if (c->isFaceOnBoundary((i + 2) % (M + 1))) {
         a = c->getNode((i + 1) % (M + 1))->getVertex();
         a -= node->getVertex();
         ext::normalize(&a);
#ifdef STLIB_DEBUG
         ++aCount;
#endif
      }

      if (c->isFaceOnBoundary((i + 1) % (M + 1))) {
         b = c->getNode((i + 2) % (M + 1))->getVertex();
         b -= node->getVertex();
         ext::normalize(&b);
#ifdef STLIB_DEBUG
         ++bCount;
#endif
      }
   }

#ifdef STLIB_DEBUG
   // Make sure we found the two boundary faces.
   assert(aCount == 1 && bCount == 1);
#endif

   //
   // Get the inward normal direction, n.
   //

   Vertex d, n;

   d = a;
   rotatePiOver2(&d);
   n = d;

   d = b;
   rotateMinusPiOver2(&d);
   n += d;

   // No need to normalize n.
   return computeAngle(a, n) + computeAngle(n, b);
}





template<class SMR>
inline
typename SMR::Number
computeIncidentCellsAngle
(const typename SMR::Node* node,
 std::integral_constant<std::size_t, 3> /*space dimension*/,
 std::integral_constant<std::size_t, 2> /*simplex dimension*/) {
   typedef typename SMR::Node::CellIncidentToNodeConstIterator
   CellIncidentToNodeConstIterator;
   typedef typename SMR::Number Number;
   typedef typename SMR::Vertex Vertex;

   static_assert(SMR::N == 3, "The space dimension must be 3.");
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");
   const std::size_t M = 2;

   // For each each incident cell.
   std::size_t i;
   Vertex a, b;
   Number incidentAngle = 0;
   for (CellIncidentToNodeConstIterator cell = node->getCellsBeginning();
         cell != node->getCellsEnd(); ++cell) {
      i = cell->getIndex(node);
      a = cell->getNode((i + 1) % (M + 1))->getVertex();
      a -= node->getVertex();
      b = cell->getNode((i + 2) % (M + 1))->getVertex();
      b -= node->getVertex();
      incidentAngle += computeAngle(a, b);
   }

   return incidentAngle;
}



template<class SMR>
inline
typename SMR::Number
computeIncidentCellsAngle
(const typename SMR::Node* node,
 std::integral_constant<std::size_t, 3> /*space dimension*/,
 std::integral_constant<std::size_t, 3> /*simplex dimension*/) {
   typedef typename SMR::Node::CellIncidentToNodeConstIterator
   CellIncidentToNodeConstIterator;
   typedef typename SMR::Number Number;
   typedef typename SMR::Vertex Vertex;
   typedef std::array < Vertex, 3 + 1 > Tetrahedron;

   static_assert(SMR::N == 3, "The space dimension must be 3.");
   static_assert(SMR::M == 3, "The simplex dimension must be 3.");
   const std::size_t M = 3;

   // For each each incident cell.
   Tetrahedron t;
   std::size_t i, m;
   Vertex a, b;
   Number incidentAngle = 0;
   for (CellIncidentToNodeConstIterator cell = node->getCellsBeginning();
         cell != node->getCellsEnd(); ++cell) {
      // The local index of the node.
      i = cell->getIndex(node);
      // Copy the cell into a simplex.
      for (m = 0; m != (M + 1); ++m) {
         t[m] = cell->getNode(m)->getVertex();
      }
      // Add the solid angle at the i_th vertex of the simplex.
      incidentAngle += computeAngle(t, i);
   }

   return incidentAngle;
}



template<class SMR>
inline
typename SMR::Number
computeIncidentCellsAngle(const typename SMR::Node* node) {
   return computeIncidentCellsAngle<SMR>
     (node, std::integral_constant<std::size_t, SMR::N>(),
      std::integral_constant<std::size_t, SMR::M>());
}




// Compute the dihedral angle at the specified edge.
// The dihedral angle is accumulated from the incident cells.
template<class SMR>
inline
typename SMR::Number
computeDihedralAngle(typename SMR::ConstEdge edge) {
   typedef typename SMR::Number Number;
   typedef typename SMR::Node Node;
   typedef typename Node::CellIteratorConstIterator CellIteratorConstIterator;
   typedef typename SMR::Simplex Simplex;

   // The simplex dimension must be 3.
   static_assert(SMR::M == 3, "The simplex dimension must be 3.");

   // The source node of the edge.
   const Node* const a = edge.first->getNode(edge.second);
   // The target node of the edge.
   const Node* const b = edge.first->getNode(edge.third);

   Number dihedralAngle = 0;
   Simplex simplex;
   std::size_t faceIndex1, faceIndex2;
   // For each cell incident to the source node.
   for (CellIteratorConstIterator c = a->getCellIteratorsBeginning();
         c != a->getCellIteratorsEnd(); ++c) {
      // If the cell is incident to the target node as well, it is incident
      // to the edge.
      if ((*c)->hasNode(b)) {
         // Make a simplex from the cell.
         (*c)->getSimplex(&simplex);
         // Get the indices of the two faces that are incident to the edge.
         computeOtherIndices((*c)->getIndex(a), (*c)->getIndex(b),
                             &faceIndex1, &faceIndex2);
         // The the contribution from this cell.
         dihedralAngle += computeAngle(simplex, faceIndex1, faceIndex2);
      }
   }
   return dihedralAngle;
}



// Return the cosine of the interior angle at the specified 1-face.
// The 1-face must have two incident simplices.
template<class SMR>
inline
typename SMR::Number
computeCosineAngle(typename SMR::FaceConstIterator face) {
   typedef typename SMR::Vertex Vertex;

   static_assert(SMR::N == 3 && SMR::M == 2, "Must be a 3-2 mesh.");

   // Check that the face is valid.
   // CONTINUE: I could test this if faces used pointers instead of iterators.
   //assert(face->first != typename SMR::CellConstIterator(0));
   assert(face->second < SMR::M + 1);
   // It must be an internal face.
   assert(! isOnBoundary<SMR>(face));

   // The cosine of the angle is the negative of the dot product of the
   // incident simplex normals.
   // n0 . n1 == cos(pi - a) == - cos(a)
   const Vertex n0 = computeCellNormal<SMR>(face->first);
   const Vertex n1 = computeCellNormal<SMR>
      (face->first->getNeighbor(face->second)->getSelf());
   return - ext::dot(n0, n1);
}




// Compute the normal to the surface at the node.
template<class SMR>
inline
void
computeNodeNormal(const typename SMR::Node* node,
                  typename SMR::Vertex* normal,
                  std::integral_constant<std::size_t, 2> /*space_dimension*/,
                  std::integral_constant<std::size_t, 2> /*simplex_dimension*/)
{
   typedef typename SMR::Node::CellIncidentToNodeConstIterator
   CellIncidentToNodeConstIterator;
   typedef typename SMR::Vertex Vertex;

   static_assert(SMR::N == 2, "The space dimension must be 2.");
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   const std::size_t M = 2;

   std::size_t i;
   Vertex v;

   std::fill(normal->begin(), normal->end(), 0);
   // For each incident cell.
   for (CellIncidentToNodeConstIterator c = node->getCellsBeginning();
         c != node->getCellsEnd(); ++c) {
      i = c->getIndex(node);

      if (c->isFaceOnBoundary((i + 2) % (M + 1))) {
         v = c->getNode((i + 1) % (M + 1))->getVertex();
         v -= node->getVertex();
         rotateMinusPiOver2(&v);
         ext::normalize(&v);
         *normal += v;
      }

      if (c->isFaceOnBoundary((i + 1) % (M + 1))) {
         v = c->getNode((i + 2) % (M + 1))->getVertex();
         v -= node->getVertex();
         rotatePiOver2(&v);
         ext::normalize(&v);
         *normal += v;
      }
   }
   ext::normalize(normal);
}



// Compute the normal to the surface at the node.
template<class SMR>
inline
void
computeNodeNormal(const typename SMR::Node* node,
                  typename SMR::Vertex* vertexNormal,
                  std::integral_constant<std::size_t, 3> /*space_dimension*/,
                  std::integral_constant<std::size_t, 2> /*simplex_dimension*/)
{
   typedef typename SMR::Node::CellIncidentToNodeConstIterator
   CellIncidentToNodeConstIterator;
   typedef typename SMR::Vertex Vertex;

   static_assert(SMR::N == 3, "The space dimension must be 3.");
   static_assert(SMR::M == 2, "The simplex dimension must be 2.");

   const std::size_t M = 2;

   // The vertex should have at least 3 incident faces.
   assert(node->getCellsSize() >= 3);

   std::fill(vertexNormal->begin(), vertexNormal->end(), 0.0);
   std::size_t i;
   Vertex x, y, faceNormal;

   // For each incident face.
   for (CellIncidentToNodeConstIterator c = node->getCellsBeginning();
         c != node->getCellsEnd(); ++c) {
      // The local index of the node in the face.
      i = c->getIndex(node);

      // Compute the face normal.
      x = c->getNode((i + 1) % (M + 1))->getVertex();
      x -= node->getVertex();
      ext::normalize(&x);
      y = c->getNode((i + 2) % (M + 1))->getVertex();
      y -= node->getVertex();
      ext::normalize(&y);
      ext::cross(x, y, &faceNormal);
      ext::normalize(&faceNormal);

      // Contribute to the vertex normal.
      // Multiply by the angle between the edges.
      faceNormal *= std::acos(ext::dot(x, y));
      *vertexNormal += faceNormal;
   }
   ext::normalize(vertexNormal);
}



template<class SMR>
inline
void
computeNodeNormal(const typename SMR::Node* node,
                  typename SMR::Vertex* normal) {
   computeNodeNormal<SMR>(node, normal,
                          std::integral_constant<std::size_t, SMR::N>(),
                          std::integral_constant<std::size_t, SMR::M>());
}


// Implementation for 2-1 meshes.
template<class SMR>
inline
void
computeCellNormal(typename SMR::CellConstIterator cell,
                  typename SMR::Vertex* normal,
                  std::integral_constant<std::size_t, 1> /*dummy*/) {
   static_assert(SMR::M == 1, "Incorrectly called.");
   static_assert(SMR::M + 1 == SMR::N, "Bad dimensions for a normal.");

   *normal = cell->getNode(1)->getVertex();
   *normal -= cell->getNode(0)->getVertex();
   rotateMinusPiOver2(normal);
   normalize(normal);
}



// Implementation for 3-2 meshes.
template<class SMR>
inline
void
computeCellNormal(typename SMR::CellConstIterator cell,
                  typename SMR::Vertex* normal,
                  std::integral_constant<std::size_t, 2> /*dummy*/) {
   static_assert(SMR::M == 2, "Incorrectly called.");
   static_assert(SMR::M + 1 == SMR::N, "Bad dimensions for a normal.");

   typedef typename SMR::Vertex Vertex;

   Vertex a = cell->getNode(1)->getVertex();
   a -= cell->getNode(0)->getVertex();
   Vertex b = cell->getNode(2)->getVertex();
   b -= cell->getNode(0)->getVertex();
   ext::cross(a, b, normal);
   ext::normalize(normal);
}


// Compute the cell normal.
template<class SMR>
void
computeCellNormal(typename SMR::CellConstIterator cell,
                  typename SMR::Vertex* normal) {
   computeCellNormal<SMR>(cell, normal,
                          std::integral_constant<std::size_t, SMR::M>());
}







// Implementation for 2-2 meshes.
template<class SMR>
inline
void
computeFaceNormal(typename SMR::CellConstIterator cell, const std::size_t i,
                  typename SMR::Vertex* normal,
                  std::integral_constant<std::size_t, 2>
                  /*The space and simplex dimension*/) {
   static_assert(SMR::N == 2 && SMR::M == 2, "Incorrectly called.");

   const std::size_t M = 2;

   *normal = cell->getNode((i + 2) % (M + 1))->getVertex();
   *normal -= cell->getNode((i + 1) % (M + 1))->getVertex();
   rotateMinusPiOver2(normal);
   normalize(normal);

   // For the simplex (v[0], ... v[N]) the face is
   // (-1)^n (v[0], ..., v[n-1], v[n+1], ..., v[N]).
   if (i % 2 == 1) {
      negateElements(normal);
   }
}


// Implementation for 3-3 meshes.
template<class SMR>
inline
void
computeFaceNormal(typename SMR::CellConstIterator cell, const std::size_t i,
                  typename SMR::Vertex* normal,
                  std::integral_constant<std::size_t, 3>
                  /*The space and simplex dimension*/) {
   static_assert(SMR::N == 3 && SMR::M == 3, "Incorrectly called.");

   typedef typename SMR::Vertex Vertex;

   const std::size_t M = 3;

   Vertex a = cell->getNode((i + 2) % (M + 1))->getVertex();
   a -= cell->getNode((i + 1) % (M + 1))->getVertex();
   Vertex b = cell->getNode((i + 3) % (M + 1))->getVertex();
   b -= cell->getNode((i + 1) % (M + 1))->getVertex();
   computeCrossProduct(a, b, normal);
   normalize(normal);

   // For the simplex (v[0], ... v[N]) the face is
   // (-1)^n (v[0], ..., v[n-1], v[n+1], ..., v[N]).
   if (i % 2 == 1) {
      negateElements(normal);
   }
}


// Compute the face normal.
template<class SMR>
void
computeFaceNormal(const typename SMR::CellConstIterator cell, const std::size_t i,
                  typename SMR::Vertex* normal) {
   static_assert(SMR::N == SMR::M,
                 "The space and simplex dimension must be the same.");
   computeFaceNormal<SMR>(cell, i, normal,
                          std::integral_constant<std::size_t, SMR::N>());
}





// Project the line segments to 1-D and collect them.
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename OutputIterator >
inline
void
projectAndGetSimplices(const SimpMeshRed<2, 1, T, Node, Cell, Cont>& mesh,
                       OutputIterator simplices) {
   typedef SimpMeshRed<2, 1, T, Node, Cell, Cont> SMR;
   typedef typename SMR::SimplexIterator SimplexIterator;
   typedef std::array < std::array<T, 1>, 1 + 1 > Segment;

   Segment t;

   // For each simplex.
   for (SimplexIterator s = mesh.getSimplicesBeginning();
         s != mesh.getSimplicesEnd(); ++s) {
      // Project the line segment in 2-D to a line segment in 1-D.
      projectToLowerDimension(*s, &t);
      // Add the line segment to the sequence of simplices.
      *simplices++ = t;
   }
}


// Project the triangle simplices to 2-D and collect them.
template < typename T,
         template<class> class Node,
         template<class> class Cell,
         template<class, class> class Cont,
         typename OutputIterator >
inline
void
projectAndGetSimplices(const SimpMeshRed<3, 2, T, Node, Cell, Cont>& mesh,
                       OutputIterator simplices) {
   typedef SimpMeshRed<3, 2, T, Node, Cell, Cont> SMR;
   typedef typename SMR::SimplexIterator SimplexIterator;
   typedef std::array < std::array<T, 2>, 2 + 1 > Triangle;

   Triangle t;

   // For each simplex.
   for (SimplexIterator s = mesh.getSimplicesBeginning();
         s != mesh.getSimplicesEnd(); ++s) {
      // Project the triangle in 3-D to a triangle in 2-D.
      projectToLowerDimension(*s, &t);
      // Add the triangle to the sequence of simplices.
      *simplices++ = t;
   }
}

} // namespace geom
}
