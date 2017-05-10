// -*- C++ -*-

#if !defined(__geom_ISS_SignedDistance_3_ipp__)
#error This file is an implementation detail of the class ISS_SignedDistance.
#endif

namespace stlib
{
namespace geom {

//! Signed distance to a triangle faceted surface in 3-D.
/*!
  \param ISS is the indexed simplex set.

  This class stores a constant reference to an indexed simplex set.
*/
template<class ISS>
class ISS_SignedDistance<ISS, 3> {
   //
   // Private types.
   //

private:

   //! The indexed simplex set.
   typedef ISS IssType;

   //! The (un-indexed) simplex type.
   typedef typename IssType::Simplex Simplex;
   //! The (un-indexed) edge type.
   typedef typename IssType::SimplexFace Edge;

   //
   // Public types.
   //

public:

   //! The number type.
   typedef typename IssType::Number Number;
   //! A vertex.
   typedef typename IssType::Vertex Vertex;
   //! A bounding box.
   typedef geom::BBox<Number, ISS::N> BBox;

   //
   // Member data.
   //

private:

   //! The indexed simplex set.
   const IssType& _iss;
   //! The vertex normals.
   std::vector<Vertex> _vertexNormals;
   //! The edge normals.
   std::vector<Vertex> _edgeNormals;
   //! The face-face adjacencies.
   std::vector<std::array<std::size_t, 3> > _adjacentFaces;
   //! The face-edge incidence.
   std::vector<std::array<std::size_t, 3> > _incidentEdges;
   //! The face normals.
   std::vector<Vertex> _faceNormals;
   //! Bounding box tree.
   BBoxTree<ISS::N, Number> _bboxTree;


   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   ISS_SignedDistance();

   //! Copy constructor not implemented
   ISS_SignedDistance(const ISS_SignedDistance&);

   //! Assignment operator not implemented
   ISS_SignedDistance&
   operator=(const ISS_SignedDistance&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //! @{

   //! Construct from the indexed simplex set.
   /*!
     \param iss is the indexed simplex set.
   */
   ISS_SignedDistance(const IssType& iss) :
      _iss(iss),
      _vertexNormals(_iss.vertices.size()),
      // Initially, we don't know the number of interior edges.  Thus we
      // delay allocating memory.
      _edgeNormals(),
      _adjacentFaces(),
      _incidentEdges(_iss.indexedSimplices.size()),
      _faceNormals(_iss.indexedSimplices.size()),
      _bboxTree() {
      simplexAdjacencies(&_adjacentFaces, _iss.vertices.size(),
                         _iss.indexedSimplices);
      build();
   }

   //! Destructor has no effect on the indexed simplex set.
   ~ISS_SignedDistance() {}

   //! @}
   //--------------------------------------------------------------------------
   //! \name Signed distance.
   //! @{

   //! Return the signed distance to the manifold.
   Number
   operator()(const Vertex& x) const {
      // CONTINUE: Should this be static?
      static Vertex closestPoint;
      return operator()(x, &closestPoint);
   }

   //! Return the signed distance to the manifold.  Compute the closest point.
   Number
   operator()(const Vertex& x, Vertex* closestPoint) const;

   //! Return the signed distance to the manifold.  Compute the closest point, gradient of the distance and the index of the closest triangle simplex.
   Number
   operator()(const Vertex& x, Vertex* closestPoint, Vertex* gradient,
              std::size_t* index) const;

   //! Return the closest point on the manifold.
   const Vertex&
   computeClosestPoint(const Vertex& x) const {
      static Vertex closestPoint;
      operator()(x, &closestPoint);
      return closestPoint;
   }

   //! @}

   //
   // Private member functions.
   //

private:

   //! Build the bounding box tree.
   void
   build();
};




//
// Signed distance.
//

// Return the signed distance to the manifold.  Compute the closest point.
template<class ISS>
inline
typename ISS_SignedDistance<ISS, 3>::Number
ISS_SignedDistance<ISS, 3>::
operator()(const Vertex& x, Vertex* closestPoint) const {
   Vertex dummy_grad;
   std::size_t dummy_index;
   return operator()(x, closestPoint, &dummy_grad, &dummy_index);
}



// Return the signed distance to the manifold.  Compute the closest point, gradient of the distance and the index of the closest triangle simplex.
template<class ISS>
inline
typename ISS_SignedDistance<ISS, 3>::Number
ISS_SignedDistance<ISS, 3>::
operator()(const Vertex& x, Vertex* closestPoint, Vertex* gradient,
           std::size_t* index) const {
   // CONTINUE: Make these into member data.
   static Simplex f;
   static Edge e;
   static Vertex v;
   static std::vector<std::size_t> candidateFaces;
   static std::set<std::size_t> faceSet;
   static std::map<std::size_t, std::size_t> candidateVertices;
   std::size_t n;
   Number d, mag;

   assert(_iss.indexedSimplices.size() != 0);

   //
   // Get the candidate simplices.
   //
   candidateFaces.clear();
   _bboxTree.computeMinimumDistanceQuery(std::back_inserter(candidateFaces), x);

   //
   // Calculate the distance to the candidate triangle faces.
   //
   Number minDistance = std::numeric_limits<Number>::max();
   const std::size_t i_end = candidateFaces.size();
   for (std::size_t i = 0; i != i_end; ++i) {
      n = candidateFaces[i];
      // Get the simplex.
      _iss.getSimplex(n, &f);
      // Calculate the signed distance to the simplex.
      d = computeSignedDistance(f, _faceNormals[n], x, &v);
      // If this is the current minimum distance.
      if (std::abs(d) < std::abs(minDistance)) {
         // Record the distance and closest point.
         minDistance = d;
         *closestPoint = v;
         *gradient = _faceNormals[n];
         // Closest simplex index.
         *index = n;
      }
   }

   //
   // Calculate the distance to the candidate edges.
   //
   // Make a set of the candidate faces.
   faceSet.clear();
   faceSet.insert(candidateFaces.begin(), candidateFaces.end());
   std::size_t m;
   // For each candidate face.
   for (std::size_t i = 0; i != i_end; ++i) {
      // The face index.
      n = candidateFaces[i];
      // For each edge of the face.
      for (m = 0; m != 3; ++m) {
         // If there is an adjacent face and the index of the adjacent face
         // is less than our index and the adjacent face is in the set of
         // candidate faces.
         if (_adjacentFaces[n][m] != std::size_t(-1) &&
             n < _adjacentFaces[n][m] && faceSet.count(n) == 1) {
            // Make the edge.
            e[0] = _iss.vertices[_iss.indexedSimplices[n][(m+1)%3]];
            e[1] = _iss.vertices[_iss.indexedSimplices[n][(m+2)%3]];
            // Calculate the signed distance to the edge.
            d = computeSignedDistance(e, _edgeNormals[_incidentEdges[n][m]], x,
                                      &v);
            // If this is the current minimum distance.
            if (std::abs(d) < std::abs(minDistance)) {
               // Record the distance and closest point.
               minDistance = d;
               *closestPoint = v;
               // Gradient of the distance.
               *gradient = x;
               *gradient -= *closestPoint;
               mag = ext::magnitude(*gradient);
               if (mag != 0) {
                  *gradient /= mag;
               }
               else {
                  *gradient = _edgeNormals[_incidentEdges[n][m]];
               }
               if (d < 0) {
                  ext::negateElements(gradient);
               }
               // Closest simplex index.
               *index = n;
            }
         }
      }
   }

   //
   // Build the candidate vertices.
   //
   candidateVertices.clear();
   for (std::size_t i = 0; i != i_end; ++i) {
      n = candidateFaces[i];
      // For each of the three edges/vertices.
      for (m = 0; m != 3; ++m) {
         // Insert the vertex, incident face pair.
         candidateVertices.insert(std::pair<const std::size_t, std::size_t>
                                  (_iss.indexedSimplices[n][m], n));
      }
   }

   //
   // Calculate the distance to the candidate vertices.
   //
   std::map<std::size_t, std::size_t>::iterator vEnd = candidateVertices.end();
   std::map<std::size_t, std::size_t>::iterator vIter = candidateVertices.begin();
   for (; vIter != vEnd; ++vIter) {
      n = vIter->first;
      // Calculate the signed distance to the vertex.
      d = computeSignedDistance(_iss.vertices[n], _vertexNormals[n], x);
      // If this is the current minimum distance.
      if (std::abs(d) < std::abs(minDistance)) {
         // Record the distance and closest point.
         minDistance = d;
         *closestPoint = _iss.vertices[n];
         // Gradient of the distance.
         *gradient = x;
         *gradient -= *closestPoint;
         mag = ext::magnitude(*gradient);
         if (mag != 0) {
            *gradient /= mag;
         }
         else {
            *gradient = _vertexNormals[n];
         }
         if (d < 0) {
            ext::negateElements(gradient);
         }
         // Closest simplex index.
         *index = vIter->second;
      }
   }

   // CONTINUE: Fix and REMOVE
   if (minDistance == std::numeric_limits<Number>::max()) {
      std::cerr << "candidateFaces.size() = "
                << candidateFaces.size() << "\n"
                << "candidateVertices.size() = "
                << candidateVertices.size() << "\n"
                << "x = " << x << "\n"
                << "*closestPoint = " << *closestPoint << "\n";
   }

   assert(minDistance != std::numeric_limits<Number>::max());
   return minDistance;
}



//
// Private member functions.
//



template<class ISS>
inline
void
ISS_SignedDistance<ISS, 3>::
build() {
   //
   // Build the bounding box tree.
   //

   std::vector<BBox> boxes(_iss.indexedSimplices.size());
   Simplex simplex;
   // For each simplex.
   for (std::size_t n = 0; n != _iss.indexedSimplices.size(); ++n) {
      _iss.getSimplex(n, &simplex);
      // Make a bounding box around the simplex.
      boxes[n] = specificBBox<BBox>(simplex.begin(), simplex.end());
   }
   // Build the tree from the bounding boxes.
   _bboxTree.build(boxes.begin(), boxes.end());

   //
   // Build the face-edge incidences.
   //

   // Build the face-edge incidences.
   std::size_t m, adj;
   std::size_t edgeIndex = 0;
   for (std::size_t n = 0; n != _iss.indexedSimplices.size(); ++n) {
      for (m = 0; m != 3; ++m) {
         // The adjacent face index.
         adj = _adjacentFaces[n][m];
         // If there is no adjacent face.
         if (adj == std::size_t(-1)) {
            // There is no incident interior edge.
            _incidentEdges[n][m] = std::numeric_limits<std::size_t>::max();
         }
         // There is an adjacent face.
         else {
            // If our index is less than the adjacent index.
            if (n < adj) {
               _incidentEdges[n][m] = edgeIndex++;
            }
            else {
               if (_adjacentFaces[adj][0] == n) {
                  _incidentEdges[n][m] = _incidentEdges[adj][0];
               }
               else if (_adjacentFaces[adj][1] == n) {
                  _incidentEdges[n][m] = _incidentEdges[adj][1];
               }
               else if (_adjacentFaces[adj][2] == n) {
                  _incidentEdges[n][m] = _incidentEdges[adj][2];
               }
               else {
                  assert(false);
               }
            }
         }
      }
   }
   // Allocate memory for the interior edge normals.
   _edgeNormals.resize(edgeIndex);

   //
   // Build the vertex, edge, and face normals.
   //

   std::fill(_vertexNormals.begin(), _vertexNormals.end(),
             ext::filled_array<Vertex>(0));
   std::fill(_edgeNormals.begin(), _edgeNormals.end(),
             ext::filled_array<Vertex>(0));
   Vertex normal, norm, x, y;
   std::size_t i, inc;
   // For each simplex.
   for (std::size_t n = 0; n != _iss.indexedSimplices.size(); ++n) {
      _iss.getSimplex(n, &simplex);

      // Compute the face normal.
      x = simplex[2];
      x -= simplex[1];
      y = simplex[0];
      y -= simplex[1];
      ext::cross(x, y, &normal);
      ext::normalize(&normal);
      _faceNormals[n] = normal;

      // Contribute to the vertex normal.
      for (m = 0; m != 3; ++m) {
         x = simplex[(m+1)%3];
         x -= simplex[m];
         ext::normalize(&x);
         y = simplex[(m+2)%3];
         y -= simplex[m];
         ext::normalize(&y);
         i = _iss.indexedSimplices[n][m];
         norm = normal;
         // Multiply by the angle between the edges.
         norm *= std::acos(ext::dot(x, y));
         _vertexNormals[i] += norm;
      }

      // Contribute to the edge normal.
      for (m = 0; m != 3; ++m) {
         inc = _incidentEdges[n][m];
         // If there is an m_th incident interior edge.
         if (inc != std::size_t(-1)) {
            _edgeNormals[inc] += normal;
         }
      }
   }

   // Normalize the vertex directions.
   for (std::size_t n = 0; n != _vertexNormals.size(); ++n) {
      ext::normalize(&_vertexNormals[n]);
   }
   // Normalize the edge directions.
   for (std::size_t n = 0; n != _edgeNormals.size(); ++n) {
     ext::normalize(&_edgeNormals[n]);
   }
}

} // namespace geom
}
