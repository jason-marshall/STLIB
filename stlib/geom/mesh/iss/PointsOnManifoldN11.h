// -*- C++ -*-

/*!
  \file PointsOnManifoldN11.h
  \brief Represent the features of a N-1 mesh.
*/

#if !defined(__geom_PointsOnManifoldN11_h__)
#define __geom_PointsOnManifoldN11_h__


namespace stlib
{
namespace geom {


//! Feature-based manifold data structure.
/*!
  <!--I put an anchor here because I cannot automatically reference this
  class. -->
  \anchor PointsOnManifoldN11T

  \param T is the number type.  By default it is double.
*/
template<std::size_t SpaceD, typename T>
class PointsOnManifold<SpaceD, 1, 1, T> {
   //
   // Public enumerations.
   //

public:

   //! The space dimension, simplex dimension, and spline degree.
   enum {N = SpaceD, M = 1, SD = 1};

   //
   // Private enumerations.
   //

   // CONTINUE: With the Intel compiler, private members are not accessible in
   // nested classes.
#ifdef __INTEL_COMPILER
public:
#else
private:
#endif

   //! The features of the manifold at the vertices.
   enum Feature {NullFeature, CornerFeature, SurfaceFeature};

   //
   // Nested classes.
   //

private:

   //! A face handle. (1-face or 0-face)
   class FaceHandle {
   private:

      Feature _feature;
      std::size_t _index;

   public:

      //
      // Constructors, etc.
      //

      //! Default constructor.  Make an invalid face handle.
      FaceHandle() :
         _feature(NullFeature),
         _index(std::numeric_limits<std::size_t>::max()) {}

      //! Copy constructor.
      FaceHandle(const FaceHandle& other) :
         _feature(other._feature),
         _index(other._index) {}

      //! Assignment operator.
      FaceHandle&
      operator=(const FaceHandle& other) {
         // Avoid assignment to self.
         if (this != &other) {
            _feature = other._feature;
            _index = other._index;
         }
         // Return a reference to this so assignments can chain.
         return *this;
      }

      //! Construct from a feature and an index.
      FaceHandle(const Feature feature, const std::size_t index) :
         _feature(feature),
         _index(index) {}

      //! Trivial destructor.
      ~FaceHandle() {}

      //
      // Accessors.
      //

      //! Get the feature.
      Feature
      getFeature() const {
         return _feature;
      }

      //! Get the index.
      std::size_t
      getIndex() const {
         return _index;
      }

      //
      // Manipulators
      //

      //! Set the feature.
      void
      setFeature(const Feature feature) {
         _feature = feature;
      }

      //! Set the index.
      void
      setIndex(const std::size_t index) {
         _index = index;
      }

      //! Set the face handle to an invalid value.
      void
      setToNull() {
         _feature = NullFeature;
         _index = std::numeric_limits<std::size_t>::max();
      }
   };

   //
   // Private types.
   //

private:

   //! The representation of the surface.
   typedef IndSimpSetIncAdj<N, M, T> SurfaceManifold;
   //! The size type.
   typedef std::size_t SizeType;
   //! The container for the registered points on the manifold.
   typedef std::map<std::size_t, FaceHandle> PointMap;
   //! The representation for a point.
   typedef typename PointMap::value_type Point;

   //
   // Public types.
   //

public:

   //! The number type.
   typedef T Number;
   //! A vertex.
   typedef typename SurfaceManifold::Vertex Vertex;

   //
   // Member data.
   //

private:

   //! The surface manifold.
   SurfaceManifold _surfaceManifold;
   //! The indices of the corner vertices.
   std::vector<int> _cornerIndices;
   //! The vertex features.
   std::vector<Feature> _vertexFeatures;
   //! The registered points on the surface.
   PointMap _points;
   //! The data structure for simplex queries.
   ISS_SimplexQuery<SurfaceManifold> _simplexQuery;
   //! The maximum distance between a point and a corner vertex for the point to be placed there.
   Number _maxCornerDistance;

   //! A cached point identifier.
   mutable std::size_t _cachedIdentifier;
   //! A cached face.
   mutable FaceHandle _cachedFace;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   PointsOnManifold();

   //! Copy constructor not implemented
   PointsOnManifold(const PointsOnManifold&);

   //! Assignment operator not implemented
   PointsOnManifold&
   operator=(const PointsOnManifold&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //!@{

   //! Construct from the mesh and the corners.
   /*!
     \param iss is the indexed simplex set.
     \param cornersBegin The beginning of the range of corner indices.
     \param cornersEnd The end of the range of corner indices.

     Boundary vertices (with only one incident simplex) will also be set as
     corner vertices.

     The maximum corner distance will be
     set to 0.1 times the minimum edge length in surface mesh.  You can change
     this default value with setMaxCornerDistance().
   */
   template<typename IntInIter>
   PointsOnManifold(const IndSimpSet<N, M, T>& iss,
                    IntInIter cornersBegin, IntInIter cornersEnd);

   //! Construct from the mesh and an angle to define corners.
   /*!
     \param iss is the indexed simplex set.
     \param maxAngleDeviation The maximum angle deviation (from straight)
     for a surface vertex.  The rest are corner vertices.  If not specified,
     all interior vertices will be set as surface vertices.

     The maximum corner distance will be
     set to 0.1 times the minimum edge length in surface mesh.  You can change
     this default value with setMaxCornerDistance().
   */
   PointsOnManifold(const IndSimpSet<N, M, T>& iss,
                    Number maxAngleDeviation = -1);

   //! Destructor.  Free internally allocated memory.
   ~PointsOnManifold() {}

   //!@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //!@{

   //! Count the number of registered points.
   SizeType
   countNumberOfPoints() const {
      return SizeType(_points.size());
   }

   //! Count the number of registered points on each feature.
   void
   countNumberOfPointsOnFeatures(SizeType* surfaceCount, SizeType* cornerCount)
   const;

   //! Count the number of points on surface features.
   SizeType
   countNumberOfSurfacePoints() const;

   //! Count the number of points on corner features.
   SizeType
   countNumberOfCornerPoints() const;

   //! Return true if the vertex is a surface feature.
   bool
   isVertexASurfaceFeature(const std::size_t n) const {
      return _vertexFeatures[n] == SurfaceFeature;
   }

   //! Return true if the vertex is a corner feature.
   bool
   isVertexACornerFeature(const std::size_t n) const {
      return _vertexFeatures[n] == CornerFeature;
   }

   //! Return true if the point is on a surface.
   bool
   isOnSurface(const std::size_t identifier) const {
      return isOnFeature(identifier, SurfaceFeature);
   }

   //! Return true if the point is on a corner.
   bool
   isOnCorner(const std::size_t identifier) const {
      return isOnFeature(identifier, CornerFeature);
   }

   //! Return true if the point has been registered.
   bool
   hasPoint(const std::size_t identifier) const {
      return _points.count(identifier) == 1;
   }

   //! Get the simplex index associated with the point.
   /*!
     The point must be on a surface feature.
   */
   int
   getSimplexIndex(const std::size_t identifier) const;

   //! Get the vertex index associated with the point.
   /*!
     The point must be on a corner feature.
   */
   int
   getVertexIndex(const std::size_t identifier) const;

   //! Get the maximum distance between a point and a corner vertex for the point to be placed there.
   Number
   getMaxCornerDistance() const {
      return _maxCornerDistance;
   }

   //! Get the simplices in the neighborhood of the face. (0-face or 1-face)
   template<typename IntInsertIterator>
   void
   getNeighborhood(const std::size_t identifier, IntInsertIterator iter) const {
      // If the vertex is a corner feature.
      if (isOnCorner(identifier)) {
         getCornerNeighborhood(getVertexIndex(identifier), iter);
      }
      // Otherwise, it is a surface feature.
      else {
         getSurfaceNeighborhood(getSimplexIndex(identifier), iter);
      }
   }

   //! Get the specified vertex of the specified surface simplex.
   const Vertex&
   getSurfaceSimplexVertex(const std::size_t simplexIndex, const std::size_t m) const {
      return _surfaceManifold.getSimplexVertex(simplexIndex, m);
   }

   //!@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //!@{

   //! Set the maximum distance between a point and a corner vertex for the point to be placed there.
   /*!
     In the constructor, the maximum corner distance is initialized to
     0.1 times the minimum edge length in the surface manifold.  If this is not
     an appropriate value, change it with this function.
   */
   void
   setMaxCornerDistance(const Number x) {
      _maxCornerDistance = x;
   }

   //! Change the identifier of a registered point.
   void
   changeIdentifier(std::size_t existingIdentifier, std::size_t newIdentifier);

   //! Change the location of a registered point.
   /*!
     Return the new point on the manifold.
   */
   Vertex
   changeLocation(std::size_t pointIdentifier, const Vertex& newLocation);

   //! Change the surface simplex for a registered surface feature.
   void
   changeSurfaceSimplex(std::size_t pointIdentifier, std::size_t simplexIndex);

   //!@}
   //--------------------------------------------------------------------------
   //! \name Insert/Erase Points.
   //!@{

   //! Insert a point at the specified vertex.
   void
   insertAtVertex(std::size_t pointIdentifier, std::size_t vertexIndex);

   //! Insert a point at each vertex.
   /*!
     The point identifiers will be the vertex indices.
   */
   void
   insertAtVertices();

   //! Insert a point at the closest point to the specified position that is near the existing point.
   /*!
     Return the point's position on the manifold.
   */
   Vertex
   insertNearPoint(std::size_t newPointID, const Vertex& position, std::size_t existingPointID);

   //! Insert a point at the closest point to the specified position that is in the neighborhood of one of the existing points.
   /*!
     Return the point's position on the manifold.
   */
   Vertex
   insertNearPoints(std::size_t newPointID, const Vertex& position,
                    std::size_t existingPointID1, std::size_t existingPointID2);

   //! Insert a point at the closest point on a surface feature.
   Vertex
   insertOnASurface(std::size_t pointIdentifier, const Vertex& position);

   //! Insert a point at the closest point to the specified position.
   /*!
     Return the point's position on the manifold.
   */
   Vertex
   insert(std::size_t pointIdentifier, const Vertex& position);

   //! Insert a range of points at their closest points.
   /*!
     The point identifiers will be in the range [0...numPoints).
     Put the positions at the closest points on the manifold to the output
     iterator.
   */
   template<typename PointInputIterator, typename PointOutputIterator>
   void
   insert(PointInputIterator locationsBegin, PointInputIterator locationsEnd,
          PointOutputIterator closestPoints);

   //! Insert the boundary vertices of the mesh.
   /*!
     The point identifiers will be the vertex indices.  The boundary vertices
     are moved to lay on the manifold.
   */
   void
   insertBoundaryVertices(IndSimpSetIncAdj < N, M + 1, T > * mesh);

   //! Insert the boundary vertices of the mesh.
   /*!
     The point identifiers will be the vertex identifiers.  The boundary
     vertices are moved to lay on the manifold.
   */
   template < template<class> class Node,
            template<class> class Cell,
            template<class, class> class Container >
   void
   insertBoundaryVertices(SimpMeshRed < N, M + 1, T, Node, Cell, Container > * mesh);

   //! Erase a point.
   void
   erase(std::size_t pointIdentifier);

   //! Erase all of the points.
   void
   clearPoints() {
      _points.clear();
   }

   //!@}
   //--------------------------------------------------------------------------
   //! \name Closest Point.
   //!@{

   //! Return the closest point on the manifold to the specified position.
   /*!
     \param pointIdentifier The identifier of the registered point.
     \param position The position of the point.

     Cache the new face for the closest point.
   */
   Vertex
   computeClosestPoint(std::size_t pointIdentifier, const Vertex& position) const;

   //! Update the face of a registered point from the cached value.
   void
   updatePoint();

   //!@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //!@{

   //! Print information about the data structure.
   void
   printInformation(std::ostream& out) const;

   //!@}

   //--------------------------------------------------------------------------
   // Private member functions.
   //

private:

   //! Insert the point in the specified simplex.
   /*!
     \return The closest point in the simplex.
   */
   Vertex
   insertInSimplex(std::size_t pointIdentifier, const Vertex& position,
                   std::size_t simplexIndex);

   //! Try to find a corner vertex that is very close to the position.
   /*!
     If there is a close corner, return the index of the vertex.
     Otherwise, return -1.
   */
   std::size_t
   findCornerVertex(const Vertex& position) const;

   //! Compute the closest point in the set of simplices.
   /*!
     Cache the face of the closest point.
   */
   template<typename IntInputIterator>
   Vertex
   computeClosestPointInSimplices(IntInputIterator indicesBegin,
                                  IntInputIterator indicesEnd,
                                  const Vertex& position) const;

   //! Compute the closest point to the simplex.  Return the unsigned distance.
   Number
   computeClosestPointInSimplex(std::size_t simplexIndex, const Vertex& x,
                                Vertex* closestPoint) const;

   //! Return the closest point to the n_th simplex.
   Vertex
   computeClosestPointInSimplex(std::size_t simplexIndex, const Vertex& x) const;

   //! Return true if the point is on the specified feature.
   bool
   isOnFeature(std::size_t identifier, Feature feature) const;

   //! Return true if the face is valid.
   bool
   isValid(const FaceHandle& face) const;

   //! Get the simplices in the neighborhood of the face. (0-face or 1-face)
   template<typename IntInsertIterator>
   void
   getNeighborhood(const FaceHandle& face, IntInsertIterator iter) const;

   //! Get the simplex index and the adjacent simplex indices.
   /*!
     Do not step across corners.
   */
   template<typename IntInsertIterator>
   void
   getSurfaceNeighborhood(const std::size_t simplexIndex, IntInsertIterator iter) const;

   //! Get the incident simplex indices.
   template<typename IntInsertIterator>
   void
   getCornerNeighborhood(const std::size_t vertexIndex, IntInsertIterator iter) const;

   //! Determine the corners from the maximum allowed angle deviation.
   void
   determineCorners(Number maxAngleDeviation);

   //! Determine the boundary corners.
   void
   determineBoundaryCorners();

   //! Record the indices of the corner vertices.
   void
   recordCorners();
};

} // namespace geom
}

#define __geom_PointsOnManifoldN11_ipp__
#include "stlib/geom/mesh/iss/PointsOnManifoldN11.ipp"
#undef __geom_PointsOnManifoldN11_ipp__

#endif
