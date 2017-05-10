// -*- C++ -*-

#if !defined(__cpt_BRep2_ipp__)
#error This file is an implementation detail of the class BRep.
#endif

namespace stlib
{
namespace cpt
{

//
// Free functions for performing clipping.
//


// Make this line equidistant from the two points and oriented so that
// p is above the line.
template<typename T>
void
makeEquidistantLine(geom::Line_2<T>* line,
                    const std::array<T, 2>& p,
                    const std::array<T, 2>& q)
{
  assert(p != q);
  std::array<T, 2> t = T(0.5) * (p - q);
  std::array<T, 2> n = t;
  geom::rotatePiOver2(&n);
  *line = geom::Line_2<T>(q + t, q + t + n);
}


// Used for clipping the characteristic polygon for a vertex.
// Clip the polygon using the lines that are equidistant from the vertex
// and the points in clip_points.
template<typename T>
void
clip(geom::ScanConversionPolygon<std::ptrdiff_t, T>* poly,
     const Vertex<2, T>& vertex,
     const std::vector< std::array<T, 2> >& clipPoints)
{
  geom::Line_2<T> line;
  std::array<T, 2> displacement;
  for (typename std::vector< std::array<T, 2> >::const_iterator
       i = clipPoints.begin(); i != clipPoints.end(); ++i) {
    displacement = *i;
    displacement -= vertex.getLocation();
    // If the vertex is not at the clipping point and the clipping point
    // is above the two neighboring faces.
    if (vertex.getLocation() != *i &&
        ext::dot(displacement, vertex.getRightNormal()) > 0 &&
        ext::dot(displacement, vertex.getLeftNormal()) > 0) {
      makeEquidistantLine(&line, vertex.getLocation(), *i);
      poly->clip(line);
    }
  }
}


// Find the best clipping point for a characteristic polygon of a vertex
// or face.  The point on the surface has the specified normal.
template<typename T>
bool
computeBestClipPoint(const std::array<T, 2>& point,
                     const std::array<T, 2>& normal,
                     const std::vector< std::array<T, 2> >& clipPoints,
                     std::array<T, 2>* bestPoint)
{
  T minimumValue = std::numeric_limits<T>::max();
  T val, den;
  std::array<T, 2> vec;

  for (typename std::vector< std::array<T, 2> >::const_iterator
       i = clipPoints.begin(); i != clipPoints.end(); ++i) {
    vec = *i - point;
    den = ext::dot(vec, normal);
    if (den > std::numeric_limits<T>::epsilon()) {
      val = ext::dot(vec, vec) / den;
      if (val < minimumValue) {
        minimumValue = val;
        *bestPoint = *i;
      }
    }
  }
  if (minimumValue != std::numeric_limits<T>::max()) {
    return true;
  }
  return false;
}


// Used for clipping the characteristic polygon for a vertex.
// Clip the polygon once using the best point in clipPoints.
template<typename T>
void
oneClip(geom::ScanConversionPolygon<std::ptrdiff_t, T>* poly,
        const Vertex<2, T>& vertex,
        const std::vector<std::array<T, 2> >& clipPoints)
{
  std::array<T, 2> normal = vertex.getRightNormal() +
                            vertex.getLeftNormal();
  ext::normalize(&normal);
  std::array<T, 2> bestPoint;
  if (computeBestClipPoint(vertex.getLocation(), normal, clipPoints,
                           &bestPoint)) {
    geom::Line_2<T> line;
    makeEquidistantLine(&line, vertex.getLocation(), bestPoint);
    poly->clip(line);
  }
}


// If the point is above the segment:
// Find a line that lies above the portion of the equidistant parabola
// above the segment.  The line is oriented so the segment is above
// the line.  Return true.
// Else:
// Return false.
template<typename T>
bool
makeEquidistantLine(geom::Line_2<T>* line,
                    const std::array<T, 2>& source,
                    const std::array<T, 2>& target,
                    const std::array<T, 2>& tangent,
                    const std::array<T, 2>& normal,
                    const std::array<T, 2>& point)
{
  const T d = ext::dot(normal, point - source);
  // Return false if the point is not above the segment.
  if (d <= 0) {
    return false;
  }

  T x = ext::dot(point - target, tangent);
  const std::array<T, 2> p1 = target + ((x * x + d * d) / (2 * d)) * normal;
  x = ext::dot(point - source, tangent);
  const std::array<T, 2> p2 = source + ((x * x + d * d) / (2 * d)) * normal;
  *line = geom::Line_2<T>(p1, p2);
  return true;
}


// Clip this polygon using the lines that are equidistant from
// the face and the points in clipPoints.
template<typename T>
void
clip(geom::ScanConversionPolygon<std::ptrdiff_t, T>* poly,
     const Face<2, T>& face,
     const std::vector< std::array<T, 2> >& clipPoints)
{
  geom::Line_2<T> line;
  typename std::vector< std::array<T, 2> >::const_iterator
  i = clipPoints.begin();
  const typename std::vector< std::array<T, 2> >::const_iterator
  i_end = clipPoints.end();
  for (; i != i_end; ++i) {
    if (face.getSource() != *i && face.getTarget() != *i) {
      // Clip for points above the face.
      if (makeEquidistantLine(&line, face.getSource(), face.getTarget(),
                              face.getTangent(), face.getNormal(), *i)) {
        poly->clip(line);
      }
      // Clip for points below the face.
      if (makeEquidistantLine(&line, face.getTarget(), face.getSource(),
                              -face.getTangent(), -face.getNormal(), *i)) {
        poly->clip(line);
      }
    }
  }
}


// Clip the polygon using the two best points in clipPoints.
template<typename T>
void
oneClip(geom::ScanConversionPolygon<std::ptrdiff_t, T>* poly,
        const Face<2, T>& face,
        const std::vector< std::array<T, 2> >& clipPoints)
{
  std::array<T, 2> bestPoint;
  const std::array<T, 2> mid_point = T(0.5) *
                                     (face.getSource() + face.getTarget());
  geom::Line_2<T> line;
  // Clip for points above the face.
  if (computeBestClipPoint(mid_point, face.getNormal(), clipPoints,
                           &bestPoint)) {
    if (makeEquidistantLine(&line, face.getSource(), face.getTarget(),
                            face.getTangent(), face.getNormal(),
                            bestPoint)) {
      poly->clip(line);
    }
  }
  // Clip for points below the face.
  if (computeBestClipPoint(mid_point, -face.getNormal(), clipPoints,
                           &bestPoint)) {
    if (makeEquidistantLine(&line, face.getTarget(), face.getSource(),
                            - face.getTangent(), - face.getNormal(),
                            bestPoint)) {
      poly->clip(line);
    }
  }
}

//! A class for a b-rep in 2-D.
template<typename T>
class BRep<2, T> :
  public geom::IndSimpSetIncAdj<2, 1, T>
{
  //
  // Public constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t N = 2;

  //
  // Private types.
  //

private:

  //! The base is an indexed simplex set with incidence and adjacency information.
  typedef geom::IndSimpSetIncAdj < N, N - 1, T > Base;

  //
  // Public types.
  //

public:

  //! The number type.
  typedef T Number;
  //! A bounding box.
  typedef geom::BBox<Number, N> BBox;
  //! The lattice.
  typedef geom::RegularGrid<N, T> Lattice;
  //! The grid type.
  typedef cpt::Grid<N, Number> Grid;
  //! A point in 2-D.
  typedef typename Grid::Point Point;
  //! An index extent in 2-D.
  typedef typename Grid::SizeList SizeList;
  //! A multi-index in 2-D.
  typedef typename Grid::IndexList IndexList;
  //! An indexed simplex.
  typedef typename Base::IndexedSimplex IndexedSimplex;

  //
  // Private types.
  //

private:

  typedef cpt::Vertex<N, T> VertexDistance;
  typedef cpt::Face<N, T> FaceDistance;
  // CONTINUE REMOVE
  //! The segment type.
  //typedef geom::Segment<N,T> Segment;
  //! The polygon type.
  typedef geom::ScanConversionPolygon<std::ptrdiff_t, T> Polygon;
  //! An index range.
  typedef typename Grid::Range Range;
  //! An index bounding box.
  typedef geom::BBox<std::ptrdiff_t, N> IndexBBox;

  //
  // Member data.
  //

private:

  //! The unit outward normals of the faces.
  std::vector<Point> _faceNormals;
  //! The face identifiers.
  std::vector<std::size_t> _faceIdentifiers;

  //
  // Private using the base member functions.
  //

  // Simplex Accessors
  using Base::getSimplicesBegin;
  using Base::getSimplicesEnd;
  using Base::getSimplexVertex;
  using Base::getSimplex;

  // Simplex Adjacency Accessors
  using Base::getMirrorIndex;

  // Face accessors.
  using Base::computeFacesSize;
  using Base::getFacesBeginning;
  using Base::getFacesEnd;
  using Base::isOnBoundary;

  // Other Accessors
  using Base::isVertexOnBoundary;

  //--------------------------------------------------------------------------
  // \name Constructors etc.
  //@{
public:

  //! Default constructor.  An empty b-rep.
  BRep() :
    Base(),
    _faceNormals(),
    _faceIdentifiers() {}

  //! Copy constructor.
  BRep(const BRep& other) :
    Base(other),
    _faceNormals(other._faceNormals),
    _faceIdentifiers(other._faceIdentifiers) {}

  //! Assignment operator.
  BRep&
  operator=(const BRep& other)
  {
    // Avoid assignment to self
    if (&other != this) {
      Base::operator=(other);
      _faceNormals = other._faceNormals;
      _faceIdentifiers = other._faceIdentifiers;
    }
    // Return *this so assignments can chain
    return *this;
  }

  //! Trivial destructor.
  ~BRep() {}

  //! Make from vertices and faces.  Throw away irrelevant faces.
  /*!
    \param vertices The locations of the vertices.
    \param indexedSimplices The vector of tuples of vertex indices that describe the mesh simplices.
    \param cartesianDomain is the domain of interest.
    \param maximumDistance is how far the distance will be computed.

    Make the b-rep from vertex coordinates and face indices.
    Clip the b-rep so that faces outside the relevant Cartesian domain
    are thrown away.  (Any face within \c maximumDistance of
    \c cartesianDomain is regarded as relevant.)

    This constructor calls \c make() with the same arguments.
  */
  BRep(const std::vector<std::array<Number, N> >& vertices,
       const std::vector<std::array<std::size_t, N> >& indexedSimplices,
       const BBox& cartesianDomain,
       const Number maximumDistance) :
    Base(),
    _faceNormals(),
    _faceIdentifiers()
  {
    make(vertices, indexedSimplices, cartesianDomain, maximumDistance);
  }

  //! Make from vertices and faces.
  /*!
    \param vertices The locations of the vertices.
    \param indexedSimplices The vector of tuples of vertex indices that describe the mesh simplices.
  */
  void
  make(const std::vector<std::array<Number, N> >& vertices_,
       const std::vector<std::array<std::size_t, N> >& indexedSimplices_)
  {
    // The indexed simplex set.
    Base::vertices = vertices_;
    Base::indexedSimplices = indexedSimplices_;
    Base::updateTopology();
    // The face normals.
    _faceNormals.resize(Base::indexedSimplices.size());
    geom::computeSimplexNormals(*this, &_faceNormals);
    // The face identifiers.
    _faceIdentifiers.resize(Base::indexedSimplices.size());
    for (std::size_t n = 0; n != _faceIdentifiers.size(); ++n) {
      _faceIdentifiers[n] = n;
    }
  }

  //! Make from vertices and faces.
  /*!
    \param vertices The locations of the vertices.
    \param indexedSimplices The vector of tuples of vertex indices that describe the mesh simplices.
    \param cartesianDomain is the domain of interest.
    \param maximumDistance is how far the distance will be computed.

    Make the b-rep from vertex coordinates and face indices.
    Clip the b-rep so that faces outside the relevant Cartesian domain
    are thrown away.  (Any face within \c maximumDistance of
    \c cartesianDomain is regarded as relevant.)
  */
  void
  make(const std::vector<std::array<Number, N> >& vertices,
       const std::vector<std::array<std::size_t, N> >& indexedSimplices,
       const BBox& cartesianDomain, const Number maximumDistance)
  {
    // Determine the bounding box containing the points of interest.
    BBox bbox = {cartesianDomain.lower - maximumDistance,
                 cartesianDomain.upper + maximumDistance
                };

    // Build a mesh from the vertices and indexed simplices.
    geom::IndSimpSet < N, N - 1, T > mesh(vertices, indexedSimplices);

    // Determine the set of overlapping faces.
    std::vector<std::size_t> overlappingFaces;
    geom::determineOverlappingSimplices
    (mesh, bbox, std::back_inserter(overlappingFaces));

    // Build this mesh from the subset of overlapping faces.
    geom::buildFromSubsetSimplices(mesh, overlappingFaces.begin(),
                                   overlappingFaces.end(), this);

    // The face normals.
    _faceNormals.resize(Base::indexedSimplices.size());
    geom::computeSimplexNormals(*this, &_faceNormals);

    // The overlapping faces are the face identifiers.
    std::vector<std::size_t> tmp(overlappingFaces.begin(),
                                 overlappingFaces.end());
    _faceIdentifiers.swap(tmp);
  }

  //@}
  //--------------------------------------------------------------------------
  // \name Accessors for the vertices.
  //@{
public:

  std::size_t
  getRightFace(const std::size_t vertexIndex) const
  {
    if (Base::indexedSimplices[Base::incident(vertexIndex, 0)][1] ==
        vertexIndex) {
      return Base::incident(vertexIndex, 0);
    }
    if (Base::incident.size(vertexIndex) == 2) {
      return Base::incident(vertexIndex, 1);
    }
    return std::numeric_limits<std::size_t>::max();
  }

  std::size_t
  getLeftFace(const std::size_t vertexIndex) const
  {
    if (Base::indexedSimplices[Base::incident(vertexIndex, 0)][0] ==
        vertexIndex) {
      return Base::incident(vertexIndex, 0);
    }
    if (Base::incident.size(vertexIndex) == 2) {
      return Base::incident(vertexIndex, 1);
    }
    return std::numeric_limits<std::size_t>::max();
  }

  //@}
  //--------------------------------------------------------------------------
  // \name Size accessors.
  //@{
public:

  //! Return one past the maximum face identifier.
  std::size_t
  getFaceIdentifierUpperBound() const
  {
    if (_faceIdentifiers.empty()) {
      return 0;
    }
    return *(_faceIdentifiers.end() - 1) + 1;
  }

  //@}
  //--------------------------------------------------------------------------
  // \name Mathematical Operations
  //@{
public:

  //! Calculate the signed distance, closest point, etc. for all the points in the grid.
  /*!
    \param lattice is the lattice on which the grids lie.
    \param grids is the container of grids.  Each one holds the distance,
    closest point, etc. arrays.
    \param maximumDistance is the distance to calculate distance away from the
    curve.
    \param arePerformingLocalClipping determines whether local clipping
    will be performed on the characteristic polygons.
    \param arePerformingGlobalClipping can be 0, 1 or 2.  0 indicates that
    no global clipping will be done.  1 indicates limited global clipping;
    2 indicates full global clipping.
    \param globalPoints is the set of points used in global clipping.

    \return the number of points scan converted (counting multiplicities)
    and the number of distances set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPoint(const Lattice& lattice,
                      std::vector<Grid>* grids,
                      const Number maximumDistance,
                      const bool arePerformingLocalClipping,
                      const std::size_t arePerformingGlobalClipping,
                      const std::vector<Point>& globalPoints) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::vector<IndexList> indices;
    std::vector<Point> cartesianPoints;
    BBox box;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, closest points and closest faces for the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {

      // If the i_th vertex has two adjacent faces
      // and the curve is either convex or concave here.
      if (getVertex(i, &vert) && vert.isConvexOrConcave()) {

        // Get a bounding box around the vertex.
        getVertexBBox(i, maximumDistance, &box);
        // Find the first relevant grid.
        firstGrid = 0;
        for (; firstGrid != grids->size(); ++firstGrid) {
          if (geom::doOverlap(gridDomains[firstGrid], box)) {
            break;
          }
        }
        // If there are no relevant grids, continue with the next vertex.
        if (firstGrid == grids->size()) {
          continue;
        }

        // Make the polygon containing the closest points.
        vert.buildCharacteristicPolygon(&poly, maximumDistance);

        // If doing limited global clipping
        if (arePerformingGlobalClipping == 1) {
          oneClip(&poly, vert, globalPoints);
        }
        // Else if doing full global clipping
        else if (arePerformingGlobalClipping == 2) {
          clip(&poly, vert, globalPoints);
        }

        // Convert to index coordinates.
        lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                          poly.getVerticesEnd());

        // Scan convert the polygon.
        indices.clear();
        poly.scanConvert(std::back_inserter(indices), lattice.getExtents());
        scanConversionCount += indices.size();

        // Make an index bounding box around the scan converted points.
        indexBox = geom::specificBBox<IndexBBox>
          (indices.begin(), indices.end());
        // Compute the Cartesian coordinates of the scan converted points.
        cartesianPoints.resize(indices.size());
        for (std::size_t i = 0; i != indices.size(); ++i) {
          for (std::size_t n = 0; n != N; ++n) {
            cartesianPoints[i][n] = indices[i][n];
          }
        }
        lattice.convertIndicesToLocations(cartesianPoints.begin(),
                                          cartesianPoints.end());

        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the vertex could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for scan converted grid pts.
            distanceCount += (*grids)[n].computeClosestPointTransform
                             (indices, cartesianPoints, vert, maximumDistance).second;
          }
        }
      }
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //

    FaceDistance face, prev, next;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {

      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);

      if (arePerformingLocalClipping &&
          Base::adjacent[i][0] != std::numeric_limits<std::size_t>::max() &&
          Base::adjacent[i][1] != std::numeric_limits<std::size_t>::max()) {
        // Get the previous and next face.
        getFace(Base::adjacent[i][1], &prev);
        getFace(Base::adjacent[i][0], &next);
        // Make the polygon that contains the points with positive and
        // negative distance.  Do local clipping.
        face.buildCharacteristicPolygon(&poly, prev, next, maximumDistance);
      }
      // Make the characteristic polygon and don't do local clipping.
      else {
        // Make the polygon that contains the points with positive and
        // negative distance.
        face.buildCharacteristicPolygon(&poly, maximumDistance);
      }

      // If doing limited global clipping
      if (arePerformingGlobalClipping == 1) {
        oneClip(&poly, face, globalPoints);
      }
      // Else if doing full global clipping
      else if (arePerformingGlobalClipping == 2) {
        clip(&poly, face, globalPoints);
      }

      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Scan convert the polygon.
      indices.clear();
      poly.scanConvert(std::back_inserter(indices), lattice.getExtents());
      scanConversionCount += indices.size();

      // Make an index bounding box around the scan converted points.
      indexBox = geom::specificBBox<IndexBBox>(indices.begin(), indices.end());
      // Compute the Cartesian coordinates of the scan converted points.
      cartesianPoints.resize(indices.size());
      for (std::size_t i = 0; i != indices.size(); ++i) {
        for (std::size_t n = 0; n != N; ++n) {
          cartesianPoints[i][n] = indices[i][n];
        }
      }
      lattice.convertIndicesToLocations(cartesianPoints.begin(),
                                        cartesianPoints.end());

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransform
                           (indices, cartesianPoints, face, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Calculate the unsigned distance, closest point, etc. for all the points in the grid.
  /*!
    \param lattice is the lattice on which the grids lie.
    \param grids is the container of grids.  Each one holds the distance,
    closest point, etc. arrays.
    \param maximumDistance is the distance to calculate distance away from the
    curve.
    \param arePerformingLocalClipping determines whether local clipping will be performed
    on the characteristic polygons.
    \param arePerformingGlobalClipping can be 0, 1 or 2.  0 indicates that no global
    clipping will be done.  1 indicates limited global clipping; 2 indicates
    full global clipping.
    \param globalPoints is the set of points used in global clipping.

    \return the number of points scan converted (counting multiplicities)
    and the number of distances set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointUnsigned(const Lattice& lattice,
                              std::vector<Grid>* grids,
                              const Number maximumDistance,
                              const bool arePerformingLocalClipping,
                              const std::size_t arePerformingGlobalClipping,
                              const std::vector<Point>& globalPoints) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::vector<IndexList> indices;
    std::vector<Point> cartesianPoints;
    BBox box;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, gradient, closest points and closest faces for
    // the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the i_th vertex has one more adjacent faces, compute the distance.
      if (! getVertexUnsigned(i, &vert)) {
        continue;
      }

      // Get a bounding box around the vertex.
      getVertexBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Make the polygon containing the closest points.
      vert.buildCharacteristicPolygonUnsigned(&poly, maximumDistance);

      // If doing limited global clipping
      if (arePerformingGlobalClipping == 1) {
        oneClip(&poly, vert, globalPoints);
      }
      // Else if doing full global clipping
      else if (arePerformingGlobalClipping == 2) {
        clip(&poly, vert, globalPoints);
      }

      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Scan convert the polygon.
      indices.clear();
      poly.scanConvert(std::back_inserter(indices), lattice.getExtents());
      scanConversionCount += indices.size();

      // Make an index bounding box around the scan converted points.
      indexBox = geom::specificBBox<IndexBBox>(indices.begin(), indices.end());
      // Compute the Cartesian coordinates of the scan converted points.
      cartesianPoints.resize(indices.size());
      for (std::size_t i = 0; i != indices.size(); ++i) {
        for (std::size_t n = 0; n != N; ++n) {
          cartesianPoints[i][n] = indices[i][n];
        }
      }
      lattice.convertIndicesToLocations(cartesianPoints.begin(),
                                        cartesianPoints.end());

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the vertex could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (indices, cartesianPoints, vert, maximumDistance).second;
        }
      }
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //

    FaceDistance face, prev, next;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {

      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);

      if (arePerformingLocalClipping &&
          Base::adjacent[i][0] != std::numeric_limits<std::size_t>::max() &&
          Base::adjacent[i][1] != std::numeric_limits<std::size_t>::max()) {
        // Get the previous and next face.
        getFace(Base::adjacent[i][1], &prev);
        getFace(Base::adjacent[i][0], &next);
        // Make the polygon that contains the points with positive and
        // negative distance.  Do local clipping.
        face.buildCharacteristicPolygon(&poly, prev, next, maximumDistance);
      }
      // Make the characteristic polygon and don't do local clipping.
      else {
        // Make the polygon that contains the points with positive and
        // negative distance.
        face.buildCharacteristicPolygon(&poly, maximumDistance);
      }

      // If doing limited global clipping
      if (arePerformingGlobalClipping == 1) {
        oneClip(&poly, face, globalPoints);
      }
      // Else if doing full global clipping
      else if (arePerformingGlobalClipping == 2) {
        clip(&poly, face, globalPoints);
      }

      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Scan convert the polygon.
      indices.clear();
      poly.scanConvert(std::back_inserter(indices), lattice.getExtents());
      scanConversionCount += indices.size();

      // Make an index bounding box around the scan converted points.
      indexBox = geom::specificBBox<IndexBBox>(indices.begin(), indices.end());
      // Compute the Cartesian coordinates of the scan converted points.
      cartesianPoints.resize(indices.size());
      for (std::size_t i = 0; i != indices.size(); ++i) {
        for (std::size_t n = 0; n != N; ++n) {
          cartesianPoints[i][n] = indices[i][n];
        }
      }
      lattice.convertIndicesToLocations(cartesianPoints.begin(),
                                        cartesianPoints.end());

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (indices, cartesianPoints, face, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the characteristic polygons instead of polygon scan conversion.
  std::pair<std::size_t, std::size_t>
  computeClosestPointUsingBBox(const Lattice& lattice,
                               std::vector<Grid>* grids,
                               const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    BBox box, characteristicBox;
    Range indexRange;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //
    SizeList extents;
    FaceDistance face;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {
      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);
      // Make the polygon that contains the points with positive and
      // negative distance.
      face.buildCharacteristicPolygon(&poly, maximumDistance);
      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Make a bounding box around the polygon.
      characteristicBox = geom::specificBBox<BBox>(poly.getVerticesBeginning(),
                                                   poly.getVerticesEnd());
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = characteristicBox.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = characteristicBox.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransform
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the distance, closest points and closest faces for the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the i_th vertex has two adjacent faces and the curve is either
      // convex or concave here, then compute the distance.
      if (!(getVertex(i, &vert) && vert.isConvexOrConcave())) {
        continue;
      }

      // Get a bounding box around the vertex.
      getVertexBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Make the polygon containing the closest points.
      vert.buildCharacteristicPolygon(&poly, maximumDistance);
      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Make a bounding box around the polygon.
      characteristicBox = geom::specificBBox<BBox>(poly.getVerticesBeginning(),
                                                   poly.getVerticesEnd());
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = characteristicBox.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = characteristicBox.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the vertex could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransform
                           (lattice, indexRange, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the characteristic polygons instead of polygon scan conversion.
  std::pair<std::size_t, std::size_t>
  computeClosestPointUnsignedUsingBBox(const Lattice& lattice,
                                       std::vector<Grid>* grids,
                                       const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    BBox box, characteristicBox;
    Range indexRange;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //

    SizeList extents;
    FaceDistance face, prev, next;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {
      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);
      // Make the polygon that contains the points with positive and
      // negative distance.
      face.buildCharacteristicPolygon(&poly, maximumDistance);
      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Make a bounding box around the polygon.
      characteristicBox = geom::specificBBox<BBox>(poly.getVerticesBeginning(),
                                                   poly.getVerticesEnd());
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = characteristicBox.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = characteristicBox.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the distance, gradient, closest points and closest faces for
    // the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the i_th vertex has one more adjacent faces, compute the distance.
      if (! getVertexUnsigned(i, &vert)) {
        continue;
      }

      // Get a bounding box around the vertex.
      getVertexBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Make the polygon containing the closest points.
      vert.buildCharacteristicPolygonUnsigned(&poly, maximumDistance);
      // Convert to index coordinates.
      lattice.convertLocationsToIndices(poly.getVerticesBeginning(),
                                        poly.getVerticesEnd());

      // Make a bounding box around the polygon.
      characteristicBox = geom::specificBBox<BBox>(poly.getVerticesBeginning(),
                                                   poly.getVerticesEnd());
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = characteristicBox.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = characteristicBox.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the vertex could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the primitives instead of polygon scan conversion.
  std::pair<std::size_t, std::size_t>
  computeClosestPointUsingBruteForce(const Lattice& lattice,
                                     std::vector<Grid>* grids,
                                     const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    BBox box;
    Range indexRange;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //
    SizeList extents;
    FaceDistance face;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {
      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);
      // Convert the face bounding box to index coordinates.
      lattice.convertBBoxLocationsToIndices(&box);
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = box.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = box.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransform
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the distance, closest points and closest faces for the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the i_th vertex has two adjacent faces and the curve is either
      // convex or concave here, then compute the distance.
      if (!(getVertex(i, &vert) && vert.isConvexOrConcave())) {
        continue;
      }

      // Get a bounding box around the vertex.
      getVertexBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Convert the vertex bounding box to index coordinates.
      lattice.convertBBoxLocationsToIndices(&box);
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = box.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = box.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the vertex could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransform
                           (lattice, indexRange, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the primitives instead of polygon scan conversion.
  std::pair<std::size_t, std::size_t>
  computeClosestPointUnsignedUsingBruteForce(const Lattice& lattice,
      std::vector<Grid>* grids,
      const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    BBox box;
    Range indexRange;
    IndexBBox indexBox;
    Polygon poly;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the distance, closest points and closest faces for the faces.
    //

    SizeList extents;
    FaceDistance face, prev, next;
    for (std::size_t i = 0; i != Base::indexedSimplices.size(); ++i) {
      // Get a bounding box around the face.
      getFaceBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Get the i_th face.
      getFace(i, &face);
      // Convert the face bounding box to index coordinates.
      lattice.convertBBoxLocationsToIndices(&box);
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = box.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = box.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the face could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the distance, gradient, closest points and closest faces for
    // the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the i_th vertex has one more adjacent faces, compute the distance.
      if (! getVertexUnsigned(i, &vert)) {
        continue;
      }

      // Get a bounding box around the vertex.
      getVertexBBox(i, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next vertex.
      if (firstGrid == grids->size()) {
        continue;
      }

      // Convert the vertex bounding box to index coordinates.
      lattice.convertBBoxLocationsToIndices(&box);
      // Convert to an integer index range.
      for (std::size_t n = 0; n != N; ++n) {
        // Ceiling.
        const std::ptrdiff_t lower = box.lower[n] + 1;
        indexBox.lower[n] = lower;
        // Floor for closed range.
        const std::ptrdiff_t upper = box.upper[n];
        indexBox.upper[n] = upper;
        // Add one to convert from a closed range to an extent.
        extents[n] = upper - lower + 1;
      }
      indexRange = Range(extents, indexBox.lower);
      scanConversionCount += ext::product(extents);

      // Loop over the grids.
      for (std::size_t n = firstGrid; n != grids->size(); ++n) {
        // If the vertex could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Return the bounding box that contains the mesh.
  BBox
  computeBBox() const
  {
    return geom::specificBBox<BBox>
      (Base::vertices.begin(), Base::vertices.end());
  }

private:

  //! Compute index bounding boxes for each grid.
  void
  computeIndexBoundingBoxes(const std::vector<Grid>& grids,
                            std::vector<IndexBBox>* gridIndexBBoxes) const
  {
    assert(grids.size() == gridIndexBBoxes->size());
    IndexList upper;
    for (std::size_t n = 0; n != grids.size(); ++n) {
      const SizeList& extents = grids[n].getRanges().extents();
      const IndexList& bases = grids[n].getRanges().bases();
      upper = bases;
      for (std::size_t i = 0; i != upper.size(); ++i) {
        upper[i] += extents[i] - 1;
      }
      (*gridIndexBBoxes)[n].lower = bases;
      (*gridIndexBBoxes)[n].upper = upper;
    }
  }

  //@}
  //--------------------------------------------------------------------------
  // \name File I/O.
  //@{
public:

  //! Display information about the b-rep.
  /*!
    Report if the manifold is closed.
  */
  void
  displayInformation(std::ostream& out) const
  {
    geom::printInformation(out, *this);
  }

  //! Display the b-rep.
  void
  display(std::ostream& out) const
  {
    geom::writeAscii(out, *this);
    out << _faceNormals << _faceIdentifiers;
  }

  //@}

private:

  //
  // Private member functions.
  //

  //! Make a vertex for computing the signed distance.
  /*!
    If the vertex has two neighboring faces, make the vertex and return true.
    Otherwise return false.
  */
  bool
  getVertex(const std::size_t index, VertexDistance* vert) const
  {
    if (Base::incident.size(index) != 2) {
      return false;
    }

    const std::size_t right = getRightFace(index);
    const std::size_t left = getLeftFace(index);
    vert->make(Base::vertices[index], _faceNormals[right], _faceNormals[left],
               right);
    return true;
  }

  //! Make a vertex for computing the unsigned distance.
  /*!
    If the vertex has one or more neighboring faces, make the vertex and
    return true. Otherwise return false.
  */
  bool
  getVertexUnsigned(const std::size_t index, VertexDistance* vert) const
  {
    const std::size_t Unknown = std::numeric_limits<std::size_t>::max();
    const std::size_t right = getRightFace(index);
    const std::size_t left = getLeftFace(index);
    if (right != Unknown && left != Unknown) {
      vert->make(Base::vertices[index], _faceNormals[right],
                 _faceNormals[left], right);
      return true;
    }
    else if (right != Unknown) {
      vert->make(Base::vertices[index], _faceNormals[right],
                 ext::filled_array<Point>(0.0), right);
      return true;
    }
    else if (left != Unknown) {
      vert->make(Base::vertices[index], ext::filled_array<Point>(0.0),
                 _faceNormals[left], left);
      return true;
    }
    assert(false);
    return false;
  }

  // Make a bounding box around the vertex specified by the index.
  // Enlarge the bounding box by the maximumDistance.
  void
  getVertexBBox(const std::size_t index, const Number maximumDistance,
                BBox* box) const
  {
    box->lower = Base::vertices[index] - maximumDistance;
    box->upper = Base::vertices[index] + maximumDistance;
  }

  //! Make a face.
  void
  getFace(std::size_t index, FaceDistance* face) const
  {
    face->make(getSimplexVertex(index, 0), getSimplexVertex(index, 1),
               _faceNormals[index], index);
  }

  // Make a bounding box around the face specified by the index.
  // Enlarge the bounding box by the maximumDistance.
  void
  getFaceBBox(const std::size_t index, const Number maximumDistance,
              BBox* box) const
  {
    *box = geom::specificBBox<BBox>
      (std::array<Point, 2>{{getSimplexVertex(index, 0),
            getSimplexVertex(index, 1)}});
    offset(box, maximumDistance);
  }
};


// CONTINUE
#if 0
template<typename T>
inline
void
Grid<2, T>::
floodFill(Number farAway,
          const std::vector<Point>& vertices,
          const std::vector<SizeList>& faces)
{
  // First try to determine the sign of the distance from the known distances
  // and then flood fill.
  if (floodFill(farAway)) {
    return;
  }
  // If the above did not succeed then there are no known distances.

  // Ensure that the mesh is not degenerate.
  assert(vertices.size() != 0 && faces.size() != 0);

  // Find the Cartesian location of one of the grid points.
  Point location(0, 0);
  convertIndexToLocation(location);

  // We will find the closest point on the mesh to the grid location.

  // Compute the distances to the vertices.
  std::vector<Number> vertexDistance(vertices.size());
  for (std::size_t i = 0; i != vertices.size(); ++i) {
    vertexDistance[i] = geom::computeDistance(location, vertices[i]);
  }

  // Find the vertex that is closest to the grid location.
  // We use this to determine an upper bound on the distance.
  const Number upperBoundDistance = *std::min_element(vertexDistance.begin()
                                    vertexDistance.end());

  // Determine the faces that are relevant.
  std::vector<SizeList> closeFaces;
  {
    Number edgeLength, minimumVertexDistance;
    for (std::size_t i = 0; i != faces.size(); ++i) {
      edgeLength = geom::computeDistance(vertices[faces[i][0]],
                                         vertices[faces[i][1]]);
      minimumVertexDistance = std::min(vertexDistance[faces[i][0]],
                                       vertexDistance[faces[i][1]]);
      if (minimumVertexDistance <= upperBoundDistance + edgeLength) {
        closeFaces.push_back(faces[i]);
      }
    }
  }

  // Make a set of the vertex indices that comprise the close faces.
  std::set<std::size_t> indexSet;
  {
    for (std::size_t i = 0; i != closeFaces.size(); ++i) {
      indexSet.insert(closeFaces[i][0]);
      indexSet.insert(closeFaces[i][1]);
    }
  }
  std::vector<std::size_t> closeVertexIndices(indexSet.begin(), indexSet.end());

  // Make an array of the close vertices.
  std::vector<Point> closeVertices(closeVertexIndices.size());
  {
    for (std::size_t i = 0; i != closeVertexIndices.size(); ++i) {
      closeVertices[i] = vertices[closeVertexIndices[i]];
    }
  }

  // Adjust the indices of the close faces.
  {
    for (std::size_t i = 0; i != closeFaces.size(); ++i) {
      for (std::size_t j = 0; j != N; ++j) {
        closeFaces[i][j] =
          std::lower_bound(closeVertexIndices.begin(),
                           closeVertexIndices.end(), closeFaces[i][j]) -
          closeVertexIndices.begin();
      }
    }
  }

  // Make a b-rep from the close faces.
  BRep<N, Number> brep;
  brep.make(closeVertices.begin(), closeVertices.end(),
            closeFaces.begin(), closeFaces.end());

  // Make a grid with a single point.
  BBox dom(location, location);
  SizeList extents = {{1, 1}};
  container::MultiArray<Number, N> dist(extents);
  container::MultiArray<Point, N> gd;
  container::MultiArray<Point, N> cp;
  container::MultiArray<std::size_t, N> cf;
  std::vector<Grid> grids;
  grids.push_back(Grid(dom, dist, gd, cp, cf));

  // Compute the distance from the grid point to the mesh.
  grids[0].initialize();
  {
    std::vector<Point> globalPoints;
    brep.computeClosestPoint(grids, upperBoundDistance * 1.1, true, 0,
                             globalPoints);
  }
  const IndexList zero = {{}};
  Number d = dist(zero);

  // Set the distance to +- farAway.
  assert(d != std::numeric_limits<Number>::max());
  const int signDistance = (d >= 0 ? 1 : -1);
  distance() = signDistance * farAway;
}
#endif

} // namespace cpt
}
