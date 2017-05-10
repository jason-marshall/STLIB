// -*- C++ -*-

#if !defined(__cpt_BRep3_ipp__)
#error This file is an implementation detail of the class BRep.
#endif

namespace stlib
{
namespace cpt
{

//! A class for a b-rep in 3-D.
template<typename T>
class BRep<3, T> :
  public geom::IndSimpSetIncAdj<3, 2, T>
{
  //
  // Public constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t N = 3;

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
  //! The grid type stores the arrays for distance, closest point, etc.
  typedef cpt::Grid<N, Number> Grid;
  //! A point in 3-D.
  typedef typename Grid::Point Point;
  //! An index extent in 3-D.
  typedef typename Grid::SizeList SizeList;
  //! A multi-index in 3-D.
  typedef typename Grid::IndexList IndexList;
  //! An indexed simplex.
  typedef typename Base::IndexedSimplex IndexedSimplex;

  //
  // Private types.
  //

private:

  typedef typename Base::FaceIterator EdgeIterator;
  typedef cpt::Vertex<N, T> VertexDistance;
  typedef cpt::Edge<N, T> EdgeDistance;
  typedef cpt::Face<N, T> FaceDistance;
  typedef geom::ScanConversionPolyhedron<std::ptrdiff_t, T> Polyhedron;
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
  //! The unit outward normals of the vertices.
  std::vector<Point> _vertexNormals;
  //! The face identifiers.
  std::vector<std::size_t> _faceIdentifiers;

  //
  // Using the base member functions.
  //

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

public:

  //--------------------------------------------------------------------------
  // \name Constructors, etc.
  //@{
public:

  //! Default constructor.  An empty b-rep.
  BRep() :
    Base(),
    _faceNormals(),
    _vertexNormals(),
    _faceIdentifiers() {}

  //! Copy constructor.
  BRep(const BRep& other) :
    Base(other),
    _faceNormals(other._faceNormals),
    _vertexNormals(other._vertexNormals),
    _faceIdentifiers(other._faceIdentifiers) {}

  //! Assignment operator.
  BRep&
  operator=(const BRep& other)
  {
    // Avoid assignment to self
    if (&other != this) {
      Base::operator=(other);
      _faceNormals = other._faceNormals;
      _vertexNormals = other._vertexNormals;
      _faceIdentifiers = other._faceIdentifiers;
    }
    // Return *this so assignments can chain
    return *this;
  }

  //! Destructor.
  ~BRep() {}

  //! Construct from vertices and faces.  Throw away irrelevant ones.
  /*!
    \param vertices The locations of the vertices.
    \param indexedSimplices The vector of tuples of vertex indices that describe the mesh simplices.
    \param cartesianDomain is the domain of interest.
    \param maximumDistance is how far the distance will be computed.

    Make the b-rep from vertex coordinates and face indices.
    Clip the b-rep so that faces outside the relevant Cartesian domain
    are thrown away.  (Any face within \c maximumDistance of
    \c cartesianDomain is regarded as relevant.)

    This constructor calls make() with the same arguments.
  */
  BRep(const std::vector<std::array<Number, N> >& vertices,
       const std::vector<std::array<std::size_t, N> >& indexedSimplices,
       const BBox& cartesianDomain,
       const Number maximumDistance) :
    Base(),
    _faceNormals(),
    _vertexNormals(),
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
  make(const std::vector<std::array<Number, N> >& vertices,
       const std::vector<std::array<std::size_t, N> >& indexedSimplices)
  {
    // The indexed simplex set.
    geom::build(this, vertices, indexedSimplices);
    // The face and vertex normals.
    _faceNormals.resize(indexedSimplices.size());
    _vertexNormals.resize(vertices.size());
    geom::computeSimplexAndVertexNormals(*this, &_faceNormals,
                                         &_vertexNormals);
    // The face identifiers.
    _faceIdentifiers.resize(indexedSimplices.size());
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
       const BBox& cartesianDomain,
       Number maximumDistance)
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

    // The face and vertex normals.
    _faceNormals.resize(Base::indexedSimplices.size());
    _vertexNormals.resize(Base::vertices.size());
    geom::computeSimplexAndVertexNormals(*this, &_faceNormals, &_vertexNormals);

    // The overlapping faces are the face identifiers.
    std::vector<std::size_t> tmp(overlappingFaces.begin(),
                                 overlappingFaces.end());
    _faceIdentifiers.swap(tmp);
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
  // \name Mathematical Operations.
  //@{
public:

  //! Calculate the closest point transform with signed distance to this b-rep.
  /*!
    Calculate the signed distance, closest point, etc. to this b-rep for all
    the points in the grid.

    \param lattice is the lattice on which the grids lie.
    \param grids is the container of grids.  Each one holds the distance,
    closest point, etc. arrays.
    \param maximumDistance is the distance to calculate distance away from the
    surface.

    \return
    Return the number of points scan converted and the number of points
    for which the distance was computed.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPoint(const Lattice& lattice,
                      std::vector<Grid>* grids,
                      const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::vector<IndexList> indices;
    std::vector<Point> cartesianPoints;
    BBox box;
    IndexBBox indexBox;
    Polyhedron poly;
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    performance::Scope _("computeClosestPoint()");

#ifdef STLIB_PERFORMANCE
    std::pair<std::size_t, std::size_t> countPair;
#endif

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
    //
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
      // If there are no relevant grids, continue with the next face.
      if (firstGrid == grids->size()) {
        continue;
      }
      {
        performance::Event event("MakeFacePolyhedra");
        // Get the i_th face.
        getFace(i, &face);
        // Make the polyhedron containing the closest points.
        face.buildCharacteristicPolyhedron(&cartesianPolyhedron,
                                           maximumDistance);
        // Convert the polyhedron to grid coordinates.
        lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
      }
      {
        performance::Event event("ScanConvertFacePolyhedra");
        // Make the scan conversion polyhedron data structure.
        poly = cartesianPolyhedron;
        // Scan convert the polyhedron.
        indices.clear();
        poly.scanConvert(std::back_inserter(indices), lattice);
        scanConversionCount += indices.size();
        performance::record("FaceScanConverted", indices.size());
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
      }
      {
        performance::Event event("FaceCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the face could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for scan converted grid 
            // points.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (indices, cartesianPoints, face, maximumDistance);
            performance::record("FaceDistancesComputed", countPair.first);
            performance::record("FaceDistancesSet", countPair.second);
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (indices, cartesianPoints, face, maximumDistance).second;
#endif
          }
        }
      }
    }

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // If the adjacent faces are known and the surface is either
      // concave or convex at the edge.
      if (getEdge(iter, &edge) && edge.getSignOfDistance() != 0) {
        // Get a bounding box around the edge.
        getEdgeBBox(iter, maximumDistance, &box);
        // Find the first relevant grid.
        firstGrid = 0;
        for (; firstGrid != grids->size(); ++firstGrid) {
          if (geom::doOverlap(gridDomains[firstGrid], box)) {
            break;
          }
        }
        // If there are no relevant grids, continue with the next edge.
        if (firstGrid == grids->size()) {
          continue;
        }
        {
          performance::Event event("MakeEdgePolyhedra");
          // Make the polyhedron containing the closest points.
          edge.buildCharacteristicPolyhedron(&cartesianPolyhedron,
                                             maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
            (cartesianPolyhedron.getVerticesBeginning(),
             cartesianPolyhedron.getVerticesEnd());
        }
        {
          performance::Event event("ScanConvertEdgePolyhedra");
          // Make the scan conversion polyhedron data structure.
          poly = cartesianPolyhedron;
          // Scan convert the polyhedron.
          indices.clear();
          poly.scanConvert(std::back_inserter(indices), lattice);
          scanConversionCount += indices.size();
          performance::record("EdgeScanConverted", indices.size());
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
        }
        {
          performance::Event event("EdgeCpt");
          // Loop over the grids.
          for (std::size_t n = firstGrid; n != grids->size(); ++n) {
            // If the edge could influence this grid.
            if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
              // Compute closest points and distance for scan converted grid pts.
#ifdef STLIB_PERFORMANCE
              countPair = (*grids)[n].computeClosestPointTransform
                (indices, cartesianPoints, edge, maximumDistance);
              performance::record("EdgeDistancesComputed", countPair.first);
              performance::record("EdgeDistancesSet", countPair.second);
              distanceCount += countPair.second;
#else
              distanceCount += (*grids)[n].computeClosestPointTransform
                (indices, cartesianPoints, edge, maximumDistance).second;
#endif
            }
          }
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // If the vertex is not on the boundary.
      if (getVertex(i, &vert)) {
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
        indices.clear();
        if (! vert.isConcave()) {
          {
            performance::Event event("MakeVertexPolyhedra");
            // Make the polyhedron containing the closest points of positive
            // distance.
            vert.buildCharacteristicPolyhedronPositive(&cartesianPolyhedron,
                                                       maximumDistance);
            // Convert the polyhedron to grid coordinates.
            lattice.convertLocationsToIndices
              (cartesianPolyhedron.getVerticesBeginning(),
               cartesianPolyhedron.getVerticesEnd());
          }
          {
            performance::Event event("ScanConvertVertexPolyhedra");
            // Make the scan conversion polyhedron data structure.
            poly = cartesianPolyhedron;
            // Scan convert the polyhedron.
            poly.scanConvert(std::back_inserter(indices), lattice);
          }
        }
        if (! vert.isConvex()) {
          {
            performance::Event event("MakeVertexPolyhedra");
            // Make the polyhedron containing the closest points of negative
            // distance.
            vert.buildCharacteristicPolyhedronNegative(&cartesianPolyhedron,
                                                       maximumDistance);
            // Convert the polyhedron to grid coordinates.
            lattice.convertLocationsToIndices
              (cartesianPolyhedron.getVerticesBeginning(),
               cartesianPolyhedron.getVerticesEnd());
          }
          {
            performance::Event event("ScanConvertVertexPolyhedra");
            // Make the scan conversion polyhedron data structure.
            poly = cartesianPolyhedron;
            // Scan convert the polyhedron.
            poly.scanConvert(std::back_inserter(indices), lattice);
          }
        }
        scanConversionCount += indices.size();
        performance::record("VertexScanConverted", indices.size());
        {
          performance::Event event("ScanConvertVertexPolyhedra");
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
        }

        {
          performance::Event event("VertexCpt");
          // Loop over the grids.
          for (std::size_t n = firstGrid; n != grids->size(); ++n) {
            // If the vertex could influence this grid.
            if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
              // Compute cpt for scan converted grid points.
#ifdef STLIB_PERFORMANCE
              countPair = (*grids)[n].computeClosestPointTransform
                (indices, cartesianPoints, vert, maximumDistance);
              performance::countVertexDistancesComputed += countPair.first;
              performance::countVertexDistancesSet += countPair.second;
              distanceCount += countPair.second;
#else
              distanceCount += (*grids)[n].computeClosestPointTransform
                (indices, cartesianPoints, vert, maximumDistance).second;
#endif
            }
          }
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Calculate the closest point transform with unsigned distance to this b-rep.
  /*!
    Calculate the unsigned distance, closest point, etc. to this b-rep for all
    the points in the grid.

    \param lattice is the lattice on which the grids lie.
    \param grids is the container of grids.  Each one holds the distance,
    closest point, etc. arrays.
    \param maximumDistance is the distance to calculate distance away from the
    surface.

    \return
    Return the number of points scan converted and the number of points
    for which the distance was set.
  */
  std::pair<std::size_t, std::size_t>
  computeClosestPointUnsigned(const Lattice& lattice,
                              std::vector<Grid>* grids,
                              const Number maximumDistance) const
  {
    assert(grids->size() > 0);

    std::size_t firstGrid;
    std::size_t scanConversionCount = 0;
    std::size_t distanceCount = 0;
    std::vector<IndexList> indices;
    std::vector<Point> cartesianPoints;
    BBox box;
    IndexBBox indexBox;
    Polyhedron poly;
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
    //
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
      // If there are no relevant grids, continue with the next face.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Get the i_th face.
      getFace(i, &face);
      // Make the polyhedron containing the closest points.
      face.buildCharacteristicPolyhedron(&cartesianPolyhedron, maximumDistance);
      // Convert the polyhedron to grid coordinates.
      lattice.convertLocationsToIndices
      (cartesianPolyhedron.getVerticesBeginning(),
       cartesianPolyhedron.getVerticesEnd());
      // Make the scan conversion polyhedron data structure.
      poly = cartesianPolyhedron;
      // Scan convert the polyhedron.
      indices.clear();
      poly.scanConvert(std::back_inserter(indices), lattice);
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

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // Get the edge.
      getEdgeUnsigned(iter, &edge);
      // Get a bounding box around the edge.
      getEdgeBBox(iter, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next edge.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Make the polyhedron containing the closest points.
      edge.buildCharacteristicPolyhedronUnsigned(&cartesianPolyhedron,
          maximumDistance);
      // Convert the polyhedron to grid coordinates.
      lattice.convertLocationsToIndices
      (cartesianPolyhedron.getVerticesBeginning(),
       cartesianPolyhedron.getVerticesEnd());
      // Make the scan conversion polyhedron data structure.
      poly = cartesianPolyhedron;
      // Scan convert the polyhedron.
      indices.clear();
      poly.scanConvert(std::back_inserter(indices), lattice);
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
        // If the edge could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (indices, cartesianPoints, edge, maximumDistance).second;
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //
    BBox bbox;

    // The closed index range of the grid.
    geom::BBox<std::ptrdiff_t, N> latticeIndexRange;
    {
      // The lower corner is zero.
      std::array<std::ptrdiff_t, N> p = {{}};
      latticeIndexRange.lower = p;
      for (std::size_t i = 0; i != p.size(); ++i) {
        p[i] = lattice.getExtents()[i] - 1;
      }
      latticeIndexRange.upper = p;
    }

    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // Don't compute the distance if none of the neighboring vertices
      // are known.
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
      indices.clear();

      // If this is an interior vertex.
      if (! vert.getEdgeDirections().empty()) {
        if (! vert.isConcave()) {
          // Make the polyhedron containing the closest points of positive
          // distance.
          vert.buildCharacteristicPolyhedronPositive(&cartesianPolyhedron,
              maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
          // Make the scan conversion polyhedron data structure.
          poly = cartesianPolyhedron;
          // Scan convert the polyhedron.
          poly.scanConvert(std::back_inserter(indices), lattice);
        }
        if (! vert.isConvex()) {
          // Make the polyhedron containing the closest points of negative
          // distance.
          vert.buildCharacteristicPolyhedronNegative(&cartesianPolyhedron,
              maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
          // Make the scan conversion polyhedron data structure.
          poly = cartesianPolyhedron;
          // Scan convert the polyhedron.
          poly.scanConvert(std::back_inserter(indices), lattice);
        }
      }
      else { // If this is a boundary vertex.
        // Make the bounding box containing the closest points.
        bbox.lower = vert.getLocation() - maximumDistance;
        bbox.upper = vert.getLocation() + maximumDistance;
        lattice.convertBBoxLocationsToIndices(&bbox);
        // Scan convert the bounding box.
        geom::scanConvert(std::back_inserter(indices), bbox,
                          latticeIndexRange);
      }

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
          // Compute the cpt for the scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (indices, cartesianPoints, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the characteristic polyhedra instead of polyhedron scan conversion.
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
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    performance::Scope _("computeClosestPointUsingBBox()");
#ifdef STLIB_PERFORMANCE
    std::pair<std::size_t, std::size_t> countPair;
#endif

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
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
      // If there are no relevant grids, continue with the next face.
      if (firstGrid == grids->size()) {
        continue;
      }
      {
        performance::Event event("MakeFacePolyhedra");
        // Get the i_th face.
        getFace(i, &face);
        // Make the polyhedron containing the closest points.
        face.buildCharacteristicPolyhedron(&cartesianPolyhedron,
                                           maximumDistance);
        // Convert the polyhedron to grid coordinates.
        lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
      }
      // Make a bounding box around the polyhedra.
      characteristicBox =
        geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                 cartesianPolyhedron.getVerticesEnd());
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
#ifdef STLIB_PERFORMANCE
      performance::record("FaceScanConverted", indexRange.content());
#endif
      {
        performance::Event event("FaceCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the face could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for the grid points in the
            // index range.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, face, maximumDistance);
            performance::record("FaceDistancesComputed", countPair.first);
            performance::record("FaceDistancesSet", countPair.second);
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, face, maximumDistance).second;
#endif
          }
        }
      }
    }

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // Only compute the distance if the adjacent faces are known and the
      // surface is either concave or convex at the edge.
      if (! getEdge(iter, &edge) || edge.getSignOfDistance() == 0) {
        continue;
      }
      // Get a bounding box around the edge.
      getEdgeBBox(iter, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next edge.
      if (firstGrid == grids->size()) {
        continue;
      }
      {
        performance::Event event("MakeEdgePolyhedra");
        // Make the polyhedron containing the closest points.
        edge.buildCharacteristicPolyhedron(&cartesianPolyhedron,
                                           maximumDistance);
        // Convert the polyhedron to grid coordinates.
        lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
      }
      // Make a bounding box around the polyhedra.
      characteristicBox =
        geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                 cartesianPolyhedron.getVerticesEnd());
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
#ifdef STLIB_PERFORMANCE
      performance::record("EdgeScanConverted", indexRange.content());
#endif
      {
        performance::Event event("EdgeCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the edge could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for scan converted grid pts.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, edge, maximumDistance);
            performance::countEdgeDistancesComputed += countPair.first;
            performance::countEdgeDistancesSet += countPair.second;
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, edge, maximumDistance).second;
#endif
          }
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //
    BBox positiveBox, negativeBox;
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // Don't compute the distance if the vertex is on the boundary.
      if (! getVertex(i, &vert)) {
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
      if (! vert.isConcave()) {
        {
          performance::Event event("MakeVertexPolyhedra");
          // Make the polyhedron containing the closest points of positive
          // distance.
          vert.buildCharacteristicPolyhedronPositive(&cartesianPolyhedron,
                                                     maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
            (cartesianPolyhedron.getVerticesBeginning(),
             cartesianPolyhedron.getVerticesEnd());
        }
        // Make a bounding box around the polyhedra.
        positiveBox =
          geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                   cartesianPolyhedron.getVerticesEnd());
      }
      if (! vert.isConvex()) {
        {
          performance::Event event("MakeVertexPolyhedra");
          // Make the polyhedron containing the closest points of negative
          // distance.
          vert.buildCharacteristicPolyhedronNegative(&cartesianPolyhedron,
                                                     maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
            (cartesianPolyhedron.getVerticesBeginning(),
             cartesianPolyhedron.getVerticesEnd());
        }
        // Make a bounding box around the polyhedra.
        negativeBox =
          geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                   cartesianPolyhedron.getVerticesEnd());
      }
      if (! vert.isConcave() && ! vert.isConvex()) {
        characteristicBox = positiveBox;
        characteristicBox += negativeBox.lower;
        characteristicBox += negativeBox.upper;
      }
      else if (! vert.isConcave()) {
        characteristicBox = positiveBox;
      }
      else if (! vert.isConvex()) {
        characteristicBox = negativeBox;
      }
      else {
        continue;
      }
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
#ifdef STLIB_PERFORMANCE
      performance::record("VertexScanConverted", indexRange.content());
#endif

      {
        performance::Event event("VertexCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the vertex could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute cpt for scan converted grid points.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, vert, maximumDistance);
            performance::record("VertexDistancesComputed", countPair.first);
            performance::record("VertexDistancesSet", countPair.second);
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, vert, maximumDistance).second;
#endif
          }
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the characteristic polyhedra instead of polyhedron scan conversion.
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
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
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
      // If there are no relevant grids, continue with the next face.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Get the i_th face.
      getFace(i, &face);
      // Make the polyhedron containing the closest points.
      face.buildCharacteristicPolyhedron(&cartesianPolyhedron, maximumDistance);
      // Convert the polyhedron to grid coordinates.
      lattice.convertLocationsToIndices
      (cartesianPolyhedron.getVerticesBeginning(),
       cartesianPolyhedron.getVerticesEnd());
      // Make a bounding box around the polyhedra.
      characteristicBox =
        geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                 cartesianPolyhedron.getVerticesEnd());
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
          // Compute closest points and distance for the grid points in the
          // index range.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // Get the edge.
      getEdgeUnsigned(iter, &edge);
      // Get a bounding box around the edge.
      getEdgeBBox(iter, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next edge.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Make the polyhedron containing the closest points.
      edge.buildCharacteristicPolyhedronUnsigned(&cartesianPolyhedron,
          maximumDistance);
      // Convert the polyhedron to grid coordinates.
      lattice.convertLocationsToIndices
      (cartesianPolyhedron.getVerticesBeginning(),
       cartesianPolyhedron.getVerticesEnd());
      // Make a bounding box around the polyhedra.
      characteristicBox =
        geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                 cartesianPolyhedron.getVerticesEnd());
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
        // If the edge could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, edge, maximumDistance).second;
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //

    BBox positiveBox, negativeBox;
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // Don't compute the distance if none of the neighboring vertices
      // are known.
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

      // If this is an interior vertex.
      if (! vert.getEdgeDirections().empty()) {
        if (! vert.isConcave()) {
          // Make the polyhedron containing the closest points of positive
          // distance.
          vert.buildCharacteristicPolyhedronPositive(&cartesianPolyhedron,
              maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
          // Make a bounding box around the polyhedra.
          positiveBox =
            geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                     cartesianPolyhedron.getVerticesEnd());
        }
        if (! vert.isConvex()) {
          // Make the polyhedron containing the closest points of negative
          // distance.
          vert.buildCharacteristicPolyhedronNegative(&cartesianPolyhedron,
              maximumDistance);
          // Convert the polyhedron to grid coordinates.
          lattice.convertLocationsToIndices
          (cartesianPolyhedron.getVerticesBeginning(),
           cartesianPolyhedron.getVerticesEnd());
          // Make a bounding box around the polyhedra.
          negativeBox =
            geom::specificBBox<BBox>(cartesianPolyhedron.getVerticesBeginning(),
                                     cartesianPolyhedron.getVerticesEnd());
        }
        if (! vert.isConcave() && ! vert.isConvex()) {
          characteristicBox = positiveBox;
          characteristicBox += negativeBox.lower;
          characteristicBox += negativeBox.upper;
        }
        else if (! vert.isConcave()) {
          characteristicBox = positiveBox;
        }
        else if (! vert.isConvex()) {
          characteristicBox = negativeBox;
        }
        else {
          continue;
        }
      }
      else { // If this is a boundary vertex.
        // Start with the box of radius maximumDistance around the vertex.
        // Then convert to grid coordinates.
        characteristicBox = box;
        lattice.convertBBoxLocationsToIndices(&characteristicBox);
      }

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
          // Compute the cpt for the scan converted grid points.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, vert, maximumDistance).second;
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount, distanceCount);
  }

  //! Use bounding boxes around the primitives instead of polyhedron scan conversion.
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
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    performance::Scope _("computeClosestPointUsingBruteForce()");
#ifdef STLIB_PERFORMANCE
    std::pair<std::size_t, std::size_t> countPair;
#endif

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n],
                                            &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
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
      // If there are no relevant grids, continue with the next face.
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
#ifdef STLIB_PERFORMANCE
      performance::record("FaceScanConverted", indexRange.content());
#endif
      {
        performance::Event event("FaceCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the face could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for the grid points in the
            // index range.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, face, maximumDistance);
            performance::record("FaceDistancesComputed", countPair.first);
            performance::record("FaceDistancesSet", countPair.second);
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, face, maximumDistance).second;
#endif
          }
        }
      }
    }

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // Only compute the distance if the adjacent faces are known and the
      // surface is either concave or convex at the edge.
      if (! getEdge(iter, &edge) || edge.getSignOfDistance() == 0) {
        continue;
      }
      // Get a bounding box around the edge.
      getEdgeBBox(iter, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next edge.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Convert the edge bounding box to index coordinates.
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
#ifdef STLIB_PERFORMANCE
      performance::record("EdgeScanConverted", indexRange.content());
#endif
      {
        performance::Event event("EdgeCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the edge could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute closest points and distance for scan converted grid pts.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, edge, maximumDistance);
            performance::record("EdgeDistancesComputed", countPair.first);
            performance::record("EdgeDistancesSet", countPair.second);
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, edge, maximumDistance).second;
#endif
          }
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //
    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // Don't compute the distance if the vertex is on the boundary.
      if (! getVertex(i, &vert)) {
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
#ifdef STLIB_PERFORMANCE
      performance::record("VertexScanConverted", indexRange.content());
#endif

      {
        performance::Event event("VertexCpt");
        // Loop over the grids.
        for (std::size_t n = firstGrid; n != grids->size(); ++n) {
          // If the vertex could influence this grid.
          if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
            // Compute cpt for scan converted grid points.
#ifdef STLIB_PERFORMANCE
            countPair = (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, vert, maximumDistance);
            performance::countVertexDistancesComputed += countPair.first;
            performance::countVertexDistancesSet += countPair.second;
            distanceCount += countPair.second;
#else
            distanceCount += (*grids)[n].computeClosestPointTransform
              (lattice, indexRange, vert, maximumDistance).second;
#endif
          }
        }
      }
    }

    return std::pair<std::size_t, std::size_t>(scanConversionCount,
                                               distanceCount);
  }

  //! Use bounding boxes around the primitives instead of polyhedron scan conversion.
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
    geom::IndexedEdgePolyhedron<Number> cartesianPolyhedron;

    // Store index bounding boxes for each grid.
    std::vector<IndexBBox> gridIndexBBoxes(grids->size());
    computeIndexBoundingBoxes(*grids, &gridIndexBBoxes);

    // Compute Cartesian bounding boxes around each grid.
    std::vector<BBox> gridDomains(grids->size());
    for (std::size_t n = 0; n != grids->size(); ++n) {
      lattice.convertBBoxIndicesToLocations(gridIndexBBoxes[n], &gridDomains[n]);
    }

    //
    // Find the closest points and distance for the faces.
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
      // If there are no relevant grids, continue with the next face.
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
          // Compute closest points and distance for the grid points in the
          // index range.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, face, maximumDistance).second;
        }
      }
    }

    //
    // Find the closest points and distance for the edges.
    //
    EdgeDistance edge;
    const EdgeIterator edgesEnd = getFacesEnd();
    for (EdgeIterator iter = getFacesBeginning(); iter != edgesEnd; ++iter) {
      // Get the edge.
      getEdgeUnsigned(iter, &edge);
      // Get a bounding box around the edge.
      getEdgeBBox(iter, maximumDistance, &box);
      // Find the first relevant grid.
      firstGrid = 0;
      for (; firstGrid != grids->size(); ++firstGrid) {
        if (geom::doOverlap(gridDomains[firstGrid], box)) {
          break;
        }
      }
      // If there are no relevant grids, continue with the next edge.
      if (firstGrid == grids->size()) {
        continue;
      }
      // Convert the edge bounding box to index coordinates.
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
        // If the edge could influence this grid.
        if (geom::doOverlap(gridIndexBBoxes[n], indexBox)) {
          // Compute closest points and distance for scan converted grid pts.
          distanceCount += (*grids)[n].computeClosestPointTransformUnsigned
                           (lattice, indexRange, edge, maximumDistance).second;
        }
      }
    }

    //
    // Find the closest points and distance for the vertices.
    //

    VertexDistance vert;
    for (std::size_t i = 0; i != Base::vertices.size(); ++i) {
      // Don't compute the distance if none of the neighboring vertices
      // are known.
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
          // Compute the cpt for the scan converted grid points.
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
    out << _faceNormals << _vertexNormals << _faceIdentifiers;
  }

  //@}

private:

  //
  // Accessors
  //

  // Make a Face out of the face specified by the index.
  void
  getFace(const std::size_t index, FaceDistance* face) const
  {
    face->make(getSimplexVertex(index, 0),
               getSimplexVertex(index, 1),
               getSimplexVertex(index, 2),
               _faceNormals[index],
               _faceIdentifiers[index]);
  }

  // Make a bounding box around the face specified by the index.
  // Enlarge the bounding box by the maximumDistance.
  void
  getFaceBBox(const std::size_t index, const Number maximumDistance, BBox* box)
  const
  {
    *box = geom::specificBBox<BBox>(std::array<Point, 3>{{
        getSimplexVertex(index, 0),
          getSimplexVertex(index, 1),
          getSimplexVertex(index, 2)}});
    box->lower = box->lower - maximumDistance;
    box->upper = box->upper + maximumDistance;
  }

  // Make the edge.
  // Return true if this is an interior edge.
  bool
  getEdge(const EdgeIterator iter, EdgeDistance* edge) const
  {
    // We don't compute signed distance for boundary edges.
    if (isOnBoundary(iter)) {
      return false;
    }
    // The simplex index.
    const std::size_t si = iter->first;
    // The local face index.
    const std::size_t i = iter->second;
    // Make the edge.
    edge->make(getSimplexVertex(si, (i + 1) % N), // source
               getSimplexVertex(si, (i + 2) % N), // target
               _faceNormals[si], // left normal
               _faceNormals[Base::adjacent[si][i]], // right normal
               _faceIdentifiers[si]); // face identifier
    return true;
  }

  // Make the edge.
  // The edge may be in the interior or on the boundary.
  void
  getEdgeUnsigned(const EdgeIterator iter, EdgeDistance* edge) const
  {
    // The simplex index.
    const std::size_t si = iter->first;
    // The local face index.
    const std::size_t i = iter->second;
    // If this is an interior edge.
    if (Base::adjacent[si][i] != std::numeric_limits<std::size_t>::max()) {
      // Make the edge.
      edge->make(getSimplexVertex(si, (i + 1) % N), // source
                 getSimplexVertex(si, (i + 2) % N), // target
                 _faceNormals[si], // left normal
                 _faceNormals[Base::adjacent[si][i]], // right normal
                 _faceIdentifiers[si]); // face identifier
    }
    // Else, this is a boundary edge.
    else {
      // Make the edge.
      edge->make(getSimplexVertex(si, (i + 1) % N), // source
                 getSimplexVertex(si, (i + 2) % N), // target
                 _faceNormals[si], // left normal
                 // null value for the right normal
                 ext::filled_array<Point>(0.0),
                 _faceIdentifiers[si]); // face identifier
    }
  }

  // Make a bounding box around the edge specified by the index.
  // Enlarge the bounding box by the maximumDistance.
  void
  getEdgeBBox(const EdgeIterator iter, const Number maximumDistance,
              BBox* box) const
  {
    // The simplex index.
    const std::size_t si = iter->first;
    // The local face index.
    const std::size_t i = iter->second;
    *box = geom::specificBBox<BBox>(std::array<Point, 2>{{
        getSimplexVertex(si, (i + 1) % N), // source
          getSimplexVertex(si, (i + 2) % N)}}); // target
    box->lower = box->lower - maximumDistance;
    box->upper = box->upper + maximumDistance;
  }

  // Make the vertex.
  // Return true iff the vertex is not on the boundary.
  bool
  getVertex(const std::size_t vertexIndex, VertexDistance* vert) const
  {
    // Boundary vertices are not used in computing signed distance.
    if (isVertexOnBoundary(vertexIndex)) {
      return false;
    }

    // The number of incident triangle faces.
    const std::size_t incidentSize = Base::incident.size(vertexIndex);

    std::size_t i;
    // The neighboring vertices.
    std::vector<Point> neighbors(incidentSize);
    // We can get these in any order.
    for (std::size_t n = 0; n != incidentSize; ++n) {
      // The face as an indexed simplex.
      const IndexedSimplex& is =
        Base::indexedSimplices[Base::incident(vertexIndex, n)];
      // The local index of this vertex in the indexed simplex.
      i = ext::index(is, vertexIndex);
      // Get the vertex in the positive direction from this one.
      neighbors[n] = Base::vertices[is[(i + 1) % N]];
    }

    // CONTINUE: I don't think they need to be in any particular order.
    // The normals of the adjacent faces.
    std::vector<Point> normals(incidentSize);
    // We need to walk around the vertex in the positive direction.
    // The current simplex index.
    std::size_t si = Base::incident(vertexIndex, 0);
    for (std::size_t n = 0; n != incidentSize; ++n) {
      normals[n] = _faceNormals[si];
      // The local index of this vertex in the indexed simplex.
      i = ext::index(Base::indexedSimplices[si], vertexIndex);
      // Move to the next face.
      si = Base::adjacent[si][(i + 1) % N];
    }

#ifdef STLIB_DEBUG
    // We should have come back to the starting face.
    assert(si == Base::incident(vertexIndex, 0));
#endif

    // Make the vertex.
    vert->make(Base::vertices[vertexIndex], _vertexNormals[vertexIndex],
               neighbors, normals, _faceIdentifiers[si]);

    return true;
  }

  // Make the vertex for computing unsigned distance.
  // Return true if any of the neigboring vertices are known.
  // Otherwise return false.
  bool
  getVertexUnsigned(const std::size_t index, VertexDistance* vert) const
  {
    // If there are no incident faces.
    if (Base::incident.empty(index)) {
      return false;
    }

    // If this vertex is not on the boundary of the surface, we can use the
    // same function as is used in computing unsigned distance.
    if (! getVertex(index, vert)) {
      // Otherwise, make the vertex without using any b-rep information.
      vert->make(Base::vertices[index],
                 _faceIdentifiers[Base::incident(index, 0)]);
    }

    return true;
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
};


//
// Private Member Functions
//

//
// This grid function is defined here because it uses BRep.
//

// CONTINUE
#if 0
template<typename T>
inline
void
Grid<3, T>::
flood_fill(Number far_away,
           const std::vector<Point>& vertices,
           const std::vector<SizeList>& faces)
{
  // First try to determine the sign of the distance from the known distances
  // and then flood fill.
  if (flood_fill(far_away)) {
    return;
  }
  // If the above did not succeed then there are no known distances.

  // Ensure that the mesh is not degenerate.
  assert(vertices.size() != 0 && faces.size() != 0);

  // Find the Cartesian location of one of the grid points.
  Point location(0, 0, 0);
  index_to_location(location);

  // We will find the closest point on the mesh to the grid location.

  // Compute the distances to the vertices.
  std::vector<Number> vertex_distance(vertices.size());
  for (std::size_t i = 0; i != vertices.size(); ++i) {
    vertex_distance[i] = euclideanDistance(location, vertices[i]);
  }

  // Find the vertex that is closest to the grid location.
  // We use this to determine an upper bound on the distance.
  const Number upper_bound_distance = *std::min_element(vertex_distance.begin(),
                                      vertex_distance.end());

  // Determine the faces that are relevant.
  std::vector<SizeList> close_faces;
  {
    Number max_edge_length, min_vertex_distance;
    for (std::size_t i = 0; i != faces.size(); ++i) {
      max_edge_length =
        ads::max(euclideanDistance(vertices[faces[i][0]],
                                   vertices[faces[i][1]]),
                 euclideanDistance(vertices[faces[i][1]],
                                   vertices[faces[i][2]]),
                 euclideanDistance(vertices[faces[i][2]],
                                   vertices[faces[i][0]]));
      min_vertex_distance = ads::min(vertex_distance[faces[i][0]],
                                     vertex_distance[faces[i][1]],
                                     vertex_distance[faces[i][2]]);
      if (min_vertex_distance <= upper_bound_distance + max_edge_length) {
        close_faces.push_back(faces[i]);
      }
    }
  }

  // Make a set of the vertex indices that comprise the close faces.
  std::set<std::size_t> index_set;
  {
    for (std::size_t i = 0; i != close_faces.size(); ++i) {
      index_set.insert(close_faces[i][0]);
      index_set.insert(close_faces[i][1]);
      index_set.insert(close_faces[i][2]);
    }
  }
  std::vector<std::size_t> close_vertex_indices(index_set.begin(),
      index_set.end());

  // Make an array of the close vertices.
  std::vector<Point> close_vertices(close_vertex_indices.size());
  {
    for (std::size_t i = 0; i != close_vertex_indices.size(); ++i) {
      close_vertices[i] = vertices[close_vertex_indices[i]];
    }
  }

  // Adjust the indices of the close faces.
  {
    for (std::size_t i = 0; i != close_faces.size(); ++i) {
      for (std::size_t j = 0; j != N; ++j) {
        close_faces[i][j] =
          std::lower_bound(close_vertex_indices.begin(),
                           close_vertex_indices.end(), close_faces[i][j]) -
          close_vertex_indices.begin();
      }
    }
  }

  // Make a b-rep from the close faces.
  BRep<N, Number> brep;
  brep.make(close_vertices.begin(), close_vertices.end(),
            close_faces.begin(), close_faces.end());

  // Make one grid with a single point.
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
  brep.computeClosestPoint(grids, upper_bound_distance * 1.1);
  const IndexList zero = {{}};
  Number d = dist(zero);

  // Set the distance to +- far_away.
  assert(d != std::numeric_limits<Number>::max());
  int sign_distance = (d >= 0 ? 1 : -1);
  distance() = sign_distance * far_away;
}
#endif

} // namespace cpt
}
