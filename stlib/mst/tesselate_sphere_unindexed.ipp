// -*- C++ -*-

#if !defined(__tesselate_sphere_unindexed_ipp__)
#error This file is an implementation detail of tesselate_sphere_unindexed.
#endif

namespace stlib
{
namespace mst
{


template<typename T>
inline
void
makeOctahedron
(std::vector<std::array<std::array<T, 3>, 3> >& mesh)
{
  typedef T Number;
  typedef std::array<Number, 3> Point;

  if (mesh.size() != 8) {
    mesh.resize(8);
  }

  const Point xp = {{1, 0, 0}};
  const Point xm = {{ -1, 0, 0}};
  const Point yp = {{0, 1, 0}};
  const Point ym = {{0, -1, 0}};
  const Point zp = {{0, 0, 1}};
  const Point zm = {{0, 0, -1}};

  mesh[0][0] = xp;
  mesh[0][1] = zp;
  mesh[0][2] = yp;

  mesh[1][0] = yp;
  mesh[1][1] = zp;
  mesh[1][2] = xm;

  mesh[2][0] = xm;
  mesh[2][1] = zp;
  mesh[2][2] = ym;

  mesh[3][0] = ym;
  mesh[3][1] = zp;
  mesh[3][2] = xp;

  mesh[4][0] = xp;
  mesh[4][1] = yp;
  mesh[4][2] = zm;

  mesh[5][0] = yp;
  mesh[5][1] = xm;
  mesh[5][2] = zm;

  mesh[6][0] = xm;
  mesh[6][1] = ym;
  mesh[6][2] = zm;

  mesh[7][0] = ym;
  mesh[7][1] = xp;
  mesh[7][2] = zm;
}



template<typename T>
inline
void
subdivide(std::vector<std::array<std::array<T, 3>, 3> >& mesh)
{
  typedef T Number;
  typedef std::array<Number, 3> Point;
  typedef std::array<Point, 3> Triangle;
  typedef std::vector<Triangle> Mesh;

  // Make a new mesh that has 4 times as many triangles.
  Mesh m(4 * mesh.size());

  Point a, b, c;
  // For each triangle in the old mesh.
  for (std::size_t i = 0; i != mesh.size(); ++i) {

    // Subdivide the triangle.  Normalize each point to lie on the unit sphere.

    /*
            2
            /\
           /  \
         c/____\ b
         /\    /\
        /  \  /  \
       /____\/____\
      0     a      1
    */
    a = mesh[i][0];
    a += mesh[i][1];
    a *= 0.5;
    ext::normalize(&a);

    b = mesh[i][1];
    b += mesh[i][2];
    b *= 0.5;
    ext::normalize(&b);

    c = mesh[i][2];
    c += mesh[i][0];
    c *= 0.5;
    ext::normalize(&c);

    m[4 * i][0] = mesh[i][0];
    m[4 * i][1] = a;
    m[4 * i][2] = c;

    m[4 * i + 1][0] = a;
    m[4 * i + 1][1] = mesh[i][1];
    m[4 * i + 1][2] = b;

    m[4 * i + 2][0] = a;
    m[4 * i + 2][1] = b;
    m[4 * i + 2][2] = c;

    m[4 * i + 3][0] = c;
    m[4 * i + 3][1] = b;
    m[4 * i + 3][2] = mesh[i][2];
  }

  // Swap the old and new meshes.
  mesh.swap(m);
}


template<typename T>
inline
T
computeMaxEdgeLength
(const std::vector<std::array<std::array<T, 3>, 3> >& mesh)
{
  typedef T Number;

  Number maximumLength = std::numeric_limits<Number>::max();
  Number x;
  for (std::size_t m = 0; m != mesh.size(); ++m) {
    for (std::size_t n = 0; n != 3; ++n) {
      x = geom::computeDistance(mesh[m][n], mesh[m][(n + 1) % 3]);
      if (x < maximumLength) {
        maximumLength = x;
      }
    }
  }
  return maximumLength;
}



template<typename T, typename PointOutIter>
inline
void
tesselateUnitSphereUnindexed(const T maxEdgeLength, PointOutIter points)
{
  typedef T Number;
  typedef std::array<Number, 3> Point;
  typedef std::array<Point, 3> Triangle;
  typedef std::vector<Triangle> Mesh;

  assert(maxEdgeLength > 0);

  // The vector of meshes that we have obtained through subdivision.
  // We make this static to avoid recomputing the meshes.
  static std::vector<Mesh> meshes;
  static std::vector<Number> maxEdgeLengths;

  // If we have not yet computed the coarsest mesh.
  if (meshes.size() == 0) {
    // Start with an octahedron.
    Mesh mesh(8);
    makeOctahedron(mesh);
    meshes.push_back(mesh);
    maxEdgeLengths.push_back(computeMaxEdgeLength(mesh));
  }

  // If the current meshes are too coarse, compute finer ones.
  while (maxEdgeLength < maxEdgeLengths.back()) {
    // Copy the current finest mesh.
    meshes.push_back(meshes.back());
    // Subdivide it.
    subdivide(meshes.back());
    // Compute the maximum edge length for the refined mesh.
    maxEdgeLengths.push_back(computeMaxEdgeLength(meshes.back()));
  }

  // Determine the appropriate mesh.
  std::size_t index = 0;
  while (maxEdgeLengths[index] > maxEdgeLength) {
    ++index;
  }
  const Mesh& mesh = meshes[index];

  // Write the points that define the triangulation.
  for (std::size_t m = 0; m != mesh.size(); ++m) {
    for (std::size_t n = 0; n != 3; ++n) {
      *points++ = mesh[m][n];
    }
  }
}




template < typename T, typename PointInIter,
           typename NumberInIter, typename PointOutIter >
inline
void
tesselateAllSurfacesUnindexed(PointInIter centersBegin, PointInIter centersEnd,
                              NumberInIter radiiBegin, NumberInIter radiiEnd,
                              const T maxEdgeLength, PointOutIter points)
{
  typedef T Number;
  typedef std::array<Number, 3> Point;

  assert(maxEdgeLength > 0);

  // For each sphere.
  for (; centersBegin != centersEnd; ++centersBegin, ++radiiBegin) {
    // Check that the two ranges are the same size.
    assert(radiiBegin != radiiEnd);
    // The center.
    const Point center = *centersBegin;
    // The radius.
    const Number radius = *radiiBegin;
    // The triangulation of this sphere.
    std::vector<Point> oneSurface;
    // Compute the triangulation of the unit sphere.
    tesselateUnitSphereUnindexed(maxEdgeLength / radius,
                                 std::back_inserter(oneSurface));
    // Transform the unit sphere to match this sphere.
    for (typename std::vector<Point>::iterator i = oneSurface.begin();
         i != oneSurface.end(); ++i) {
      *i *= radius;
      *i += center;
    }
    // Add the triangulation of this sphere to the output.
    for (typename std::vector<Point>::const_iterator i = oneSurface.begin();
         i != oneSurface.end(); ++i) {
      *points++ = *i;
    }
  }
  // Check that the two ranges are the same size.
  assert(radiiBegin == radiiEnd);
}

} // namespace mst
}
