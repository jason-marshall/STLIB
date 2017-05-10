// -*- C++ -*-

#include "stlib/geom/mesh/iss/closestSimplex.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/orq/CellArrayStatic.h"

using namespace stlib;

// 1-D.
template<typename _T>
void
test1()
{
  const std::size_t N = 1;
  typedef std::array<_T, N> Point;
  typedef geom::IndSimpSet<N, N, _T> Mesh;
  typedef typename Mesh::IndexedSimplex IndexedSimplex;

  std::vector<Point> points;
  Mesh mesh;
  std::vector<std::size_t> indices;

  // No points and empty mesh.
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.empty());

  // Mesh with one simplex.
  std::vector<Point> vertices;
  vertices.push_back(Point{{0}});
  vertices.push_back(Point{{1}});
  std::vector<IndexedSimplex> indexedSimplices;
  indexedSimplices.push_back(IndexedSimplex{{0, 1}});
  build(&mesh, vertices, indexedSimplices);

  // No points and one simplex.
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.empty());

  // Point inside.
  points.push_back(Point{{0.5}});
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Point above.
  points[0] = Point{{2}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Point below.
  points[0] = Point{{-2}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Add another simplex.
  vertices.push_back(Point{{2}});
  indexedSimplices.push_back(IndexedSimplex{{1, 2}});
  build(&mesh, vertices, indexedSimplices);

  // Inside first.
  points[0] = Point{{0.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Inside second.
  points[0] = Point{{1.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 1);

  // Above.
  points[0] = Point{{3}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 1);

  // Below.
  points[0] = Point{{-3}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Add another simplex.
  vertices.push_back(Point{{3}});
  indexedSimplices.push_back(IndexedSimplex{{2, 3}});
  build(&mesh, vertices, indexedSimplices);

  // Various tests.

  points[0] = Point{{-0.5}};
  points.push_back(Point{{0.5}});
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 2);
  assert(indices[0] == 0);
  assert(indices[1] == 0);

  points[0] = Point{{0.5}};
  points[1] = Point{{1.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 2);
  assert(indices[0] == 0);
  assert(indices[1] == 1);

  points[0] = Point{{1.5}};
  points[1] = Point{{2.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 2);
  assert(indices[0] == 1);
  assert(indices[1] == 2);

  points[0] = Point{{2.5}};
  points[1] = Point{{3.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 2);
  assert(indices[0] == 2);
  assert(indices[1] == 2);
}

// 2-D.
template<typename _T>
void
test2()
{
  const std::size_t N = 2;
  typedef std::array<_T, N> Point;
  typedef geom::IndSimpSet<N, N, _T> Mesh;
  typedef typename Mesh::IndexedSimplex IndexedSimplex;

  std::vector<Point> points;
  Mesh mesh;
  std::vector<std::size_t> indices;

  // No points and empty mesh.
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.empty());

  // Mesh with one simplex.
  std::vector<Point> vertices;
  vertices.push_back(Point{{0, 0}});
  vertices.push_back(Point{{1, 0}});
  vertices.push_back(Point{{0, 1}});
  std::vector<IndexedSimplex> indexedSimplices;
  indexedSimplices.push_back(IndexedSimplex{{0, 1, 2}});
  build(&mesh, vertices, indexedSimplices);

  // No points and one simplex.
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.empty());

  // Point inside.
  points.push_back(Point{{0.25, 0.25}});
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Point outside.
  points[0] = Point{{-1, -1}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Very close.
  points[0] = Point{{-1e-10, -1e-10}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  // Add another simplex.
  vertices.push_back(Point{{1, 1}});
  indexedSimplices.push_back(IndexedSimplex{{1, 3, 2}});
  build(&mesh, vertices, indexedSimplices);

  // Various tests.

  points[0] = Point{{-1, -1}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  points[0] = Point{{0.25, 0.25}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  points[0] = Point{{0.75, 0.75}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 1);

  points[0] = Point{{2, 2}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 1);

  points[0] = Point{{0.5, -1}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 0);

  points[0] = Point{{0.5, 1.5}};
  geom::closestSimplex<geom::CellArrayStatic>(mesh, points, &indices);
  assert(indices.size() == 1);
  assert(indices[0] == 1);
}

int
main()
{
  test1<float>();
  test1<double>();
  test2<float>();
  test2<double>();

  return 0;
}
