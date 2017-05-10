// -*- C++ -*-

#include "stlib/geom/mesh/iss/lor.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/equality.h"
#include "stlib/geom/mesh/iss/quality.h"

using namespace stlib;

template<typename _Integer>
void
iss()
{
  typedef geom::IndSimpSet<3, 3> Mesh;

  //
  // Data for an octahedron
  //
  const std::size_t numVertices = 7;
  double vertices[] = { 0, 0, 0,
                        1, 0, 0,
                        -1, 0, 0,
                        0, 1, 0,
                        0, -1, 0,
                        0, 0, 1,
                        0, 0, -1
                      };
  const std::size_t numTets = 8;
  std::size_t tets[] = { 0, 1, 3, 5,
                         0, 3, 2, 5,
                         0, 2, 4, 5,
                         0, 4, 1, 5,
                         0, 3, 1, 6,
                         0, 2, 3, 6,
                         0, 4, 2, 6,
                         0, 1, 4, 6
                       };
  // Construct a mesh from vertices and tetrahedra.
  Mesh mesh;
  build(&mesh, numVertices, vertices, numTets, tets);

  // Order the vertices and simplices.
  Mesh ordered(mesh);
  geom::mortonOrder<_Integer>(&ordered);

  // Check the content.
  const double a = geom::computeContent(mesh);
  const double b = geom::computeContent(ordered);
  assert(std::abs(a - b) < 100 * a * std::numeric_limits<double>::epsilon());
}

template<typename _Integer>
void
issia()
{
  typedef geom::IndSimpSetIncAdj<3, 3> Mesh;

  //
  // Data for an octahedron
  //
  const std::size_t numVertices = 7;
  double vertices[] = { 0, 0, 0,
                        1, 0, 0,
                        -1, 0, 0,
                        0, 1, 0,
                        0, -1, 0,
                        0, 0, 1,
                        0, 0, -1
                      };
  const std::size_t numTets = 8;
  std::size_t tets[] = { 0, 1, 3, 5,
                         0, 3, 2, 5,
                         0, 2, 4, 5,
                         0, 4, 1, 5,
                         0, 3, 1, 6,
                         0, 2, 3, 6,
                         0, 4, 2, 6,
                         0, 1, 4, 6
                       };
  // Construct a mesh from vertices and tetrahedra.
  Mesh mesh;
  build(&mesh, numVertices, vertices, numTets, tets);

  // Order the vertices and simplices.
  Mesh ordered(mesh);
  geom::mortonOrder<_Integer>(&ordered);

  // Check the content.
  const double a = geom::computeContent(mesh);
  const double b = geom::computeContent(ordered);
  assert(std::abs(a - b) < 100 * a * std::numeric_limits<double>::epsilon());

  // Check the incidence and adjacencies.
  {
    Mesh reconstructed = geom::IndSimpSet<3, 3>(ordered);
    assert(reconstructed == ordered);
  }
}

int
main()
{
  iss<unsigned char>();
  issia<unsigned char>();
  iss<unsigned short>();
  issia<unsigned short>();
  iss<unsigned>();
  issia<unsigned>();
  iss<std::size_t>();
  issia<std::size_t>();

  return 0;
}
