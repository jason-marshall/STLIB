// -*- C++ -*-

#include "stlib/geom/mesh/iss/subdivide.h"
#include "stlib/geom/mesh/iss/quality.h"

using namespace stlib;

void
test3()
{
  const std::size_t N = 3;
  typedef geom::IndSimpSetIncAdj<N, N> ISS;
  typedef ISS::Vertex Vertex;
  typedef ISS::IndexedSimplex IndexedSimplex;

  std::array<ISS, 7> mesh;
  {
    Vertex vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{0, 1, 0}},
      {{0, 0, 1}}
    };
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2, 3}}};
    const std::size_t numSimplices =
      sizeof(indexedSimplices) / sizeof(IndexedSimplex);
    build(&mesh[0], numVertices, vertices, numSimplices, indexedSimplices);
  }

  for (std::size_t i = 1; i != mesh.size(); ++i) {
    const std::size_t n = 1 << (3 * i);
    subdivide(mesh[i - 1], &mesh[i]);
    assert(mesh[i].indexedSimplices.size() == n);
    assert(std::abs(computeContent(mesh[0]) - computeContent(mesh[i])) <
           n * std::numeric_limits<double>::epsilon());
    std::cout << "\nLevels of refinement = " << i << '\n';
    printQualityStatistics(std::cout, mesh[i]);
  }
}

int
main()
{
  test3();

  return 0;
}
