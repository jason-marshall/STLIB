// -*- C++ -*-

#include "stlib/geom/mesh/iss/geometry.h"
#include "stlib/geom/mesh/iss/build.h"

using namespace stlib;

int
main()
{
  // 1-D.
  {
    const std::size_t N = 1;
    typedef geom::IndSimpSetIncAdj<N, N> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;

    Vertex vertices[] = {{{0}}, {{1}}, {{2}}};
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    IndexedSimplex indexedSimplices[] = {{{0, 1}}, {{1, 2}}};
    const std::size_t numSimplices =
      sizeof(indexedSimplices) / sizeof(IndexedSimplex);

    ISS mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(computeAngle(mesh, 0) == 1);
    assert(computeAngle(mesh, 1) == 2);
    assert(computeAngle(mesh, 2) == 1);
  }

  // 2-D.
  {
    const std::size_t N = 2;
    typedef geom::IndSimpSetIncAdj<N, N> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;

    Vertex vertices[] = {{{0, 0}},
      {{1, 0}},
      {{0, 1}},
      {{ -1, 0}},
      {{0, -1}}
    };
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    IndexedSimplex indexedSimplices[] = {{{0, 1, 2}},
      {{0, 2, 3}},
      {{0, 3, 4}},
      {{0, 4, 1}}
    };
    const std::size_t numSimplices =
      sizeof(indexedSimplices) / sizeof(IndexedSimplex);

    const double Pi = numerical::Constants<double>::Pi();
    const double Eps = 10 * std::numeric_limits<double>::epsilon();
    ISS mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(std::abs(computeAngle(mesh, 0) - 2 * Pi) < Eps);
    for (std::size_t i = 1; i != numVertices; ++i) {
      assert(std::abs(computeAngle(mesh, i) - 0.5 * Pi) < Eps);
    }
  }

  // 3-D.
  {
    const std::size_t N = 3;
    typedef geom::IndSimpSetIncAdj<N, N> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;

    Vertex vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{ -1, 0, 0}},
      {{0, 1, 0}},
      {{0, -1, 0}},
      {{0, 0, 1}},
      {{0, 0, -1}}
    };
    const std::size_t numVertices = sizeof(vertices) / sizeof(Vertex);
    IndexedSimplex indexedSimplices[] = {{{0, 1, 3, 5}},
      {{0, 3, 2, 5}},
      {{0, 2, 4, 5}},
      {{0, 4, 1, 5}},
      {{0, 3, 1, 6}},
      {{0, 2, 3, 6}},
      {{0, 4, 2, 6}},
      {{0, 1, 4, 6}}
    };
    const std::size_t numSimplices =
      sizeof(indexedSimplices) / sizeof(IndexedSimplex);

    const double Pi = numerical::Constants<double>::Pi();
    const double Eps = 10 * std::numeric_limits<double>::epsilon();
    ISS mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(std::abs(computeAngle(mesh, 0) - 4 * Pi) < Eps);
    // CONTINUE: What is the correct angle?
#if 0
    for (std::size_t i = 1; i != numVertices; ++i) {
      assert(std::abs(computeAngle(mesh, 0) - ?) < Eps);
    }
#endif
  }

  return 0;
}
