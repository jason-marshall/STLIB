// -*- C++ -*-

#include "stlib/geom/mesh/iss/transform.h"

#include "stlib/geom/mesh/iss/equality.h"

#include "stlib/ads/functor/constant.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // pack
  //
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;

    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t* indexedSimplices = 0;
    const std::size_t numSimplices = 0;

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 0);
    geom::pack(&mesh);
    assert(mesh.vertices.size() == 0);
    assert(mesh.indexedSimplices.size() == 0);
  }
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;

    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3
                                     };
    const std::size_t numSimplices = sizeof(indexedSimplices) / sizeof(
                                       std::size_t) / (M + 1);

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 2);
    geom::pack(&mesh);
    assert(mesh.vertices.size() == 4);
    assert(mesh.indexedSimplices.size() == 2);
  }
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;

    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };
    const std::size_t numSimplices = sizeof(indexedSimplices) / sizeof(
                                       std::size_t) / (M + 1);

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);
    geom::pack(&mesh);
    assert(mesh.vertices.size() == 5);
    assert(mesh.indexedSimplices.size() == 4);
  }

  //
  // orientPositive
  //
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;

    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };
    const std::size_t numSimplices = sizeof(indexedSimplices) / sizeof(
                                       std::size_t) / (M + 1);
    std::size_t flipped[] = {1, 0, 2,
                             2, 0, 3,
                             3, 0, 4,
                             4, 0, 1
                            };

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    Mesh x;
    build(&x, numVertices, vertices, numSimplices, flipped);
    geom::orientPositive(&x);
    assert(x == mesh);
  }

  //
  // Transform.
  //
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;
    typedef std::array<double, 2> Pt;

    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };
    const std::size_t numSimplices = sizeof(indexedSimplices) / sizeof(
                                       std::size_t) / (M + 1);

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    const Pt value = {{2, 3}};
    geom::transform(&mesh, ads::constructUnaryConstant<Pt>(value));
    for (std::size_t n = 0; n != mesh.vertices.size(); ++n) {
      assert(mesh.vertices[n] == value);
    }
  }
  {
    const std::size_t N = 2;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> Mesh;
    typedef std::array<double, 2> Pt;

    double vertices[] = {
      0, 0,
      1, 0,
      0, 1,
      -1, 0,
      0, -1
    };
    const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
    std::size_t indexedSimplices[] = {
      0, 1, 2,
      0, 2, 3,
      0, 3, 4,
      0, 4, 1
    };
    const std::size_t numSimplices = sizeof(indexedSimplices) / sizeof(
                                       std::size_t) / (M + 1);

    Mesh mesh;
    build(&mesh, numVertices, vertices, numSimplices, indexedSimplices);
    const Pt value = {{2, 3}};
    std::vector<std::size_t> indices;
    for (std::size_t n = 0; n != numVertices; ++n) {
      indices.push_back(n);
    }
    geom::transform(&mesh, indices.begin(), indices.end(),
                    ads::constructUnaryConstant<Pt>(value));
    for (std::size_t n = 0; n != mesh.vertices.size(); ++n) {
      assert(mesh.vertices[n] == value);
    }
  }

  return 0;
}
