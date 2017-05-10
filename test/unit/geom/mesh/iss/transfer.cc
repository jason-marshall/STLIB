// -*- C++ -*-

#include "stlib/geom/mesh/iss/transfer.h"
#include "stlib/geom/mesh/iss/build.h"

#include "stlib/container/MultiArray.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  const double eps = 100.0 * std::numeric_limits<double>::epsilon();

  //
  // 3-D space.  3-D simplex.  Indices.
  //
  {
    const std::size_t N = 3;
    const std::size_t M = 3;
    typedef geom::IndSimpSet<N> ISS;
    typedef ISS::Vertex Vertex;

    //
    // Data for an octahedron
    //
    double verticesData[] = {0, 0, 0,   // 0
                             1, 0, 0,    // 1
                             -1, 0, 0,   // 2
                             0, 1, 0,    // 3
                             0, -1, 0,   // 4
                             0, 0, 1,    // 5
                             0, 0, -1
                            }; // 6
    const std::size_t numVertices = sizeof(verticesData) / sizeof(double) / N;
    std::size_t simplicesData[] = { 0, 1, 3, 5,
                                    0, 3, 2, 5,
                                    0, 2, 4, 5,
                                    0, 4, 1, 5,
                                    0, 3, 1, 6,
                                    0, 2, 3, 6,
                                    0, 4, 2, 6,
                                    0, 1, 4, 6
                                  };
    const std::size_t numSimplices = sizeof(simplicesData) / sizeof(std::size_t) /
                                     (M + 1);
    Vertex vertexData[] = {{{1, 1, 1}},
      {{ -1, 1, 1}},
      {{ -1, -1, 1}},
      {{1, -1, 1}},
      {{1, 1, -1}},
      {{ -1, 1, -1}},
      {{ -1, -1, -1}},
      {{1, -1, -1}}
    };
    const std::size_t numPoints = sizeof(vertexData) / sizeof(Vertex);

    ISS mesh;
    build(&mesh, numVertices, verticesData, numSimplices, simplicesData);
    std::vector<Vertex> points(vertexData, vertexData + numPoints);
    std::vector<std::size_t> indices(numPoints);

    geom::transferIndices(mesh, points, &indices);

    for (std::size_t i = 0; i != points.size(); ++i) {
      assert(indices[i] == i);
    }
  }


  //
  // 3-D space.  3-D simplex.  1-D field.  Linear interpolation transfer.
  //
  {
    const std::size_t N = 3;
    const std::size_t M = 3;
    typedef geom::IndSimpSet<N> ISS;
    typedef ISS::Vertex Vertex;
    typedef double Field;

    //
    // Data for an octahedron
    //
    double verticesData[] = { 0, 0, 0,
                              1, 0, 0,
                              -1, 0, 0,
                              0, 1, 0,
                              0, -1, 0,
                              0, 0, 1,
                              0, 0, -1
                            };
    const std::size_t numVertices = sizeof(verticesData) / sizeof(double) / N;
    std::size_t simplicesData[] = { 0, 1, 3, 5,
                                    0, 3, 2, 5,
                                    0, 2, 4, 5,
                                    0, 4, 1, 5,
                                    0, 3, 1, 6,
                                    0, 2, 3, 6,
                                    0, 4, 2, 6,
                                    0, 1, 4, 6
                                  };
    const std::size_t numSimplices = sizeof(simplicesData) / sizeof(std::size_t) /
                                     (M + 1);

    ISS mesh;
    build(&mesh, numVertices, verticesData, numSimplices, simplicesData);
    std::vector<Field> fields(numVertices);

    for (std::size_t i = 0; i != mesh.vertices.size(); ++i) {
      fields[i] = mesh.vertices[i][0] + 2 * mesh.vertices[i][1]
                  + 3 * mesh.vertices[i][2];
    }

    container::MultiArray<Vertex, 3>
      targetVerticesGrid(std::array<std::size_t, 3>{{11, 11, 11}});
    container::MultiArray<Vertex, 3>::IndexList i;
    for (i[0] = 0; i[0] != 11; ++i[0]) {
      for (i[1] = 0; i[1] != 11; ++i[1]) {
        for (i[2] = 0; i[2] != 11; ++i[2]) {
          targetVerticesGrid(i) = Vertex{{0.4 * i[0] - 2.,
                                          0.4 * i[1] - 2.,
                                          0.4 * i[2] - 2.}};
        }
      }
    }

    std::vector<Vertex>
    targetVertices(targetVerticesGrid.data(), targetVerticesGrid.data() +
                   targetVerticesGrid.size());
    std::vector<Field> targetFields(targetVertices.size());

    geom::transfer(mesh, fields, targetVertices, &targetFields);

    for (std::size_t i = 0; i != targetVertices.size(); ++i) {
      assert(std::abs(targetFields[i] -
                      (targetVertices[i][0] + 2 * targetVertices[i][1] +
                       3 * targetVertices[i][2])) < eps);
    }
  }


  //
  // 3-D space.  3-D simplex.  3-D field.  Linear interpolation transfer.
  //
  {
    const std::size_t N = 3;
    typedef geom::IndSimpSet<N> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef Vertex Field;

    //
    // Data for an octahedron
    //
    Vertex verticesData[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{ -1, 0, 0}},
      {{0, 1, 0}},
      {{0, -1, 0}},
      {{0, 0, 1}},
      {{0, 0, -1}}
    };
    const std::size_t numVertices = sizeof(verticesData) / sizeof(Vertex);
    IndexedSimplex simplicesData[] = {{{0, 1, 3, 5}},
                                      {{0, 3, 2, 5}},
                                      {{0, 2, 4, 5}},
                                      {{0, 4, 1, 5}},
                                      {{0, 3, 1, 6}},
                                      {{0, 2, 3, 6}},
                                      {{0, 4, 2, 6}},
                                      {{0, 1, 4, 6}}
                                     };
    const std::size_t numSimplices = sizeof(simplicesData) /
                                     sizeof(IndexedSimplex);

    ISS mesh;
    build(&mesh, numVertices, verticesData, numSimplices, simplicesData);
    // Use the vertex locations as the field.
    std::vector<Field> fields(verticesData, verticesData + numVertices);

    container::MultiArray<Vertex, 3>
      targetVerticesGrid(std::array<std::size_t, 3>{{11, 11, 11}});
    container::MultiArray<Vertex, 3>::IndexList i;
    for (i[0] = 0; i[0] != 11; ++i[0]) {
      for (i[1] = 0; i[1] != 11; ++i[1]) {
        for (i[2] = 0; i[2] != 11; ++i[2]) {
          targetVerticesGrid(i) = Vertex{{0.4 * i[0] - 2.,
                                          0.4 * i[1] - 2.,
                                          0.4 * i[2] - 2.}};
        }
      }
    }

    std::vector<Vertex>
    targetVertices(targetVerticesGrid.data(), targetVerticesGrid.data() +
                   targetVerticesGrid.size());
    std::vector<Field> targetFields(targetVertices.size());

    geom::transfer(mesh, fields, targetVertices, &targetFields);

    for (std::size_t i = 0; i != targetVertices.size(); ++i) {
      assert(geom::computeDistance(targetFields[i], targetVertices[i]) < eps);
    }
  }


  //
  // 3-D space.  2-D simplex.  1-D field.  Linear interpolation transfer.
  //
  {
    const std::size_t N = 3;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> ISS;
    typedef ISS::Vertex Vertex;
    typedef double Field;

    //
    // Data for an octahedron
    //
    double verticesData[] = {1, 0, 0,    // 0
                             -1, 0, 0,    // 1
                             0, 1, 0,     // 2
                             0, -1, 0,    // 3
                             0, 0, 1,     // 4
                             0, 0, -1
                            };  // 5
    const std::size_t numVertices = sizeof(verticesData) / sizeof(double) / N;
    // z value.
    double fieldsData[] = {0, 0, 0, 0, 1, -1};
    std::size_t simplicesData[] = {0, 2, 4,
                                   2, 0, 5,
                                   2, 1, 4,
                                   1, 2, 5,
                                   1, 3, 4,
                                   3, 1, 5,
                                   3, 0, 4,
                                   0, 3, 5
                                  };
    const std::size_t numSimplices = sizeof(simplicesData) / sizeof(std::size_t) /
                                     (M + 1);

    // The source mesh.
    ISS mesh;
    build(&mesh, numVertices, verticesData, numSimplices, simplicesData);
    std::vector<Field> fields(fieldsData, fieldsData + numVertices);

    // The target vertices lie on the octahedron.
    // |x| + |y| + |z| = 1.
    Vertex targetVerticesData[] = {{{0, 0, -1}},
      {{0.5, 0, -0.5}},
      {{0.25, -0.25, -0.5}},
      {{0.25, 0.75, 0}},
      {{0.5, 0, 0.5}},
      {{0, 0, 1}}
    };
    const std::size_t numTargetVertices =
      sizeof(targetVerticesData) / sizeof(Vertex);

    std::vector<Vertex>
    targetVertices(targetVerticesData, targetVerticesData +
                   numTargetVertices);
    std::vector<Field> targetFields(numTargetVertices);

    geom::transfer(mesh, fields, targetVertices, &targetFields);

    for (std::size_t i = 0; i != targetVertices.size(); ++i) {
      assert(std::abs(targetFields[i] - targetVertices[i][2]) < eps);
    }
  }


  //
  // 3-D space.  2-simplex.  3-D field.  Linear interpolation transfer.
  //
  {
    const std::size_t N = 3;
    const std::size_t M = 2;
    typedef geom::IndSimpSet<N, M> ISS;
    typedef ISS::Vertex Vertex;
    typedef ISS::IndexedSimplex IndexedSimplex;
    typedef Vertex Field;

    //
    // Data for an octahedron
    //
    Vertex verticesData[] = {{{1, 0, 0}},    // 0
      {{ -1, 0, 0}},   // 1
      {{0, 1, 0}},     // 2
      {{0, -1, 0}},    // 3
      {{0, 0, 1}},     // 4
      {{0, 0, -1}}
    };   // 5
    const std::size_t numVertices = sizeof(verticesData) / sizeof(Vertex);
    IndexedSimplex simplicesData[] = {{{0, 2, 4}},
                                      {{2, 0, 5}},
                                      {{2, 1, 4}},
                                      {{1, 2, 5}},
                                      {{1, 3, 4}},
                                      {{3, 1, 5}},
                                      {{3, 0, 4}},
                                      {{0, 3, 5}}
                                     };
    const std::size_t numSimplices = sizeof(simplicesData) /
                                     sizeof(IndexedSimplex);

    // The source mesh.
    ISS mesh;
    build(&mesh, numVertices, verticesData, numSimplices, simplicesData);
    // Use the vertex locations as the field.
    std::vector<Field> fields(verticesData, verticesData + numVertices);

    // The target vertices lie on the octahedron.
    // |x| + |y| + |z| = 1.
    Vertex targetVerticesData[] = {{{0, 0, -1}},
      {{0.5, 0, -0.5}},
      {{0.25, -0.25, -0.5}},
      {{0.25, 0.75, 0}},
      {{0.5, 0, 0.5}},
      {{0, 0, 1}}
    };
    const std::size_t numTargetVertices =
      sizeof(targetVerticesData) / sizeof(Vertex);

    std::vector<Vertex>
    targetVertices(targetVerticesData, targetVerticesData +
                   numTargetVertices);
    std::vector<Field> targetFields(numTargetVertices);

    geom::transfer(mesh, fields, targetVertices, &targetFields);

    for (std::size_t i = 0; i != targetVertices.size(); ++i) {
      assert(geom::computeDistance(targetFields[i], targetVertices[i]) < eps);
    }
  }
  return 0;
}
