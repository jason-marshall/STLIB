// -*- C++ -*-

#include "stlib/geom/mesh/iss/topology.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/kernel/BBox.h"

using namespace stlib;

int
main()
{
  //
  // 3-D space.  3-D simplex.
  //
  {
    typedef geom::IndSimpSetIncAdj<3, 3> ISS;
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

    // Construct from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    for (std::size_t i = 0; i != mesh.indexedSimplices.size(); ++i) {
      for (std::size_t j = 0; j != mesh.indexedSimplices.size(); ++j) {
        assert(geom::doSimplicesShareAnyVertex(mesh, i, j));
      }
    }
  }

  //
  // 3-D space.  2-D simplex.
  //
  {
    typedef geom::IndSimpSetIncAdj<3, 2> ISS;
    //
    // Data for an octahedron
    //
    const std::size_t numVertices = 6;
    double vertices[] = {1, 0, 0,    // 0
                         -1, 0, 0,    // 1
                         0, 1, 0,     // 2
                         0, -1, 0,    // 3
                         0, 0, 1,     // 4
                         0, 0, -1
                        };   // 5
    const std::size_t num_triangles = 8;
    std::size_t triangles[] = {0, 2, 4, // 0
                               2, 0, 5, // 1
                               2, 1, 4, // 2
                               1, 2, 5, // 3
                               1, 3, 4, // 4
                               3, 1, 5, // 5
                               3, 0, 4, // 6
                               0, 3, 5
                              };// 7

    // Construct from vertices and triangles.
    ISS mesh;
    build(&mesh, numVertices, vertices, num_triangles, triangles);

    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 0));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 1));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 2));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 3));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 4));
    assert(! geom::doSimplicesShareAnyVertex(mesh, 0, 5));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 6));
    assert(geom::doSimplicesShareAnyVertex(mesh, 0, 7));
  }

  return 0;
}
