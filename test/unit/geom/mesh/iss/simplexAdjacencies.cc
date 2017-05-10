// -*- C++ -*-

#include "stlib/geom/mesh/iss/simplexAdjacencies.h"
#include "stlib/geom/mesh/iss/build.h"

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::IndSimpSet<3> ISS;
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
  {
    // Construct a mesh from vertices and tetrahedra.
    ISS mesh;
    build(&mesh, numVertices, vertices, numTets, tets);

    {
      // Make the vertex-simplex incidences.
      container::StaticArrayOfArrays<std::size_t> inc;
      geom::vertexSimplexIncidence(&inc, numVertices, mesh.indexedSimplices);
      // Make the simplex adjacencies.
      std::vector < std::array < std::size_t, ISS::M + 1 > > adjacent;
      geom::simplexAdjacencies(&adjacent, mesh.indexedSimplices, inc);
      // Check the adjacencies.
      assert(adjacent.size() == mesh.indexedSimplices.size());
      for (std::size_t i = 0; i != adjacent.size(); ++i) {
        assert(geom::numAdjacent(adjacent[i]) == 3);
      }
    }
    {
      // Make the simplex adjacencies.
      std::vector < std::array < std::size_t, ISS::M + 1 > > adjacent;
      geom::simplexAdjacencies(&adjacent, mesh.vertices.size(),
                               mesh.indexedSimplices);
      // Check the adjacencies.
      assert(adjacent.size() == mesh.indexedSimplices.size());
      for (std::size_t i = 0; i != adjacent.size(); ++i) {
        assert(geom::numAdjacent(adjacent[i]) == 3);
      }
    }
  }

  return 0;
}
