// -*- C++ -*-

#include "stlib/geom/mesh/iss/SimplexAdj.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::SimplexAdj<3> SA;
  typedef geom::IndSimpSet<3> ISS;
  {
    // default constructor
    SA adj;
  }
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

    // Print the mesh.
    std::cout << "The mesh for an octahedron:\n";
    geom::writeAscii(std::cout, mesh);
    std::cout << "\n";

    // Make the vertex-simplex incidences.
    container::StaticArrayOfArrays<std::size_t> inc;
    geom::vertexSimplexIncidence(&inc, numVertices, mesh.indexedSimplices);

    // Make the simplex adjacencies.
    SA adj(mesh.indexedSimplices, inc);

    // Print the adjacencies.
    std::cout << "The adjacencies:\n" << adj << '\n';

    // Print the number of adjacencies for each simplex.
    std::cout << "The number of adjacencies for each simplex:\n";
    for (std::size_t n = 0; n != adj.getSize(); ++n) {
      std::cout << n << " " << adj.getSize(n) << "\n";
    }
  }

  return 0;
}
