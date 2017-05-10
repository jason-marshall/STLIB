// -*- C++ -*-

#include "stlib/geom/mesh/iss/VertexSimplexInc.h"
#include "stlib/geom/mesh/iss/build.h"
#include "stlib/geom/mesh/iss/file_io.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::VertexSimplexInc<3> VSI;
  typedef geom::IndSimpSet<3> ISS;
  {
    // default constructor
    VSI inc;
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

    // Make the incidences.
    VSI x(numVertices, mesh.indexedSimplices);

    // Print the incidences.
    std::cout << "The incidences:\n" << x << "\n";

    {
      VSI y(x);
      assert(x == y);
    }
    {
      VSI y;
      y = x;
      assert(x == y);
    }
    {
      VSI y;
      y.build(numVertices, mesh.indexedSimplices);
      assert(x == y);
    }

    assert(x.getNumVertices() == numVertices);
    assert(x.getSize() == 4 * mesh.indexedSimplices.size());
    assert(! x.isEmpty());
    assert(x.getMaxSize() > 0);
    assert(x.getMemoryUsage() > 0);
    assert(x.getBeginning() + x.getSize() == x.getEnd());

    assert(x.getSize(0) == 8);
    assert(! x.isEmpty(0));
    assert(x.getBeginning(0) + x.getSize(0) == x.getEnd(0));
    assert(x[0] == x.getBeginning(0));
    assert(x(0) == x.getBeginning(0));
    assert(x(0, 0) == 0);

    x.clear();
    assert(x.getSize() == 0);
  }
  // CONTINUE: Write more tests.

  return 0;
}
