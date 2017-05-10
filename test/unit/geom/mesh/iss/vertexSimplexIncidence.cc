// -*- C++ -*-

#include "stlib/geom/mesh/iss/vertexSimplexIncidence.h"
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

    // Make the incidences.
    container::StaticArrayOfArrays<std::size_t> x;
    geom::vertexSimplexIncidence(&x, numVertices, mesh.indexedSimplices);

    assert(x.getNumberOfArrays() == numVertices);
    assert(x.size() == 4 * mesh.indexedSimplices.size());
    assert(! x.empty());
    assert(x.begin() + x.size() == x.end());

    assert(x.size(0) == 8);
    assert(! x.empty(0));
    assert(x.begin(0) + x.size(0) == x.end(0));
    assert(x(0) == x.begin(0));
    assert(x(0, 0) == 0);

    x.clear();
    assert(x.size() == 0);
  }

  return 0;
}
