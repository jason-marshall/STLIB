// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/manipulators.h"
#include "stlib/geom/mesh/simplicial/build.h"

#include "stlib/geom/mesh/iss/equality.h"
#include "stlib/geom/mesh/iss/build.h"

#include "stlib/ads/functor.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // orient_positive
  //
  {
    typedef geom::IndSimpSet<2, 2> ISS;
    typedef geom::SimpMeshRed<2, 2> SMR;

    const std::size_t numVertices = 5;
    double vertices[] = {0, 0,
                         1, 0,
                         0, 1,
                         -1, 0,
                         0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = {0, 1, 2,
                                      0, 2, 3,
                                      0, 3, 4,
                                      0, 4, 1
                                     };
    std::size_t flipped[] = {1, 0, 2,
                             2, 0, 3,
                             3, 0, 4,
                             4, 0, 1
                            };

    ISS iss_p;
    build(&iss_p, numVertices, vertices, numSimplices, indexedSimplices);
    ISS iss_n;
    build(&iss_n, numVertices, vertices, numSimplices, flipped);
    SMR p(iss_p);
    SMR n(iss_n);
    geom::orientPositive(&n);
    ISS x, y;
    geom::buildIndSimpSetFromSimpMeshRed(p, &x);
    geom::buildIndSimpSetFromSimpMeshRed(n, &y);
    assert(x == y);
  }


  //
  // Cells with minimum adjacencies.
  //
  {
    typedef geom::SimpMeshRed<3> SMR;
    //
    // Data for an octahedron
    //
    const std::array<double, 3> vertices[] = {{{0, 0, 0}},
      {{1, 0, 0}},
      {{ -1, 0, 0}},
      {{0, 1, 0}},
      {{0, -1, 0}},
      {{0, 0, 1}},
      {{0, 0, -1}}
    };
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 3>);
    const std::array<std::size_t, 4> simplices[] = {{{0, 1, 3, 5}},
      {{0, 3, 2, 5}},
      {{0, 2, 4, 5}},
      {{0, 4, 1, 5}},
      {{0, 3, 1, 6}},
      {{0, 2, 3, 6}},
      {{0, 4, 2, 6}},
      {{0, 1, 4, 6}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 4>);

    // Build from an indexed simplex set.
    SMR x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);

    eraseCellsWithLowAdjacencies(&x, 0);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == 7);

    eraseCellsWithLowAdjacencies(&x, 1);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == 7);

    eraseCellsWithLowAdjacencies(&x, 2);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == 7);

    eraseCellsWithLowAdjacencies(&x, 3);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == 7);

    eraseCellsWithLowAdjacencies(&x, 4);
    assert(x.computeCellsSize() == 0);
    assert(x.computeNodesSize() == 0);
  }
  return 0;
}
