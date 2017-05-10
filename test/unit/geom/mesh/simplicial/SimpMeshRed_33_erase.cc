// -*- C++ -*-

//
// Test the erase operations for SimpMeshRed<3,3>.
//

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplicial/manipulators.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  // 3-D space, 3-simplex.
  typedef geom::SimpMeshRed<3> SM;

  typedef SM::CellIterator CI;

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


  //
  // Erase one cell at a time.
  //
  {
    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);

    while (x.computeCellsSize() != 0) {
      std::size_t size = x.computeCellsSize();
      x.eraseCell(x.getCellsBeginning());
      geom::eraseUnusedNodes(&x);
      assert(x.computeCellsSize() == size - 1);
      assert(geom::isValid(x));
    }
  }

  //
  // Erase a range of cells.
  //
  {
    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    std::vector<CI> s;
    for (CI i = x.getCellsBeginning(); i != x.getCellsEnd(); ++i) {
      s.push_back(i);
    }

    x.eraseCells(s.begin(), s.end());
    assert(x.computeCellsSize() == 0);
    assert(x.computeNodesSize() == numVertices);
    geom::eraseUnusedNodes(&x);
    assert(x.computeNodesSize() == 0);
    assert(geom::isValid(x));
  }

  //
  // Erase cells with low adjacencies.
  //
  {
    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);

    geom::eraseCellsWithLowAdjacencies(&x, 1);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == numVertices);
    assert(geom::isValid(x));

    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    geom::eraseCellsWithLowAdjacencies(&x, 2);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == numVertices);
    assert(geom::isValid(x));

    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    geom::eraseCellsWithLowAdjacencies(&x, 3);
    assert(x.computeCellsSize() == 8);
    assert(x.computeNodesSize() == numVertices);
    assert(geom::isValid(x));

    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    geom::eraseCellsWithLowAdjacencies(&x, 4);
    assert(x.computeCellsSize() == 0);
    assert(x.computeNodesSize() == 0);
    assert(geom::isValid(x));
  }

  return 0;
}
