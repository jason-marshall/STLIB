// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/accessors.h"
#include "stlib/geom/mesh/iss/build.h"

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // isOnBoundary
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef geom::SimpMeshRed<3, 3> SMR;

    // Tetrahedron.
    const std::size_t numVertices = 4;
    double vertices[] = {0, 0, 0,
                         1, 0, 0,
                         0, 1, 0,
                         0, 0, 1
                        };
    const std::size_t numSimplices = 1;
    std::size_t indexedSimplices[] = {0, 1, 2, 3};

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, indexedSimplices);
    SMR x(iss);
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 0, 1));
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 0, 2));
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 0, 3));
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 1, 2));
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 1, 3));
    assert(geom::isOnBoundary<SMR>(x.getCellsBeginning(), 2, 3));
  }
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef geom::SimpMeshRed<3, 3> SMR;
    typedef SMR::CellIterator CellIterator;

    // Four simplices, one interior edge.
    const std::size_t numVertices = 6;
    double vertices[] = {0, 0, 0,
                         1, 0, 0,
                         0, 1, 0,
                         0, 0, 1,
                         0, -1, 0,
                         0, 0, -1
                        };
    const std::size_t numSimplices = 4;
    std::size_t indexedSimplices[] = {0, 1, 2, 3,
                                      0, 1, 3, 4,
                                      0, 1, 4, 5,
                                      0, 1, 5, 2
                                     };

    ISS iss;
    build(&iss, numVertices, vertices, numSimplices, indexedSimplices);
    SMR x(iss);
    for (CellIterator i = x.getCellsBeginning(); i != x.getCellsEnd(); ++i) {
      assert(! geom::isOnBoundary<SMR>(i, 0, 1));
      assert(geom::isOnBoundary<SMR>(i, 0, 2));
      assert(geom::isOnBoundary<SMR>(i, 0, 3));
      assert(geom::isOnBoundary<SMR>(i, 1, 2));
      assert(geom::isOnBoundary<SMR>(i, 1, 3));
      assert(geom::isOnBoundary<SMR>(i, 2, 3));
    }
  }

  return 0;
}
