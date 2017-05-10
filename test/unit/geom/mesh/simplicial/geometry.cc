// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/geometry.h"
#include "stlib/geom/mesh/iss/build.h"

#include <cassert>

using namespace stlib;

int
main()
{
  // CONTINUE: use a standard constant.
  const double pi = 3.1415926535897932384626433832795;
  const double eps = std::sqrt(std::numeric_limits<double>::epsilon());

  //
  // computeDihedralAngle
  //
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef geom::SimpMeshRed<3, 3> SMR;
    typedef SMR::ConstEdge Edge;

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

    {
      const Edge edge(x.getCellsBeginning(), 0, 1);
      const double angle = pi / 2.0;
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
    {
      const Edge edge(x.getCellsBeginning(), 0, 2);
      const double angle = pi / 2.0;
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
    {
      const Edge edge(x.getCellsBeginning(), 0, 3);
      const double angle = pi / 2.0;
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
    {
      const Edge edge(x.getCellsBeginning(), 1, 2);
      const double angle = std::atan(std::sqrt(2.0));
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
    {
      const Edge edge(x.getCellsBeginning(), 1, 3);
      const double angle = std::atan(std::sqrt(2.0));
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
    {
      const Edge edge(x.getCellsBeginning(), 2, 3);
      const double angle = std::atan(std::sqrt(2.0));
      assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
    }
  }
  {
    typedef geom::IndSimpSet<3, 3> ISS;
    typedef geom::SimpMeshRed<3, 3> SMR;
    typedef SMR::ConstEdge Edge;
    typedef SMR::CellConstIterator CellConstIterator;

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

    for (CellConstIterator i = x.getCellsBeginning(); i != x.getCellsEnd();
         ++i) {
      {
        const Edge edge(x.getCellsBeginning(), 0, 1);
        const double angle = 2.0 * pi;
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
      {
        const Edge edge(x.getCellsBeginning(), 0, 2);
        const double angle = pi;
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
      {
        const Edge edge(x.getCellsBeginning(), 0, 3);
        const double angle = pi;
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
      {
        const Edge edge(x.getCellsBeginning(), 1, 2);
        const double angle = 2.0 * std::atan(std::sqrt(2.0));
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
      {
        const Edge edge(x.getCellsBeginning(), 1, 3);
        const double angle = 2.0 * std::atan(std::sqrt(2.0));
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
      {
        const Edge edge(x.getCellsBeginning(), 2, 3);
        const double angle = std::atan(std::sqrt(2.0));
        assert(std::abs(geom::computeDihedralAngle<SMR>(edge) - angle) < eps);
      }
    }
  }

  return 0;
}
