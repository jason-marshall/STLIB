// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/coarsen.h"

#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/tile.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef geom::SimpMeshRed<2> SM;
  typedef SM::CellIterator CI;
  typedef SM::Vertex VT;

  //
  // Simple test with one internal edge.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}}, // 0
      {{1, 0}},  // 1
      {{1, 1}},  // 2
      {{0, 1}}
    }; // 3
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 2}},
      {{0, 2, 3}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    assert(geom::isValid(x));

    std::cout << "\n";
    geom::print(std::cout, x);
    std::cout << "\n";

    // Collapse a boundary edge.
    CI ci = x.getCellsBeginning();
    geom::collapse(&x, ci, 0);
    //assert(isValid(x));

    std::cout << "\n";
    geom::print(std::cout, x);
    std::cout << "\n";
  }

  //
  // Test with tiled region.
  //
  {
    typedef geom::BBox<double, 2> BBox;
    // Tile the unit square.
    SM x;
    tile(BBox{{{0., 0.}}, {{1., 1.}}}, 0.1, &x);
    // Write the initial mesh.
    std::cout << "square_fine.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Fine square:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    // Coarsen the mesh.
    const double minimumAllowedQuality = 0;
    const double qualityFactor = 0;
    std::cout
        << "Number of edges collapsed = "
        << geom::coarsen<geom::SimplexModCondNum<2> >
        (&x, ads::constructUnaryConstant<VT, double>(0.11),
         minimumAllowedQuality, qualityFactor)
        << "\n\n";

    // Write the coarsened mesh.
    std::cout << "square_coarsened.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "Coarsened square:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n\n";
  }

  //
  // Test with a zero length edge.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}}, // 0
      {{0, 0}},  // 1 (repeated vertex)
      {{1, 0}},  // 2
      {{1, 1}},  // 3
      {{0, 1}}
    }; // 4
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 3}},
      {{1, 2, 3}},
      {{3, 4, 0}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    assert(geom::isValid(x));

    // Write the initial mesh.
    std::cout << "zeroLengthInitial.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Zero length initial:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    // Coarsen the mesh.
    const double minimumAllowedQuality = 0;
    const double qualityFactor = 0;
    std::cout
        << "Number of edges collapsed = "
        << geom::coarsen<geom::SimplexModCondNum<2> >
        (&x, ads::constructUnaryConstant<VT, double>(0.1),
         minimumAllowedQuality, qualityFactor)
        << "\n\n";

    // Write the coarsened mesh.
    std::cout << "zeroLengthCoarsened.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "zero length coarsened:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n\n";
  }

  //
  // Test with a zero content interior simplex.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},  // 0
      {{0, 0}},   // 1 (repeated vertex)
      {{0, 0}},   // 2 (repeated vertex)
      {{1, 0}},   // 3
      {{ -1, 1}}, // 4
      {{ -1, -1}}
    };// 5
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 2}},
      {{0, 3, 1}},
      {{3, 4, 1}},
      {{1, 4, 2}},
      {{4, 5, 2}},
      {{2, 5, 0}},
      {{5, 3, 0}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    assert(geom::isValid(x));

    // Write the initial mesh.
    std::cout << "zeroContentInteriorInitial.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Zero content interior initial:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    // Coarsen the mesh.
    const double minimumAllowedQuality = 0;
    const double qualityFactor = 0;
    std::cout
        << "Number of edges collapsed = "
        << geom::coarsen<geom::SimplexModCondNum<2> >
        (&x, ads::constructUnaryConstant<VT, double>(0.1),
         minimumAllowedQuality, qualityFactor)
        << "\n\n";

    // Write the coarsened mesh.
    std::cout << "zeroContentInteriorCoarsened.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "zero content interior coarsened:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n\n";
  }

  //
  // Test with a zero content boundary simplex.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},    // 0
      {{1, 0}},     // 1
      {{0, 1}},     // 2
      {{0.5, 0.5}}, // 3
      {{0.5, 0.5}}, // 4
      {{0.5, 0.5}}
    };// 5
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 3}},
      {{0, 4, 2}},
      {{3, 4, 5}},
      {{0, 3, 5}},
      {{0, 5, 4}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    assert(geom::isValid(x));

    // Write the initial mesh.
    std::cout << "zeroContentBoundaryInitial.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Zero content boundary initial:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    // Coarsen the mesh.
    const double minimumAllowedQuality = 0;
    const double qualityFactor = 0;
    std::cout
        << "Number of edges collapsed = "
        << geom::coarsen<geom::SimplexModCondNum<2> >
        (&x, ads::constructUnaryConstant<VT, double>(0.1),
         minimumAllowedQuality, qualityFactor)
        << "\n\n";

    // Write the coarsened mesh.
    std::cout << "zeroContentBoundaryCoarsened.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "zero content boundary coarsened:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n\n";
  }

  return 0;
}
