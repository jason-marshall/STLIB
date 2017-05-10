// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/refine.h"

#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/tile.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include <iostream>

#include <cassert>


using namespace stlib;

template <typename Pt, typename T>
class EdgeLength :
  public std::unary_function<Pt, T>
{
public:
  typedef std::unary_function<Pt, T> base_type;
  typedef typename base_type::argument_type argument_type;
  typedef typename base_type::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    return 0.01 + std::abs(x[0]);
  }
};


int
main()
{
  typedef geom::SimpMeshRed<2> SM;

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

    assert(geom::refine(&x, ads::constructUnaryConstant<VT, double>(1.1)) == 1);
    assert(isValid(x));

    std::cout << "\n";
    geom::print(std::cout, x);
    std::cout << "\n";
  }

  //
  // Test with tiled region, uniform refinement.
  //
  {
    typedef geom::BBox<double, 2> BBox;
    // Tile the unit square.
    SM x;
    tile(BBox{{{0., 0.}}, {{1., 1.}}}, 0.2, &x);
    // Write the initial mesh.
    std::cout << "square_coarse.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Coarse square:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    assert(isValid(x));
    // Refine the mesh.
    std::cout
        << "Number of edges split = "
        << geom::refine(&x, ads::constructUnaryConstant<VT, double>(0.19))
        << "\n\n";

    // Write the refined mesh.
    std::cout << "square_refined.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "Refined square:\n";
    geom::printQualityStatistics(std::cout, x);
  }

  //
  // Test with tiled region, non-uniform refinement.
  //
  {
    typedef geom::BBox<double, 2> BBox;
    // Tile the unit square.
    SM x;
    tile(BBox{{{0., 0.}}, {{1., 1.}}}, 0.2, &x);
    // Write the initial mesh.
    std::cout << "non_un_init.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Coarse square:\n";
    geom::printQualityStatistics(std::cout, x);
    std::cout << "\n";

    // Refine the mesh.
    EdgeLength<VT, double> f;
    std::cout
        << "Number of edges split = "
        << geom::refine(&x, f)
        << "\n\n";

    // Write the refined mesh.
    std::cout << "non_un_final.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the coarse mesh.
    std::cout << "Refined square:\n";
    geom::printQualityStatistics(std::cout, x);
  }

  //
  // Refine a set of cells.  Start with only two cells.
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

    std::vector<std::size_t> cells;
    cells.push_back(0);
    assert(geom::refine(&x, cells.begin(), cells.end()) == 1);
    assert(isValid(x));
  }

  //
  // Refine a set of cells.  Larger test.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},
      {{1, 0}},
      {{2, 0}},
      {{0, 1}},
      {{1, 1}},
      {{2, 1}},
      {{0, 2}},
      {{1, 2}},
      {{2, 2}}
    };
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 4}},
      {{0, 4, 3}},
      {{1, 2, 5}},
      {{1, 5, 4}},
      {{3, 4, 7}},
      {{3, 7, 6}},
      {{4, 5, 8}},
      {{4, 8, 7}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    SM x;
    x.build(vertices, vertices + numVertices,
            simplices, simplices + numSimplices);
    assert(geom::isValid(x));

    assert(geom::refine(&x, ads::IntIterator<>(0),
                        ads::IntIterator<>(numSimplices)) ==
           numSimplices / 2);
    assert(isValid(x));
    assert(x.computeCellsSize() == 2 * numSimplices);

    geom::refine(&x, ads::IntIterator<>(0),
                 ads::IntIterator<>(2 * numSimplices));
    assert(x.computeCellsSize() == 4 * numSimplices);
  }

  //
  // Refine a set of cells.  Start with only two cells.
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

    for (std::size_t n = 0; n != 10; ++n) {
      geom::refine(&x, ads::IntIterator<>(0), ads::IntIterator<>(1));
    }

    assert(isValid(x));

    // Write the refined mesh.
    std::cout << "refine_cells.vtu\n";
    geom::writeVtkXml(std::cout, x);
  }

  return 0;
}
