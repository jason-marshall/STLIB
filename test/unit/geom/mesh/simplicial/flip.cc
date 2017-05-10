// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/flip.h"

#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/set.h"
#include "stlib/geom/mesh/simplicial/tile.h"
#include "stlib/geom/mesh/simplicial/transform.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include <iostream>

#include <cassert>

using namespace stlib;

// x -> x + y^2, y -> y
class Deform :
  public std::unary_function < const std::array<double, 2>&,
  const std::array<double, 2>& >
{
private:
  // Types.
  typedef std::unary_function < const std::array<double, 2>&,
          const std::array<double, 2>& > Base;

// Data.
  mutable std::array<double, 2> _p;

public:
// Types.
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

// Functor.
  result_type
  operator()(argument_type x) const
  {
    _p[0] = x[0] + x[1] * x[1];
    _p[1] = x[1];
    return _p;
  }
};


int
main()
{
  typedef geom::SimpMeshRed<2> SM;

  typedef SM::Node Node;
  typedef SM::CellIterator CI;

  // Test flip.
  {
    //
    // Data for a square
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},  // 0
      {{1, 0}},   // 1
      {{1, 1}},   // 2
      {{0, 1}}
    };  // 3
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
    assert(isValid(x));
    std::cout << "\n";
    geom::print(std::cout, x);
    std::cout << "\n";

    // Flip the diagonal.
    CI ch = x.getCellsBeginning();
    geom::flip<SM>(ch, 1);
    std::cout << "\n";
    geom::print(std::cout, x);
    std::cout << "\n";
    assert(isValid(x));
  }


  //
  // Simple test with one internal face.
  //
  {
    //
    // Data for the mesh.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},    // 0
      {{0.9, 0.1}}, // 1
      {{1, 1}},     // 2
      {{0.1, 0.9}}
    };// 3
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
    assert(isValid(x));
    /*
    std::cout << "\n";
    x.print(std::cout);
    std::cout << "\n";
    */

    // Flip the diagonal.
    assert(geom::flipUsingModifiedMeanRatio(&x) == 1);
    assert(isValid(x));
    /*
    std::cout << "\n";
    x.print(std::cout);
    std::cout << "\n";
    */
  }

  //
  // Test with a deformed, tiled region.
  //
  {
    typedef geom::BBox<double, 2> BBox;
    // Tile the unit square.
    SM x;
    tile(BBox{{{0., 0.}}, {{1., 1.}}}, 0.1, &x);
    x.setNodeIdentifiers();
    x.setCellIdentifiers();
    // Deform the mesh.
    std::vector<Node*> verts;
    geom::getNodes(x, std::back_inserter(verts));
    geom::transformNodes<SM>(verts.begin(), verts.end(), Deform());
    // Write the initial mesh.
    std::cout << "square_init.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the initial mesh.
    std::cout << "Initial Mesh:\n";
    geom::printQualityStatistics(std::cout, x);

    // Perform flips.
    std::cout << "\nNumber of flips = " << flipUsingModifiedMeanRatio(&x)
              << "\n\n";
    // Write the flipped mesh.
    std::cout << "square_flip.vtu\n";
    geom::writeVtkXml(std::cout, x);
    // Print the quality of the flipped mesh.
    std::cout << "Flipped Mesh:\n";
    geom::printQualityStatistics(std::cout, x);
  }

  return 0;
}
