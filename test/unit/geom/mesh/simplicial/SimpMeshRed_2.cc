// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace geom;

  typedef SimpMeshRed<2> SM;
  typedef SM::NodeConstIterator NCI;
  typedef SM::CellConstIterator CCI;
  typedef SM::FaceConstIterator FCI;

  // Default constructor.
  {
    SM x;
  }

  {
    //
    // Data for a diamond.
    //
    const std::array<double, 2> vertices[] = {{{0, 0}},   // 0
      {{1, 0}},    // 1
      {{0, 1}},    // 2
      {{ -1, 0}},  // 3
      {{0, -1}}
    };  // 4
    const std::size_t numVertices = sizeof(vertices) /
                                    sizeof(std::array<double, 2>);
    const std::array<std::size_t, 3> simplices[] = {{{0, 1, 2}},
      {{0, 2, 3}},
      {{0, 3, 4}},
      {{0, 4, 1}}
    };
    const std::size_t numSimplices = sizeof(simplices) /
                                     sizeof(std::array<std::size_t, 3>);

    // Build from an indexed simplex set.
    {
      SM x;
      x.build(vertices, vertices + numVertices,
              simplices, simplices + numSimplices);
      assert(isValid(x));
      assert(x.computeNodesSize() == numVertices);
      assert(x.computeCellsSize() == numSimplices);
      std::cout << "The mesh:\n";
      geom::writeAscii(std::cout, x);
      std::cout << "\n";

      // Make a indexed simplex set from the mesh.
      {
        IndSimpSet<2> iss;
        geom::buildIndSimpSetFromSimpMeshRed(x, &iss);
        std::cout << "The indexed simplex set:\n";
        geom::writeAscii(std::cout, iss);
        std::cout << "\n";
      }

      // Iterate over the nodes.
      {
        std::cout << "The vertices:\n";
        NCI v = x.getNodesBeginning();
        const NCI v_end = x.getNodesEnd();
        for (; v != v_end; ++v) {
          std::cout << v->getVertex() << "\n";
        }
      }

      // Iterate over the cells.
      {
        std::cout << "The vertices of the cells:\n";
        CCI c = x.getCellsBeginning();
        const CCI c_end = x.getCellsEnd();
        for (; c != c_end; ++c) {
          for (std::size_t i = 0; i != 3; ++i) {
            std::cout << c->getNode(i)->getVertex() << "   ";
          }
          std::cout << "\n";
        }
      }

      // Iterate over the faces.
      {
        const SM& y = x;
        std::size_t n = 0; // Face number.
        std::cout << "The vertices of the faces:\n";
        FCI f = y.getFacesBeginning();
        const FCI f_end = y.getFacesEnd();
        std::size_t vi; // vertex index.
        for (; f != f_end; ++f) {
          vi = f->second;
          std::cout << n << "   " << vi << "   ";
          ++n;
          for (std::size_t i = 1; i != 3; ++i) {
            std::cout << (*f).first->getNode((vi + i) % 3)->getVertex() << "   ";
          }
          std::cout << "\n";
        }
      }
      // Count the faces.
      assert(x.computeFacesSize() == 8);
    }
  }

  return 0;
}
