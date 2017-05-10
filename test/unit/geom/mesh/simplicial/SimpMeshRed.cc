// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/SimpMeshRed.h"

#include "stlib/geom/mesh/simplicial/build.h"
#include "stlib/geom/mesh/simplicial/topology.h"
#include "stlib/geom/mesh/simplicial/valid.h"

#include "stlib/geom/mesh/iss/file_io.h"

#include <iostream>

#include <cassert>

using namespace stlib;

// Distance function for a unit sphere.
class UnitSphere :
  public std::unary_function<const std::array<double, 3>&, double>
{
private:
  typedef std::unary_function<const std::array<double, 3>&, double> Base;

public:
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    return stlib::ext::magnitude(x) - 1;
  }
};


int
main()
{
  using namespace geom;

  //---------------------------------------------------------------------------
  // 3-D space, 3-D simplex.
  //---------------------------------------------------------------------------
  {
    typedef SimpMeshRed<3> SM;

    typedef SM::NodeConstIterator NCI;
    typedef SM::VertexIterator VI;
    typedef SM::NodeIdentifierIterator NII;

    typedef SM::CellConstIterator CCI;

    typedef SM::IndexedSimplexIterator ISI;

    typedef SM::FaceConstIterator FCI;
    typedef SM::FaceIterator FI;

    // Default constructor.
    {
      SM x;
    }

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
    {
      SM x;
      x.build(vertices, vertices + numVertices,
              simplices, simplices + numSimplices);
      assert(isValid(x));
      assert(x.computeNodesSize() == numVertices);
      assert(x.computeCellsSize() == numSimplices);

      // Make a indexed simplex set from the mesh.
      {
        IndSimpSet<3> iss;
        geom::buildIndSimpSetFromSimpMeshRed(x, &iss);
        std::cout << "The indexed simplex set:\n";
        geom::writeAscii(std::cout, iss);
      }

      // Iterate over the nodes.
      {
        std::cout << "The nodes:\n";
        NCI iter = x.getNodesBeginning();
        const NCI iter_end = x.getNodesEnd();
        for (; iter != iter_end; ++iter) {
          std::cout << iter->getVertex() << "\n";
        }
      }
      {
        std::cout << "The nodes using a vertex iterator:\n";
        VI iter = x.getVerticesBeginning();
        const VI iter_end = x.getVerticesEnd();
        for (; iter != iter_end; ++iter) {
          std::cout << *iter << "\n";
        }
      }
      {
        std::cout << "The node identifiers:\n";
        NII iter = x.getNodeIdentifiersBeginning();
        const NII iter_end = x.getNodeIdentifiersEnd();
        for (; iter != iter_end; ++iter) {
          std::cout << *iter << "\n";
        }
      }

      // Iterate over the cells.
      {
        std::cout << "The nodes of the cells:\n";
        CCI c = x.getCellsBeginning();
        const CCI c_end = x.getCellsEnd();
        for (; c != c_end; ++c) {
          for (std::size_t i = 0; i != 4; ++i) {
            std::cout << c->getNode(i)->getVertex() << "   ";
          }
          std::cout << "\n";
        }
      }

      // Iterate over the indexed simplices.
      {
        std::cout << "The indexed simplices:\n";
        x.setNodeIdentifiers();
        ISI i = x.getIndexedSimplicesBeginning();
        ISI i_end = x.getIndexedSimplicesEnd();
        for (; i != i_end; ++i) {
          std::cout << *i << "\n";
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
          for (std::size_t i = 1; i != 4; ++i) {
            std::cout << (*f).first->getNode((vi + i) % 4)->getVertex() << "   ";
          }
          std::cout << "\n";
        }
      }
      // Count the faces.
      assert(x.computeFacesSize() == 20);
      // Count the boundary faces.
      {
        std::size_t count = 0;
        FI f = x.getFacesBeginning();
        const FI f_end = x.getFacesEnd();
        for (; f != f_end; ++f) {
          if (isOnBoundary<SM>(f)) {
            ++count;
          }
        }
        assert(count == 8);
      }
    }

  } // 3-D space, 3-D simplex.

  return 0;
}
