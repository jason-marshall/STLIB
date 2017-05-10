// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/SmrNode.h"

#include "stlib/geom/mesh/simplicial/insert.h"

#include <iostream>

#include <cassert>

using namespace stlib;

void
fcn(geom::SimpMeshRed<3, 3, double, geom::SmrNode>::NodeConstIterator /*v*/) {}

int
main()
{
  using namespace geom;

  {
    typedef SimpMeshRed<3, 3, double, SmrNode> SM;

    typedef SM::Node NT;
    typedef SM::NodeIterator NI;
    typedef SM::NodeConstIterator NCI;

    typedef SM::CellIterator CI;

    typedef SM::Vertex VT;

    {
      NI x;
      NCI y(x);
      assert(x == y);
      fcn(x);
      fcn(y);
    }
    {
      NI x;
      NCI y;
      y = x;
      assert(x == y);
    }


    //
    // Tests on a single vertex.
    //
    {
      SM mesh;
      CI c = mesh.insertCell();
      assert(c->getIdentifier() == 0);

      // Default constructor.
      {
        NT x;
        assert(x.getCellsSize() == 0);
      }

      // Vertex constructor.
      VT p = {{0, 1, 2}};
      NT x(p, 0);
      assert(x.getVertex() == p);
      assert(x.getCellsSize() == 0);

      // Copy constructor.
      {
        NT y(x);
        assert(x == y);
      }
      // Assignment operator.
      {
        NT y;
        y = x;
        assert(x == y);
      }
      // Manipulators.
      {
        NT y;
        y.setVertex(p);
        y.setIdentifier(0);
        assert(x == y);
      }
    }

    //
    // Tests on a mesh.
    //
    {
      // Make a cell.
      SM x;
      std::array<NT*, 4> v;
      for (std::size_t i = 0; i != 4; ++i) {
        v[i] = &*x.insertNode();
      }
      CI c = geom::insertCell(&x, v[0], v[1], v[2], v[3]);

      for (std::size_t i = 0; i != 4; ++i) {
        assert(v[i]->getCellsBeginning()->getIdentifier() ==
               c->getIdentifier());
      }
    }
  }

  return 0;
}
