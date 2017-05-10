// -*- C++ -*-

#include "stlib/geom/mesh/structured_grid/StructuredGrid.h"

#include <iostream>
#include <sstream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 2-D space.  2-D grid.
  //
  {
    typedef geom::StructuredGrid<2, 2> Grid;

    {
      Grid g;
      assert(g.getSpaceDimension() == 2);
      assert(g.getGridDimension() == 2);
      assert(g.getExtents()[0] == 0 && g.getExtents()[1] == 0);
      assert(g.getBeginning() == g.getEnd());

      std::istringstream in("2 2\n"\
                            "0.0 0.0\n"\
                            "1.0 0.0\n"\
                            "0.0 1.0\n"\
                            "1.0 1.0\n");
      in >> g;
      assert(g.getSpaceDimension() == 2);
      assert(g.getGridDimension() == 2);
      std::cout << g;
      assert(g.getExtents()[0] == 2 && g.getExtents()[1] == 2);
      assert(g.getBeginning() + 4 == g.getEnd());
      std::cout << "The quad mesh of a square:\n" << g;
      std::cout << "The triangle mesh of a square:\n";
      g.writeIndexedSimplexSet(std::cout);
    }
  }

  return 0;
}
