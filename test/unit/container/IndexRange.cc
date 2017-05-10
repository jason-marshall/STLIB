// -*- C++ -*-

#include "stlib/container/IndexRange.h"

#include <sstream>

using namespace stlib;

int
main()
{
  {
    typedef container::IndexRange IndexRange;
    typedef IndexRange::size_type size_type;
    typedef IndexRange::Index Index;

    {
      size_type extent = 0;
      Index base = 0;
      Index step = 1;
      IndexRange x(extent);
      assert(x.extent() == extent);
      assert(x.base() == base);
      assert(x.step() == step);
    }
    {
      size_type extent = 0;
      Index base = -2;
      Index step = 1;
      IndexRange x(extent, base);
      assert(x.extent() == extent);
      assert(x.base() == base);
      assert(x.step() == step);
    }
    {
      size_type extent = 0;
      Index base = -2;
      Index step = 3;
      IndexRange x(extent, base, step);
      assert(x.extent() == extent);
      assert(x.base() == base);
      assert(x.step() == step);

      {
        IndexRange y = x;
        assert(x == y);
      }
      {
        IndexRange y;
        y = x;
        assert(x == y);
      }
      {
        std::ostringstream out;
        out << x;
        IndexRange y;
        std::istringstream in(out.str());
        in >> y;
        assert(x == y);
      }
    }

    // Overlap.
    {
      IndexRange x(5);
      IndexRange y(3);
      IndexRange z(3);
      assert(z == overlap(x, y));
    }
    {
      IndexRange x(5);
      IndexRange y(3, -7);
      IndexRange z(0, 0);
      assert(z == overlap(x, y));
    }
    {
      IndexRange x(5);
      IndexRange y(3, -1);
      IndexRange z(2, 0);
      assert(z == overlap(x, y));
    }
    {
      IndexRange x(5);
      IndexRange y(3, 1);
      IndexRange z(3, 1);
      assert(z == overlap(x, y));
    }
    {
      IndexRange x(5);
      IndexRange y(3, 3);
      IndexRange z(2, 3);
      assert(z == overlap(x, y));
    }
    {
      IndexRange x(5);
      IndexRange y(3, 5);
      IndexRange z(0, 5);
      assert(z == overlap(x, y));
    }

    // isIn.
    {
      IndexRange range(5);
      assert(isIn(range, 0));
      assert(isIn(range, 4));
      assert(! isIn(range, -1));
      assert(! isIn(range, 5));
    }
    {
      IndexRange range(5, -1);
      assert(isIn(range, -1));
      assert(isIn(range, 3));
      assert(! isIn(range, -2));
      assert(! isIn(range, 4));
    }
    {
      IndexRange range(5, 0, 2);
      assert(isIn(range, 0));
      assert(isIn(range, 2));
      assert(isIn(range, 8));
      assert(! isIn(range, -1));
      assert(! isIn(range, 1));
      assert(! isIn(range, 10));
    }
  }
  return 0;
}
