// -*- C++ -*-

#include "stlib/ads/algorithm/OrderedPair.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    typedef ads::OrderedPair<int> T;
    {
      // Default constructor.
      T x;
    }
    {
      // Element constructor.
      T x(1, 2);
      assert(x.getFirst() == 1 && x.getSecond() == 2);
      // Copy constructor.
      {
        T y(x);
        assert(x == y);
      }
      // Assignment operator.
      {
        T y;
        y = x;
        assert(x == y);
      }
    }
    {
      // Construct from different types.
      T x('a', true);
    }
    {
      // Accessors.
      T x(3, 2);
      assert(x.getFirst() == 2 && x.getSecond() == 3);
    }
    {
      // Manipulators.
      T x(1, 2);
      assert(x.getFirst() == 1 && x.getSecond() == 2);
      x.set(2, 3);
      assert(x.getFirst() == 2 && x.getSecond() == 3);
      x.set(5, 3);
      assert(x.getFirst() == 3 && x.getSecond() == 5);
      x.setFirst(7);
      assert(x.getFirst() == 5 && x.getSecond() == 7);
      x.setSecond(3);
      assert(x.getFirst() == 3 && x.getSecond() == 5);
    }
    {
      // ==
      {
        T x(2, 3);
        T y(2, 3);
        assert(x == y);
      }
      {
        T x(2, 3);
        T y(1, 3);
        assert(!(x == y));
      }
      {
        T x(2, 3);
        T y(2, 4);
        assert(!(x == y));
      }
    }

    {
      // <
      {
        T x(2, 3);
        T y(2, 3);
        assert(!(x < y));
      }
      {
        T x(1, 3);
        T y(2, 3);
        assert(x < y);
      }
      {
        T x(2, 1);
        T y(2, 3);
        assert(x < y);
      }
    }

    {
      // !=
      {
        T x(2, 3);
        T y(2, 3);
        assert(!(x != y));
      }
      {
        T x(2, 3);
        T y(1, 3);
        assert(x != y);
      }
      {
        T x(2, 3);
        T y(2, 4);
        assert(x != y);
      }
    }

    {
      // >
      {
        T x(2, 3);
        T y(2, 3);
        assert(!(x > y));
      }
      {
        T x(2, 3);
        T y(1, 3);
        assert(x > y);
      }
      {
        T x(2, 3);
        T y(2, 2);
        assert(x > y);
      }
    }

    {
      // <=
      {
        T x(2, 3);
        T y(2, 3);
        assert(x <= y);
      }
      {
        T x(1, 3);
        T y(2, 3);
        assert(x <= y);
      }
      {
        T x(2, 1);
        T y(2, 3);
        assert(x <= y);
      }
    }

    {
      // >=
      {
        T x(2, 3);
        T y(2, 3);
        assert(x >= y);
      }
      {
        T x(2, 3);
        T y(1, 3);
        assert(x >= y);
      }
      {
        T x(2, 3);
        T y(2, 2);
        assert(x >= y);
      }
    }

    {
      // makeOrderedPair()
      T x = ads::makeOrderedPair(2, 3);
      T y(2, 3);
      assert(x == y);
    }
  }
}
