// -*- C++ -*-

#include "stlib/ads/algorithm/Triplet.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    typedef ads::Triplet<int, float, double> T;
    {
      // Default constructor.
      T x;
    }
    {
      // Element constructor.
      int a = 3;
      float b = 3.4f;
      double c = 3.45;
      T x(a, b, c);
      assert(x.first == a && x.second == b && x.third == c);
    }

    {
      // ==
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(x == y);
      }
      {
        T x(2, 3, 5);
        T y(1, 3, 5);
        assert(!(x == y));
      }
      {
        T x(2, 3, 5);
        T y(2, 1, 5);
        assert(!(x == y));
      }
      {
        T x(2, 3, 5);
        T y(2, 3, 1);
        assert(!(x == y));
      }
    }

    {
      // <
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(!(x < y));
      }
      {
        T x(1, 3, 5);
        T y(2, 3, 5);
        assert(x < y);
      }
      {
        T x(2, 1, 5);
        T y(2, 3, 5);
        assert(x < y);
      }
      {
        T x(2, 3, 1);
        T y(2, 3, 5);
        assert(x < y);
      }
    }

    {
      // !=
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(!(x != y));
      }
      {
        T x(2, 3, 5);
        T y(1, 3, 5);
        assert(x != y);
      }
      {
        T x(2, 3, 5);
        T y(2, 1, 5);
        assert(x != y);
      }
      {
        T x(2, 3, 5);
        T y(2, 3, 1);
        assert(x != y);
      }
    }

    {
      // >
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(!(x > y));
      }
      {
        T x(2, 3, 5);
        T y(1, 3, 5);
        assert(x > y);
      }
      {
        T x(2, 3, 5);
        T y(2, 1, 5);
        assert(x > y);
      }
      {
        T x(2, 3, 5);
        T y(2, 3, 1);
        assert(x > y);
      }
    }

    {
      // <=
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(x <= y);
      }
      {
        T x(1, 3, 5);
        T y(2, 3, 5);
        assert(x <= y);
      }
      {
        T x(2, 1, 5);
        T y(2, 3, 5);
        assert(x <= y);
      }
      {
        T x(2, 3, 1);
        T y(2, 3, 5);
        assert(x <= y);
      }
    }

    {
      // >=
      {
        T x(2, 3, 5);
        T y(2, 3, 5);
        assert(x >= y);
      }
      {
        T x(2, 3, 5);
        T y(1, 3, 5);
        assert(x >= y);
      }
      {
        T x(2, 3, 5);
        T y(2, 1, 5);
        assert(x >= y);
      }
      {
        T x(2, 3, 5);
        T y(2, 3, 1);
        assert(x >= y);
      }
    }

    {
      // make_triplet()
      T x = ads::makeTriplet(2, 3.f, 5.);
      T y(2, 3, 5);
      assert(x == y);
    }
  }
}
