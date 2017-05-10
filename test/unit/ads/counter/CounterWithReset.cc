// -*- C++ -*-

#include "stlib/ads/counter/CounterWithReset.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    //
    // std::ptrdiff_t
    //
    typedef std::ptrdiff_t Integer;
    {
      // Default constructor.
      ads::CounterWithReset<Integer> a;
      assert(a.getReset() == 0);
    }
    {
      // Construct from the reset value.
      const Integer reset = 1;
      ads::CounterWithReset<Integer> a(reset);
      assert(a.getReset() == reset);
      assert(a() == reset);

      a.reset();
      ++a;
      assert(a() == reset + 1);

      a.reset();
      --a;
      assert(a() == reset - 1);

      a.reset();
      a += 2;
      assert(a() == reset + 2);

      a.reset();
      a -= 2;
      assert(a() == reset - 2);

      a.setReset(10);
      assert(a.getReset() == 10);

      {
        // Copy constructor.
        ads::CounterWithReset<Integer> b(a);
        assert(a == b);
      }
      {
        // Assignment operator.
        ads::CounterWithReset<Integer> b;
        b = a;
        assert(a == b);
      }
    }
  }
  {
    //
    // std::size_t
    //
    typedef std::size_t Integer;
    {
      // Default constructor.
      ads::CounterWithReset<Integer> a;
      assert(a.getReset() == 0);
    }
    {
      // Construct from the reset value.
      const Integer reset = 1;
      ads::CounterWithReset<Integer> a(reset);
      assert(a.getReset() == reset);
      assert(a() == reset);

      a.reset();
      ++a;
      assert(a() == reset + 1);

      a.reset();
      --a;
      assert(a() == reset - 1);

      a.reset();
      a += 2;
      assert(a() == reset + 2);

      a.reset();
      a -= 2;
      assert(a() == reset - 2);

      a.setReset(10);
      assert(a.getReset() == 10);

      {
        // Copy constructor.
        ads::CounterWithReset<Integer> b(a);
        assert(a == b);
      }
      {
        // Assignment operator.
        ads::CounterWithReset<Integer> b;
        b = a;
        assert(a == b);
      }
    }
  }
  {
    //
    // int
    //
    typedef int Integer;
    {
      // Default constructor.
      ads::CounterWithReset<Integer> a;
      assert(a.getReset() == 0);
    }
    {
      // Construct from the reset value.
      const int reset = 1;
      ads::CounterWithReset<Integer> a(reset);
      assert(a.getReset() == reset);
      assert(a() == reset);

      a.reset();
      ++a;
      assert(a() == reset + 1);

      a.reset();
      --a;
      assert(a() == reset - 1);

      a.reset();
      a += 2;
      assert(a() == reset + 2);

      a.reset();
      a -= 2;
      assert(a() == reset - 2);

      a.setReset(10);
      assert(a.getReset() == 10);

      {
        // Copy constructor.
        ads::CounterWithReset<Integer> b(a);
        assert(a == b);
      }
      {
        // Assignment operator.
        ads::CounterWithReset<Integer> b;
        b = a;
        assert(a == b);
      }
    }
  }

  return 0;
}
