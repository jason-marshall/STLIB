// -*- C++ -*-

#include "stlib/ads/functor/Address.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    int x = 7;
    {
      ads::Address<int> add;
      assert(add(x) == &x);
    }
    {
      assert(ads::address<int>()(x) == &x);
    }
  }
  {
    const int x = 7;
    {
      ads::Address<const int> add;
      assert(add(x) == &x);
    }
    {
      assert(ads::address<const int>()(x) == &x);
    }
  }

  return 0;
}
