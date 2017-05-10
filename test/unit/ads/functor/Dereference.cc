// -*- C++ -*-

#include "stlib/ads/functor/Dereference.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    int x = 7;
    {
      ads::Dereference<int*> d;
      assert(d(&x) == x);
    }
    {
      ads::Dereference<const int*> d;
      assert(d(&x) == x);
    }


    {
      ads::Dereference<int*, int> d;
      assert(d(&x) == x);
    }
    {
      ads::Dereference<const int*, int> d;
      assert(d(&x) == x);
    }
    {
      ads::Dereference<const int*, int> d;
      assert(d(&x) == x);
    }
    {
      ads::Dereference<int*, int> d;
      assert(d(&x) == x);
    }


    {
      ads::Dereference<int*, int&> d;
      assert(d(&x) == x);
    }
    {
      ads::Dereference<const int*, const int&> d;
      assert(d(&x) == x);
    }
    {
      // One cannot do this.
      /*
      ads::Dereference<const int*, int&> d;
      assert(d(&x) == x);
      */
    }
    {
      ads::Dereference<int*, const int&> d;
      assert(d(&x) == x);
    }


    {
      assert(ads::dereference<int*>()(&x) == x);
    }
  }

  return 0;
}
