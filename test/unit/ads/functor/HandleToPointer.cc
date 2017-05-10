// -*- C++ -*-

#include "stlib/ads/functor/HandleToPointer.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    int x = 7;
    {
      ads::HandleToPointer<int*> d;
      assert(d(&x) == &x);
    }
    {
      ads::HandleToPointer<const int*> d;
      assert(d(&x) == &x);
    }


    {
      ads::HandleToPointer<int*, int*> d;
      assert(d(&x) == &x);
    }
    {
      ads::HandleToPointer<const int*, const int*> d;
      assert(d(&x) == &x);
    }
    // One cannot do this.
    /*
    {
      ads::HandleToPointer<const int*, int*> d;
      assert(d(&x) == &x);
    }
    */
    {
      ads::HandleToPointer<int*, const int*> d;
      assert(d(&x) == &x);
    }


    {
      assert(ads::handle_to_pointer<int*>()(&x) == &x);
    }
  }

  return 0;
}
