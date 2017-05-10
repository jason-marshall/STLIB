// -*- C++ -*-

#include "stlib/ads/functor/compare_handle.h"

#include <vector>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  //
  // Pointers.
  //
  {
    int five = 5, go = 5, ten = 10;

    EqualToByHandle<int*> equalTo;
    NotEqualToByHandle<int*> notEqualTo;
    GreaterByHandle<int*> greater;
    LessByHandle<int*> less;
    GreaterEqualByHandle<int*> greaterEqual;
    LessEqualByHandle<int*> lessEqual;

    assert(equalTo(&five, &go));
    assert(! equalTo(&five, &ten));

    assert(notEqualTo(&five, &ten));
    assert(! notEqualTo(&five, &go));

    assert(greater(&ten, &five));
    assert(! greater(&five, &ten));
    assert(! greater(&five, &go));

    assert(less(&five, &ten));
    assert(! less(&ten, &five));
    assert(! less(&five, &go));

    assert(greaterEqual(&ten, &five));
    assert(! greaterEqual(&five, &ten));
    assert(greaterEqual(&five, &go));

    assert(lessEqual(&five, &ten));
    assert(! lessEqual(&ten, &five));
    assert(lessEqual(&five, &go));
  }

  //
  // vector iterators.
  //
  {
    typedef std::vector<int> container;
    typedef container::const_iterator handle;
    container v(3);
    v[0] = 5;
    v[1] = 5;
    v[2] = 10;
    handle five = v.begin();
    handle go = v.begin() + 1;
    handle ten = v.begin() + 2;

    EqualToByHandle<handle> equalTo;
    NotEqualToByHandle<handle> notEqualTo;
    GreaterByHandle<handle> greater;
    LessByHandle<handle> less;
    GreaterEqualByHandle<handle> greaterEqual;
    LessEqualByHandle<handle> lessEqual;

    assert(equalTo(five, go));
    assert(! equalTo(five, ten));

    assert(notEqualTo(five, ten));
    assert(! notEqualTo(five, go));

    assert(greater(ten, five));
    assert(! greater(five, ten));
    assert(! greater(five, go));

    assert(less(five, ten));
    assert(! less(ten, five));
    assert(! less(five, go));

    assert(greaterEqual(ten, five));
    assert(! greaterEqual(five, ten));
    assert(greaterEqual(five, go));

    assert(lessEqual(five, ten));
    assert(! lessEqual(ten, five));
    assert(lessEqual(five, go));
  }

  return 0;
}
