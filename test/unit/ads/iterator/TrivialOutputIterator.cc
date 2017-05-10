// -*- C++ -*-

#include "stlib/ads/iterator/TrivialOutputIterator.h"

#include <cassert>

using namespace stlib;

int
main()
{
  {
    ads::TrivialOutputIterator x;

    ++x;
    x++;
    *x = 1;
    *x = 3.45;
    *x++ = 'a';

    ads::TrivialOutputIterator y(x);
    x = y;
    ads::TrivialOutputIterator z = ads::constructTrivialOutputIterator();
    *z++ = 0;
  }

  {
    int count = 0;
    ads::TrivialOutputIteratorCount x(count);

    assert(x.get() == 0);
    ++x;
    x++;
    *x = 1;
    *x = 3.45;
    *x++ = 'a';
    assert(x.get() == 3);

    ads::TrivialOutputIteratorCount y(x);
    assert(y.get() == 3);
    ads::TrivialOutputIteratorCount z =
      ads::constructTrivialOutputIteratorCount(count);
    assert(z.get() == 3);

    x.reset();
    assert(x.get() == 0);
    assert(y.get() == 0);
    assert(z.get() == 0);
  }

  return 0;
}
