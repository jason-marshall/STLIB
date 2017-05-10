// -*- C++ -*-

#include "stlib/ads/array/ArrayWithNullHoles.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  typedef ads::ArrayWithNullHoles<int> Array;

  {
    Array x(-1);
    assert(x.isValid());
    assert(x.size() == 0);
    assert(x.sizeNull() == 0);
    assert(x.sizeNonNull() == 0);

    x.insert(0);
    x.insert(1);
    x.insert(2);
    x.insert(3);

    assert(x.isValid());
    assert(x.size() == 4);
    assert(x.sizeNull() == 0);
    assert(x.sizeNonNull() == 4);
    assert(x.get(0) == 0);
    assert(x.get(1) == 1);
    assert(x.get(2) == 2);
    assert(x.get(3) == 3);

    x.erase(1);
    x.erase(3);

    assert(x.isValid());
    assert(x.size() == 4);
    assert(x.sizeNull() == 2);
    assert(x.sizeNonNull() == 2);
    assert(x.get(0) == 0);
    assert(x.get(1) == -1);
    assert(x.get(2) == 2);
    assert(x.get(3) == -1);

    x.insert(3);
    x.insert(1);

    assert(x.isValid());
    assert(x.size() == 4);
    assert(x.sizeNull() == 0);
    assert(x.sizeNonNull() == 4);
    assert(x.get(0) == 0);
    assert(x.get(1) == 1);
    assert(x.get(2) == 2);
    assert(x.get(3) == 3);

    x.set(0, 3);
    x.set(1, 2);
    x.set(2, 1);
    x.set(3, 0);

    assert(x.isValid());
    assert(x.size() == 4);
    assert(x.sizeNull() == 0);
    assert(x.sizeNonNull() == 4);
    assert(x.get(0) == 3);
    assert(x.get(1) == 2);
    assert(x.get(2) == 1);
    assert(x.get(3) == 0);
  }

  return 0;
}
