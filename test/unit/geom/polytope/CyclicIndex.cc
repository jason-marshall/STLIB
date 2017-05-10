// -*- C++ -*-

#include "stlib/geom/polytope/CyclicIndex.h"

#include <iostream>

using namespace stlib;

int
main()
{
  typedef geom::CyclicIndex<std::ptrdiff_t> CyclicIndex;

  {
    // sizeof
    std::cout << "sizeof(CyclicIndex(5)) = "
              << sizeof(CyclicIndex(5)) << '\n';
  }
  {
    // default constructor
    CyclicIndex i(5);
    assert(i() == 0);
  }
  {
    // copy constructor
    CyclicIndex i(5);
    i.set(2);
    CyclicIndex j(i);
    assert(i() == j());
  }
  {
    // assignment operator
    CyclicIndex i(5);
    i.set(2);
    CyclicIndex j(5);
    j = i;
    assert(i() == j());
  }
  {
    // accessors and manipulators
    CyclicIndex i(5);
    assert(i() == 0);
    i.set(1);
    assert(i() == 1);
    i.set(6);
    assert(i() == 1);
    i.set(10);
    assert(i() == 0);
    i.set(-6);
    assert(i() == 4);
  }
  {
    // increment
    CyclicIndex i(5);
    assert(i() == 0);
    ++i;
    assert(i() == 1);
    ++i;
    assert(i() == 2);
    ++i;
    assert(i() == 3);
    ++i;
    assert(i() == 4);
    ++i;
    assert(i() == 0);
  }
  {
    // decrement
    CyclicIndex i(5);
    assert(i() == 0);
    --i;
    assert(i() == 4);
    --i;
    assert(i() == 3);
    --i;
    assert(i() == 2);
    --i;
    assert(i() == 1);
    --i;
    assert(i() == 0);
  }

  return 0;
}
