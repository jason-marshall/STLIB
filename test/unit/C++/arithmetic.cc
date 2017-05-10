// -*- C++ -*-

#include <iostream>

#include <cassert>

int
main()
{

  {
    unsigned a = 2147483648U;
    unsigned b = 3U;
    unsigned c = a * b;
    assert(c == 2147483648U);
  }
  {
    unsigned a = 4294967295U;
    unsigned b = 5U;
    unsigned c = a + b;
    assert(c == 4U);
  }

  return 0;
}
