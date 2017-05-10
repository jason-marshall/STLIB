// -*- C++ -*-

#include "limitsTests.h"

#include <cassert>

void
test()
{
  assert(limitsFloat());
}


int
main()
{
  assert(std::numeric_limits<float>::max() <
         std::numeric_limits<float>::infinity());
  assert(std::numeric_limits<double>::max() <
         std::numeric_limits<double>::infinity());
  test();

  return 0;
}
