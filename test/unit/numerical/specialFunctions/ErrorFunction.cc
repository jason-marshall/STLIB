// -*- C++ -*-

#include "stlib/numerical/specialFunctions/ErrorFunction.h"
#include "stlib/numerical/equality.h"

#include <iostream>

using namespace stlib;

int
main()
{

  assert(numerical::areEqual(numerical::erf(0.), 0.));
  assert(numerical::areEqual
         (numerical::erf(std::numeric_limits<double>::max()), 1.));
  assert(numerical::areEqual(numerical::erf(-1.), -numerical::erf(1.)));


  assert(numerical::areEqual(numerical::erfc(0.), 1.));
  assert(numerical::areEqual
         (numerical::erfc(std::numeric_limits<double>::max()), 0.));
  assert(numerical::areEqual(numerical::erfc(-1.), 2. - numerical::erfc(1.)));

  return 0;
}
