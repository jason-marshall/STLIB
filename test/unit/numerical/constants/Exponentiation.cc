// -*- C++ -*-

#include "stlib/numerical/constants/Exponentiation.h"

#include <limits>

#include <cassert>

using namespace stlib;

int
main()
{
  using numerical::Exponentiation;

  // 0^0 is indeterminate.
  // static_assert((Exponentiation<int, 0, 0>::Result) == 0, "Problem.");

  static_assert((Exponentiation<int, 0, 1>::Result) == 0, "Problem.");
  static_assert((Exponentiation<int, 0, 2>::Result) == 0, "Problem.");
  static_assert((Exponentiation<int, 0, 3>::Result) == 0, "Problem.");
  static_assert((Exponentiation<int, 0, 100>::Result) == 0, "Problem.");

  static_assert((Exponentiation<int, 1, 0>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 1, 1>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 1, 2>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 1, 3>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 1, 100>::Result) == 1, "Problem.");

  static_assert((Exponentiation < int, -1, 0 >::Result) == 1, "Problem.");
  static_assert((Exponentiation < int, -1, 1 >::Result) == -1, "Problem.");
  static_assert((Exponentiation < int, -1, 2 >::Result) == 1, "Problem.");
  static_assert((Exponentiation < int, -1, 3 >::Result) == -1, "Problem.");
  static_assert((Exponentiation < int, -1, 100 >::Result) == 1, "Problem.");

  static_assert((Exponentiation<int, 2, 0>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 2, 1>::Result) == 2, "Problem.");
  static_assert((Exponentiation<int, 2, 2>::Result) == 4, "Problem.");
  static_assert((Exponentiation<int, 2, 3>::Result) == 8, "Problem.");
  static_assert((Exponentiation<int, 2, 8>::Result) == 256, "Problem.");
  static_assert((Exponentiation<int, 2, 16>::Result) == 65536, "Problem.");
  static_assert((Exponentiation<int, 2, 30>::Result) == 1073741824, "Problem.");
  //static_assert((Exponentiation<int, 2, 31>::Result) == 2147483648, "Problem.");

  static_assert((Exponentiation < int, -2, 0 >::Result) == 1, "Problem.");
  static_assert((Exponentiation < int, -2, 1 >::Result) == -2, "Problem.");
  static_assert((Exponentiation < int, -2, 2 >::Result) == 4, "Problem.");
  static_assert((Exponentiation < int, -2, 3 >::Result) == -8, "Problem.");
  static_assert((Exponentiation < int, -2, 8 >::Result) == 256, "Problem.");
  static_assert((Exponentiation < int, -2, 16 >::Result) == 65536, "Problem.");
  static_assert((Exponentiation < int, -2, 30 >::Result) == 1073741824,
                    "Problem.");

  static_assert((Exponentiation<int, 3, 0>::Result) == 1, "Problem.");
  static_assert((Exponentiation<int, 3, 1>::Result) == 3, "Problem.");
  static_assert((Exponentiation<int, 3, 2>::Result) == 9, "Problem.");
  static_assert((Exponentiation<int, 3, 3>::Result) == 27, "Problem.");

  static_assert((Exponentiation < int, -3, 0 >::Result) == 1, "Problem.");
  static_assert((Exponentiation < int, -3, 1 >::Result) == -3, "Problem.");
  static_assert((Exponentiation < int, -3, 2 >::Result) == 9, "Problem.");
  static_assert((Exponentiation < int, -3, 3 >::Result) == -27, "Problem.");

  static_assert((Exponentiation<unsigned, 2, 31>::Result) == 2147483648U,
                    "Problem.");

  return 0;
}
