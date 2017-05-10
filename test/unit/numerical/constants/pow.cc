// -*- C++ -*-

#include "stlib/numerical/constants/pow.h"

int
main()
{
  using stlib::numerical::pow;

// CONTINUE
// MSVS 2013 does not suppert constexpr.
#ifndef _MSC_VER

  //
  // int
  //

  // 0^0 is indeterminate. This result is incorrect.
  static_assert(pow(0, 0) == 1, "Problem.");

  static_assert(pow(0, 1) == 0, "Problem.");
  static_assert(pow(0, 2) == 0, "Problem.");
  static_assert(pow(0, 3) == 0, "Problem.");
  static_assert(pow(0, 100) == 0, "Problem.");

  static_assert(pow(1, 0) == 1, "Problem.");
  static_assert(pow(1, 1) == 1, "Problem.");
  static_assert(pow(1, 2) == 1, "Problem.");
  static_assert(pow(1, 3) == 1, "Problem.");
  static_assert(pow(1, 100) == 1, "Problem.");

  static_assert(pow(-1, 0) == 1, "Problem.");
  static_assert(pow(-1, 1) == -1, "Problem.");
  static_assert(pow(-1, 2) == 1, "Problem.");
  static_assert(pow(-1, 3) == -1, "Problem.");
  static_assert(pow(-1, 100) == 1, "Problem.");

  static_assert(pow(2, 0) == 1, "Problem.");
  static_assert(pow(2, 1) == 2, "Problem.");
  static_assert(pow(2, 2) == 4, "Problem.");
  static_assert(pow(2, 3) == 8, "Problem.");
  static_assert(pow(2, 8) == 256, "Problem.");
  static_assert(pow(2, 16) == 65536, "Problem.");
  static_assert(pow(2, 30) == 1073741824, "Problem.");
  //static_assert(pow(2, 31) == 2147483648, "Problem.");

  static_assert(pow(-2, 0) == 1, "Problem.");
  static_assert(pow(-2, 1) == -2, "Problem.");
  static_assert(pow(-2, 2) == 4, "Problem.");
  static_assert(pow(-2, 3) == -8, "Problem.");
  static_assert(pow(-2, 8) == 256, "Problem.");
  static_assert(pow(-2, 16) == 65536, "Problem.");
  static_assert(pow(-2, 30) == 1073741824, "Problem.");

  static_assert(pow(3, 0) == 1, "Problem.");
  static_assert(pow(3, 1) == 3, "Problem.");
  static_assert(pow(3, 2) == 9, "Problem.");
  static_assert(pow(3, 3) == 27, "Problem.");

  static_assert(pow(-3, 0) == 1, "Problem.");
  static_assert(pow(-3, 1) == -3, "Problem.");
  static_assert(pow(-3, 2) == 9, "Problem.");
  static_assert(pow(-3, 3) == -27, "Problem.");

  //
  // unsigned
  //

  static_assert(pow(unsigned(2), 31) == 2147483648U, "Problem.");

  //
  // double
  //

  // 0^0 is indeterminate. This result is incorrect.
  static_assert(pow(0., 0) == 1., "Problem.");

  static_assert(pow(0., 1) == 0., "Problem.");
  static_assert(pow(0., 2) == 0., "Problem.");
  static_assert(pow(0., 3) == 0., "Problem.");
  static_assert(pow(0., 100) == 0., "Problem.");

  static_assert(pow(1., 0) == 1., "Problem.");
  static_assert(pow(1., 1) == 1., "Problem.");
  static_assert(pow(1., 2) == 1., "Problem.");
  static_assert(pow(1., 3) == 1., "Problem.");
  static_assert(pow(1., 100) == 1., "Problem.");

  static_assert(pow(-1., 0) == 1., "Problem.");
  static_assert(pow(-1., 1) == -1., "Problem.");
  static_assert(pow(-1., 2) == 1., "Problem.");
  static_assert(pow(-1., 3) == -1., "Problem.");
  static_assert(pow(-1., 100) == 1., "Problem.");

  static_assert(pow(2., 0) == 1., "Problem.");
  static_assert(pow(2., 1) == 2., "Problem.");
  static_assert(pow(2., 2) == 4., "Problem.");
  static_assert(pow(2., 3) == 8., "Problem.");
  static_assert(pow(2., 8) == 256., "Problem.");
  static_assert(pow(2., 16) == 65536., "Problem.");
  static_assert(pow(2., 30) == 1073741824., "Problem.");
  //static_assert(pow(2., 31) == 2147483648., "Problem.");

  static_assert(pow(-2., 0) == 1., "Problem.");
  static_assert(pow(-2., 1) == -2., "Problem.");
  static_assert(pow(-2., 2) == 4., "Problem.");
  static_assert(pow(-2., 3) == -8., "Problem.");
  static_assert(pow(-2., 8) == 256., "Problem.");
  static_assert(pow(-2., 16) == 65536., "Problem.");
  static_assert(pow(-2., 30) == 1073741824., "Problem.");

  static_assert(pow(3., 0) == 1., "Problem.");
  static_assert(pow(3., 1) == 3., "Problem.");
  static_assert(pow(3., 2) == 9., "Problem.");
  static_assert(pow(3., 3) == 27., "Problem.");

  static_assert(pow(-3., 0) == 1., "Problem.");
  static_assert(pow(-3., 1) == -3., "Problem.");
  static_assert(pow(-3., 2) == 9., "Problem.");
  static_assert(pow(-3., 3) == -27., "Problem.");

#endif

  return 0;
}
