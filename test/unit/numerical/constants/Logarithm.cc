// -*- C++ -*-

#include "stlib/numerical/constants/Logarithm.h"

#include <cstddef>

using namespace stlib;

int
main()
{
  static_assert((numerical::Logarithm<std::size_t, 2, 1>::Result) == 0,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 2>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 3>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 4>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 5>::Result) == 3,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 6>::Result) == 3,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 7>::Result) == 3,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 8>::Result) == 3,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 9>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 10>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 11>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 12>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 13>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 14>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 15>::Result) == 4,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 2, 16>::Result) == 4,
                "Problem.");

  static_assert((numerical::Logarithm<std::size_t, 3, 1>::Result) == 0,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 2>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 3>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 4>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 5>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 6>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 7>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 8>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 3, 9>::Result) == 2,
                "Problem.");

  static_assert((numerical::Logarithm<std::size_t, 4, 1>::Result) == 0,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 2>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 3>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 4>::Result) == 1,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 5>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 6>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 7>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 8>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 9>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 10>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 11>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 12>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 13>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 14>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 15>::Result) == 2,
                "Problem.");
  static_assert((numerical::Logarithm<std::size_t, 4, 16>::Result) == 2,
                "Problem.");

  return 0;
}
