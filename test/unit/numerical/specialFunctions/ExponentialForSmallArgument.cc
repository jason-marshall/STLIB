// -*- C++ -*-

#include "stlib/numerical/specialFunctions/ExponentialForSmallArgument.h"

#include <iostream>
#include <cassert>

using namespace stlib;

int
main()
{
  {
    const double Epsilon = std::sqrt(std::numeric_limits<double>::epsilon());
    numerical::ExponentialForSmallArgument<> e;
    assert(std::abs(e(0.0) - std::exp(0.0)) < 2 * Epsilon);
    assert(std::abs(e(1e-16) - std::exp(1e-16)) < 2 * Epsilon);
    assert(std::abs(e(1e-15) - std::exp(1e-15)) < 2 * Epsilon);
    assert(std::abs(e(1e-14) - std::exp(1e-14)) < 2 * Epsilon);
    assert(std::abs(e(1e-13) - std::exp(1e-13)) < 2 * Epsilon);
    assert(std::abs(e(1e-12) - std::exp(1e-12)) < 2 * Epsilon);
    assert(std::abs(e(1e-11) - std::exp(1e-11)) < 2 * Epsilon);
    assert(std::abs(e(1e-10) - std::exp(1e-10)) < 2 * Epsilon);
    assert(std::abs(e(1e-9) - std::exp(1e-9)) < 2 * Epsilon);
    assert(std::abs(e(1e-8) - std::exp(1e-8)) < 2 * Epsilon);
    assert(std::abs(e(1e-7) - std::exp(1e-7)) < 2 * Epsilon);
    assert(std::abs(e(1e-6) - std::exp(1e-6)) < 2 * Epsilon);
    assert(std::abs(e(1e-5) - std::exp(1e-5)) < 2 * Epsilon);
    assert(std::abs(e(1e-4) - std::exp(1e-4)) < 2 * Epsilon);
    assert(std::abs(e(1e-3) - std::exp(1e-3)) < 2 * Epsilon);
    assert(std::abs(e(1e-2) - std::exp(1e-2)) < 2 * Epsilon);
    assert(std::abs(e(1e-1) - std::exp(1e-1)) < 2 * Epsilon);
    assert(std::abs(e(-1e-16) - std::exp(-1e-16)) < 2 * Epsilon);
    assert(std::abs(e(-1e-15) - std::exp(-1e-15)) < 2 * Epsilon);
    assert(std::abs(e(-1e-14) - std::exp(-1e-14)) < 2 * Epsilon);
    assert(std::abs(e(-1e-13) - std::exp(-1e-13)) < 2 * Epsilon);
    assert(std::abs(e(-1e-12) - std::exp(-1e-12)) < 2 * Epsilon);
    assert(std::abs(e(-1e-11) - std::exp(-1e-11)) < 2 * Epsilon);
    assert(std::abs(e(-1e-10) - std::exp(-1e-10)) < 2 * Epsilon);
    assert(std::abs(e(-1e-9) - std::exp(-1e-9)) < 2 * Epsilon);
    assert(std::abs(e(-1e-8) - std::exp(-1e-8)) < 2 * Epsilon);
    assert(std::abs(e(-1e-7) - std::exp(-1e-7)) < 2 * Epsilon);
    assert(std::abs(e(-1e-6) - std::exp(-1e-6)) < 2 * Epsilon);
    assert(std::abs(e(-1e-5) - std::exp(-1e-5)) < 2 * Epsilon);
    assert(std::abs(e(-1e-4) - std::exp(-1e-4)) < 2 * Epsilon);
    assert(std::abs(e(-1e-3) - std::exp(-1e-3)) < 2 * Epsilon);
    assert(std::abs(e(-1e-2) - std::exp(-1e-2)) < 2 * Epsilon);
    assert(std::abs(e(-1e-1) - std::exp(-1e-1)) < 2 * Epsilon);
    e = numerical::constructExponentialForSmallArgument<double>();
  }
  {
    const float Epsilon = std::sqrt(std::numeric_limits<float>::epsilon());
    numerical::ExponentialForSmallArgument<float> e;
    assert(std::abs(e(0.0) - std::exp(0.0)) < 2 * Epsilon);
    assert(std::abs(e(1e-16) - std::exp(1e-16)) < 2 * Epsilon);
    assert(std::abs(e(1e-15) - std::exp(1e-15)) < 2 * Epsilon);
    assert(std::abs(e(1e-14) - std::exp(1e-14)) < 2 * Epsilon);
    assert(std::abs(e(1e-13) - std::exp(1e-13)) < 2 * Epsilon);
    assert(std::abs(e(1e-12) - std::exp(1e-12)) < 2 * Epsilon);
    assert(std::abs(e(1e-11) - std::exp(1e-11)) < 2 * Epsilon);
    assert(std::abs(e(1e-10) - std::exp(1e-10)) < 2 * Epsilon);
    assert(std::abs(e(1e-9) - std::exp(1e-9)) < 2 * Epsilon);
    assert(std::abs(e(1e-8) - std::exp(1e-8)) < 2 * Epsilon);
    assert(std::abs(e(1e-7) - std::exp(1e-7)) < 2 * Epsilon);
    assert(std::abs(e(1e-6) - std::exp(1e-6)) < 2 * Epsilon);
    assert(std::abs(e(1e-5) - std::exp(1e-5)) < 2 * Epsilon);
    assert(std::abs(e(1e-4) - std::exp(1e-4)) < 2 * Epsilon);
    assert(std::abs(e(1e-3) - std::exp(1e-3)) < 2 * Epsilon);
    assert(std::abs(e(1e-2) - std::exp(1e-2)) < 2 * Epsilon);
    assert(std::abs(e(1e-1) - std::exp(1e-1)) < 2 * Epsilon);
    assert(std::abs(e(-1e-16) - std::exp(-1e-16)) < 2 * Epsilon);
    assert(std::abs(e(-1e-15) - std::exp(-1e-15)) < 2 * Epsilon);
    assert(std::abs(e(-1e-14) - std::exp(-1e-14)) < 2 * Epsilon);
    assert(std::abs(e(-1e-13) - std::exp(-1e-13)) < 2 * Epsilon);
    assert(std::abs(e(-1e-12) - std::exp(-1e-12)) < 2 * Epsilon);
    assert(std::abs(e(-1e-11) - std::exp(-1e-11)) < 2 * Epsilon);
    assert(std::abs(e(-1e-10) - std::exp(-1e-10)) < 2 * Epsilon);
    assert(std::abs(e(-1e-9) - std::exp(-1e-9)) < 2 * Epsilon);
    assert(std::abs(e(-1e-8) - std::exp(-1e-8)) < 2 * Epsilon);
    assert(std::abs(e(-1e-7) - std::exp(-1e-7)) < 2 * Epsilon);
    assert(std::abs(e(-1e-6) - std::exp(-1e-6)) < 2 * Epsilon);
    assert(std::abs(e(-1e-5) - std::exp(-1e-5)) < 2 * Epsilon);
    assert(std::abs(e(-1e-4) - std::exp(-1e-4)) < 2 * Epsilon);
    assert(std::abs(e(-1e-3) - std::exp(-1e-3)) < 2 * Epsilon);
    assert(std::abs(e(-1e-2) - std::exp(-1e-2)) < 2 * Epsilon);
    assert(std::abs(e(-1e-1) - std::exp(-1e-1)) < 2 * Epsilon);
    e = numerical::constructExponentialForSmallArgument<float>();
  }

  return 0;
}
