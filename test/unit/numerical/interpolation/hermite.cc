// -*- C++ -*-

#include "stlib/numerical/interpolation/hermite.h"

//#include "stlib/ads/functor/compose.h"
#include "stlib/ext/functional.h"

#include <iostream>
#include <limits>

#include <cassert>
#include <cmath>

using namespace stlib;

int
main()
{
  const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

  //
  // Function
  //

  // f(x) = 1
  assert(std::abs(numerical::hermiteInterpolate(0., 1., 1., 0., 0.) - 1.) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(0.5, 1., 1., 0., 0.) - 1.) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(1., 1., 1., 0., 0.) - 1.) <
         Epsilon);

  // f(x) = x
  assert(std::abs(numerical::hermiteInterpolate(0., 0., 1., 1., 1.) - 0.) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(0.5, 0., 1., 1., 1.) - 0.5) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(1., 0., 1., 1., 1.) - 1.) <
         Epsilon);

  // f(x) = x^2
  assert(std::abs(numerical::hermiteInterpolate(0., 0., 1., 0., 2.) - 0.) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(0.5, 0., 1., 0., 2.) - 0.25) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(1., 0., 1., 0., 2.) - 1.) <
         Epsilon);

  // f(x) = x^3
  assert(std::abs(numerical::hermiteInterpolate(0., 0., 1., 0., 3.) - 0.) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(0.5, 0., 1., 0., 3.) - 0.125) <
         Epsilon);
  assert(std::abs(numerical::hermiteInterpolate(1., 0., 1., 0., 3.) - 1.) <
         Epsilon);

  //
  // Class.
  //

  std::cout << "Using Hermite:\n";
  {
    numerical::Hermite<> h(std::ptr_fun<double, double>(std::exp),
                           std::ptr_fun<double, double>(std::exp),
                           0.0, 2.0, 128);
    const double Arguments[] = {0, 0.3, 0.499, 0.5, 1.0, 1.99};
    const int Size = sizeof(Arguments) / sizeof(double);
    std::cout << "exp(x) on [0..2)\n"
              << "function, argument, approximation, difference\n";
    double x, a, b;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      a = std::exp(x);
      b = h(x);
      std::cout << x << ", " << a << ", " << b << ", " << a - b << "\n";
    }
  }
  {
    numerical::Hermite<>
    h(ext::compose1(std::ptr_fun<double, double>(std::exp),
                    std::negate<double>()),
      ext::compose1(std::negate<double>(),
                    ext::compose1(std::ptr_fun<double, double>(std::exp),
                                  std::negate<double>())),
      0.0, 32, 32 * 100);
    const double Arguments[] = {0, 0.01, 2, 4.1, 16, 31.99};
    const int Size = sizeof(Arguments) / sizeof(double);
    std::cout << "exp(-x) on [0..32)\n"
              << "function, argument, approximation, difference\n";
    double x, a, b;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      a = std::exp(-x);
      b = h(x);
      std::cout << x << ", " << a << ", " << b << ", " << a - b << "\n";
    }
    double deviation, maxDeviation = 0, maxX = -1;
    for (int i = 0; i != 32 * 100 * 16; ++i) {
      x = i * 32.0 / (32. * 100. * 16.);
      deviation = std::abs(std::exp(-x) - h(x)) / std::exp(-x);
      if (deviation > maxDeviation) {
        maxDeviation = deviation;
        maxX = x;
      }
    }
    std::cout << "Maximum relative deviation of " << maxDeviation << " at " << maxX
              << "\n";
  }

  std::cout << "Using HermiteFunctionDerivative:\n";
  {
    numerical::HermiteFunctionDerivative<>
    h(std::ptr_fun<double, double>(std::exp),
      std::ptr_fun<double, double>(std::exp),
      0.0, 2.0, 128);
    const double Arguments[] = {0, 0.3, 0.499, 0.5, 1.0, 1.99};
    const int Size = sizeof(Arguments) / sizeof(double);
    std::cout << "exp(x) on [0..2)\n"
              << "function, argument, approximation, difference\n";
    double x, a, b;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      a = std::exp(x);
      b = h(x);
      std::cout << x << ", " << a << ", " << b << ", " << a - b << "\n";
    }
  }
  {
    numerical::HermiteFunctionDerivative<>
    h(ext::compose1(std::ptr_fun<double, double>(std::exp),
                    std::negate<double>()),
      ext::compose1(std::negate<double>(),
                    ext::compose1(std::ptr_fun<double, double>(std::exp),
                                  std::negate<double>())),
      0.0, 16, 1024);
    const double Arguments[] = {0, 0.01, 2, 4.1, 15.99};
    const int Size = sizeof(Arguments) / sizeof(double);
    std::cout << "exp(-x) on [0..16)\n"
              << "function, argument, approximation, difference\n";
    double x, a, b;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      a = std::exp(-x);
      b = h(x);
      std::cout << x << ", " << a << ", " << b << ", " << a - b << "\n";
    }
  }

  return 0;
}
