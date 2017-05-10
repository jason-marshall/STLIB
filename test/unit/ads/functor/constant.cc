// -*- C++ -*-

#include "stlib/ads/functor/constant.h"

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // Generator.
  //
  {
    // Value constructor.
    ads::GeneratorConstant<double> x(0.5);
    assert(x() == 0.5);
  }
  {
    // Default constructor.
    ads::GeneratorConstant<double> x;
    x.set(0.5);
    assert(x() == 0.5);
  }
  {
    // Function with value.
    ads::GeneratorConstant<double> x;
    x = ads::constructGeneratorConstant<double>(0.5);
    assert(x() == 0.5);
  }
  {
    // Function without value.
    ads::GeneratorConstant<double> x;
    x = ads::constructGeneratorConstant<double>();
    x.set(0.5);
    assert(x() == 0.5);
  }

  //
  // Generator void.
  //
  {
    // Default constructor.
    ads::GeneratorConstant<void> x;
    x.set();
    x();
  }
  {
    // Function without value.
    ads::GeneratorConstant<void> x;
    x = ads::constructGeneratorConstant<void>();
    x.set();
    x();
  }

  //
  // Unary.
  //
  {
    // Value constructor.
    ads::UnaryConstant<int, double> x(0.5);
    assert(x(2357) == 0.5);
  }
  {
    // Default constructor.
    ads::UnaryConstant<int, double> x;
    x.set(0.5);
    assert(x(2357) == 0.5);
  }
  {
    // Function with value.
    ads::UnaryConstant<int, double> x;
    x = ads::constructUnaryConstant<int, double>(0.5);
    assert(x(2357) == 0.5);
  }
  {
    // Function without value.
    ads::UnaryConstant<int, double> x;
    x = ads::constructUnaryConstant<int, double>();
    x.set(0.5);
    assert(x(2357) == 0.5);
  }

  //
  // Binary.
  //
  {
    // Value constructor.
    ads::BinaryConstant<char, int, double> x(0.5);
    assert(x('a', 2357) == 0.5);
  }
  {
    // Default constructor.
    ads::BinaryConstant<char, int, double> x;
    x.set(0.5);
    assert(x('a', 2357) == 0.5);
  }
  {
    // Function with value.
    ads::BinaryConstant<char, int, double> x;
    x = ads::constructBinaryConstant<char, int, double>(0.5);
    assert(x('a', 2357) == 0.5);
  }
  {
    // Function without value.
    ads::BinaryConstant<char, int, double> x;
    x = ads::constructBinaryConstant<char, int, double>();
    x.set(0.5);
    assert(x('a', 2357) == 0.5);
  }

  return 0;
}
