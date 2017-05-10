// -*- C++ -*-

#include "stlib/numerical/polynomial/Polynomial.h"

#include <cassert>

using namespace stlib;

int
main()
{

  // Constant. f = 1. f' = 0.
  {
    const double c[] = {1};

    assert(numerical::evaluatePolynomial<0>(c, 0.) == 1);
    assert(numerical::evaluatePolynomial<0>(c, 1.) == 1);

    double d;
    assert(numerical::evaluatePolynomial<0>(c, 0., &d) == 1);
    assert(d == 0);
    assert(numerical::evaluatePolynomial<0>(c, 1., &d) == 1);
    assert(d == 0);

    {
      numerical::Polynomial<0> p(c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
    {
      numerical::Polynomial<0> p = numerical::constructPolynomial<0>(c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
  }
  {
    const std::array<double, 1> c = {{1}};

    assert(numerical::evaluatePolynomial(c, 0.) == 1);
    assert(numerical::evaluatePolynomial(c, 1.) == 1);

    double d;
    assert(numerical::evaluatePolynomial(c, 0., &d) == 1);
    assert(d == 0);
    assert(numerical::evaluatePolynomial(c, 1., &d) == 1);
    assert(d == 0);

    {
      numerical::Polynomial<0> p(c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
    {
      numerical::Polynomial<0> p = numerical::constructPolynomial(c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
  }
  // Linear. f = 1 + 2 x. f' = 2.
  {
    const double c[] = {1, 2};
    numerical::Polynomial<1> p(c);
    assert(p(0) == 1);
    assert(p(1) == 3);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 3);
    assert(d == 2);
  }
  {
    const std::array<double, 2> c = {{1, 2}};
    numerical::Polynomial<1> p(c);
    assert(p(0) == 1);
    assert(p(1) == 3);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 3);
    assert(d == 2);
  }
  // Quadratic. f = 1 + 2 x + 3 x^2. f' = 2 + 6 x.
  {
    const double c[] = {1, 2, 3};
    numerical::Polynomial<2> p(c);
    assert(p(0) == 1);
    assert(p(1) == 6);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 6);
    assert(d == 8);
  }
  {
    const std::array<double, 3> c = {{1, 2, 3}};
    numerical::Polynomial<2> p(c);
    assert(p(0) == 1);
    assert(p(1) == 6);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 6);
    assert(d == 8);
  }
  // Cubic. f = 1 + 2 x + 3 x^2 + 4 x^3. f' = 2 + 6 x + 12 x^2.
  {
    const double c[] = {1, 2, 3, 4};
    numerical::Polynomial<3> p(c);
    assert(p(0) == 1);
    assert(p(1) == 10);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 10);
    assert(d == 20);
  }
  {
    const std::array<double, 4> c = {{1, 2, 3, 4}};
    numerical::Polynomial<3> p(c);
    assert(p(0) == 1);
    assert(p(1) == 10);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 10);
    assert(d == 20);
  }
  // Quartic. f = 1 + 2 x + 3 x^2 + 4 x^3 + 5 x^4.
  // f' = 2 + 6 x + 12 x^2 + 20 x^3.
  {
    const double c[] = {1, 2, 3, 4, 5};
    numerical::Polynomial<4> p(c);
    assert(p(0) == 1);
    assert(p(1) == 15);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 15);
    assert(d == 40);
  }
  {
    const std::array<double, 5> c = {{1, 2, 3, 4, 5}};
    numerical::Polynomial<4> p(c);
    assert(p(0) == 1);
    assert(p(1) == 15);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 15);
    assert(d == 40);
  }
  //
  // Differentiate.
  //
  {
    // Quadratic. f = 1 + 2 x + 3 x^2. f' = 2 + 6 x.
    double c[] = {1, 2, 3};
    numerical::differentiatePolynomialCoefficients<2>(c);
    assert(c[0] == 2 && c[1] == 6 && c[2] == 0);
  }

  return 0;
}
