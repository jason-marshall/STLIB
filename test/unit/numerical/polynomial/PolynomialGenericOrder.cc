// -*- C++ -*-

#include "stlib/numerical/polynomial/PolynomialGenericOrder.h"

#include <cassert>

using namespace stlib;

int
main()
{

  // Constant. f = 1. f' = 0.
  {
    const double c[] = {1};

    assert(numerical::evaluatePolynomial(0, c, 0.) == 1);
    assert(numerical::evaluatePolynomial(0, c, 1.) == 1);

    double d;
    assert(numerical::evaluatePolynomial(0, c, 0., &d) == 1);
    assert(d == 0);
    assert(numerical::evaluatePolynomial(0, c, 1., &d) == 1);
    assert(d == 0);

    {
      numerical::PolynomialGenericOrder<> p(0, c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
    {
      numerical::PolynomialGenericOrder<> p
        = numerical::constructPolynomialGenericOrder(0, c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
  }
  {
    const double data[] = {1};
    const std::vector<double> c(data, data + 1);

    assert(numerical::evaluatePolynomial(0, c, 0.) == 1);
    assert(numerical::evaluatePolynomial(0, c, 1.) == 1);

    double d;
    assert(numerical::evaluatePolynomial(0, c, 0., &d) == 1);
    assert(d == 0);
    assert(numerical::evaluatePolynomial(0, c, 1., &d) == 1);
    assert(d == 0);

    {
      numerical::PolynomialGenericOrder<> p(c);
      assert(p(0) == 1);
      assert(p(1) == 1);

      assert(p(0, &d) == 1);
      assert(d == 0);
      assert(p(1, &d) == 1);
      assert(d == 0);
    }
    {
      numerical::PolynomialGenericOrder<> p
        = numerical::constructPolynomialGenericOrder(c);
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
    numerical::PolynomialGenericOrder<> p(1, c);
    assert(p(0) == 1);
    assert(p(1) == 3);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 3);
    assert(d == 2);
  }
  {
    const double data[] = {1, 2};
    const std::vector<double> c(data, data + 2);
    numerical::PolynomialGenericOrder<> p(c);
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
    numerical::PolynomialGenericOrder<> p(2, c);
    assert(p(0) == 1);
    assert(p(1) == 6);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 6);
    assert(d == 8);
  }
  {
    const double data[] = {1, 2, 3};
    const std::vector<double> c(data, data + 3);
    numerical::PolynomialGenericOrder<> p(c);
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
    numerical::PolynomialGenericOrder<> p(3, c);
    assert(p(0) == 1);
    assert(p(1) == 10);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 10);
    assert(d == 20);
  }
  {
    const double data[] = {1, 2, 3, 4};
    const std::vector<double> c(data, data + 4);
    numerical::PolynomialGenericOrder<> p(c);
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
    numerical::PolynomialGenericOrder<> p(4, c);
    assert(p(0) == 1);
    assert(p(1) == 15);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 15);
    assert(d == 40);
  }
  {
    const double data[] = {1, 2, 3, 4, 5};
    const std::vector<double> c(data, data + 5);
    numerical::PolynomialGenericOrder<> p(c);
    assert(p(0) == 1);
    assert(p(1) == 15);
    double d;
    assert(p(0, &d) == 1);
    assert(d == 2);
    assert(p(1, &d) == 15);
    assert(d == 40);
  }

  return 0;
}
