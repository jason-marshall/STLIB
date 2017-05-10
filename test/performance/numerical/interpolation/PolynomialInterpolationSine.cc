// -*- C++ -*-

#include "stlib/numerical/constants.h"

#include<functional>

#include <cmath>

using namespace stlib;

const double Lower = 0;
const double Upper = 0.5 * numerical::Constants<double>::Pi();

struct F :
    public std::unary_function<double, double> {
  result_type
  operator()(const argument_type x) const
  {
    return std::sin(x);
  }
};

struct DF :
    public std::unary_function<double, double> {
  result_type
  operator()(const argument_type x) const
  {
    return std::cos(x);
  }
};

struct DDF :
    public std::unary_function<double, double> {
  result_type
  operator()(const argument_type x) const
  {
    return - std::sin(x);
  }
};

#define __performance_numerical_interpolation_PolynomialInterpolationSampler_ipp__
#include "PolynomialInterpolationSampler.ipp"
#undef __performance_numerical_interpolation_PolynomialInterpolationSampler_ipp__
