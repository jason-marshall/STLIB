// -*- C++ -*-

#include <cmath>

inline
double
function(const double x)
{
  return (x < 0.9 ? 1.0 - 0.5 * x* x : std::sin(x));
}

#define __performance_C_math_branch_f_ipp__
#include "f.ipp"
#undef __performance_C_math_branch_f_ipp__
