// -*- C++ -*-

inline
double
function(const double x)
{
  return 1.0 - 0.5 * x * x;
}

#define __performance_C_math_branch_f_ipp__
#include "f.ipp"
#undef __performance_C_math_branch_f_ipp__
