// -*- C++ -*-

#include <cmath>

inline
int
frexpInt(double x)
{
  int i;
  std::frexp(x, &i);
  return i;
}

#define FUNCTION(x) frexpInt(x)

namespace
{
const char* FunctionName = "std::frexp()";
const double InitialValue = 1;
typedef int Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
