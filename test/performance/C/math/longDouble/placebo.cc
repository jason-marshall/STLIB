// -*- C++ -*-

#define FUNCTION(x) x

typedef long double Number;

namespace
{
const char* FunctionName = "placebo";
const Number InitialValue = 1;
typedef Number Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
