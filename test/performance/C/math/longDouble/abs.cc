// -*- C++ -*-

#define FUNCTION(x) std::abs(x)

typedef long double Number;

namespace
{
const char* FunctionName = "std::abs()";
const Number InitialValue = 0;
typedef Number Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
