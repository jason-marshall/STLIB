// -*- C++ -*-

#define FUNCTION(x) int(x)

typedef long double Number;

namespace
{
const char* FunctionName = "int()";
const Number InitialValue = 1;
typedef int Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
