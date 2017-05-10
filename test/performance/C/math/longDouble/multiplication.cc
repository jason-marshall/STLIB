// -*- C++ -*-

#define FUNCTION(x) x * 3.14159

typedef long double Number;

namespace
{
const char* FunctionName = "multiplication";
const Number InitialValue = 1;
typedef Number Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
