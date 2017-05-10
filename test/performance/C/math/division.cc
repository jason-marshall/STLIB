// -*- C++ -*-

#define FUNCTION(x) x / 3.14159

namespace
{
const char* FunctionName = "division";
const double InitialValue = 1;
typedef double Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
