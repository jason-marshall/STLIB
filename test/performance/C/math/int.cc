// -*- C++ -*-

#define FUNCTION(x) int(x)

namespace
{
const char* FunctionName = "int()";
const double InitialValue = 1;
typedef int Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
