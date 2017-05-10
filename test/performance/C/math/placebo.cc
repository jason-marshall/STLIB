// -*- C++ -*-

#define FUNCTION(x) x

namespace
{
const char* FunctionName = "placebo";
const double InitialValue = 1;
typedef double Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
