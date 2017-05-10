// -*- C++ -*-

#define FUNCTION(x) std::abs(x)

namespace
{
const char* FunctionName = "std::abs()";
const double InitialValue = 0;
typedef double Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
