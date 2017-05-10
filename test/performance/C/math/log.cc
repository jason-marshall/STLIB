// -*- C++ -*-

#define FUNCTION(x) std::log(x)

namespace
{
const char* FunctionName = "std::log()";
const double InitialValue = 1;
typedef double Result;
}

#define __performance_C_math_f_ipp__
#include "f.ipp"
#undef __performance_C_math_f_ipp__
