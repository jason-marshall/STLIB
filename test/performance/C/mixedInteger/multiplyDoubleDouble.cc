// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "multiply Double Double";
typedef double T1;
typedef double T2;
}

#define __performance_C_mixedInteger_multiply_ipp__
#include "multiply.ipp"
#undef __performance_C_mixedInteger_multiply_ipp__
