// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "multiply 32 32";
typedef int T1;
typedef int T2;
}

#define __performance_C_mixedInteger_multiply_ipp__
#include "multiply.ipp"
#undef __performance_C_mixedInteger_multiply_ipp__
