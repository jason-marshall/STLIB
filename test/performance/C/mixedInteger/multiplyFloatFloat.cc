// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "multiply Float Float";
typedef float T1;
typedef float T2;
}

#define __performance_C_mixedInteger_multiply_ipp__
#include "multiply.ipp"
#undef __performance_C_mixedInteger_multiply_ipp__
