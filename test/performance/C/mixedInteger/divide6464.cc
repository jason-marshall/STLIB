// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "divide 64 64";
typedef std::ptrdiff_t T1;
typedef std::ptrdiff_t T2;
}

#define __performance_C_mixedInteger_divide_ipp__
#include "divide.ipp"
#undef __performance_C_mixedInteger_divide_ipp__
