// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "multiply 64 08";
typedef std::ptrdiff_t T1;
typedef char T2;
}

#define __performance_C_mixedInteger_multiply_ipp__
#include "multiply.ipp"
#undef __performance_C_mixedInteger_multiply_ipp__
