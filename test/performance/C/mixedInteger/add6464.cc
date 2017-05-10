// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "add 64 64";
typedef std::ptrdiff_t T1;
typedef std::ptrdiff_t T2;
}

#define __performance_C_mixedInteger_add_ipp__
#include "add.ipp"
#undef __performance_C_mixedInteger_add_ipp__
