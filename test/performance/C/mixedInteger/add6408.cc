// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "add 64 08";
typedef std::ptrdiff_t T1;
typedef char T2;
}

#define __performance_C_mixedInteger_add_ipp__
#include "add.ipp"
#undef __performance_C_mixedInteger_add_ipp__
