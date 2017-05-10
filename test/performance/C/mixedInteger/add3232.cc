// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "add 32 32";
typedef int T1;
typedef int T2;
}

#define __performance_C_mixedInteger_add_ipp__
#include "add.ipp"
#undef __performance_C_mixedInteger_add_ipp__
