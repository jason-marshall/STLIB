// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "add Double Double";
typedef double T1;
typedef double T2;
}

#define __performance_C_mixedInteger_add_ipp__
#include "add.ipp"
#undef __performance_C_mixedInteger_add_ipp__
