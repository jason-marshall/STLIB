// -*- C++ -*-

#include <cstddef>

namespace
{
const char* Name = "shift right U64 U64";
typedef std::size_t T1;
typedef std::size_t T2;
}

#define __performance_C_mixedInteger_shiftRight_ipp__
#include "shiftRight.ipp"
#undef __performance_C_mixedInteger_shiftRight_ipp__
