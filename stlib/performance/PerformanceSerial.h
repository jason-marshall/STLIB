// -*- C++ -*-

#if !defined(__performance_PerformanceSerial_h__)
#define __performance_PerformanceSerial_h__

/*!
  \file performance/PerformanceSerial.h
  \brief Measure the performance for serial applications
*/

#include "stlib/performance/Performance.h"
#include "stlib/performance/PerformanceDataSerial.h"

#include <iostream>

namespace stlib
{
namespace performance
{


/// Print performance data.
/** \ingroup PerformanceFunctions */
void
print(std::ostream& out = std::cout);


/// Print performance data in CSV tables.
/** \ingroup PerformanceFunctions */
void
printCsv(std::ostream& out = std::cout);


} // namespace performance
} // namespace stlib

#define __performance_PerformanceSerial_tcc__
#include "stlib/performance/PerformanceSerial.tcc"
#undef __performance_PerformanceSerial_tcc__

#endif
