// -*- C++ -*-

#if !defined(__performance_PerformanceDataSerial_h__)
#define __performance_PerformanceDataSerial_h__

/**
  \file performance/PerformanceDataSerial.h
  \brief Printing functions for performance data in serial applications.
*/

#include "stlib/performance/PerformanceData.h"

namespace stlib
{
namespace performance
{


/// Print information about the performance.
/** \relates PerformanceData */
void
print(std::ostream& out, PerformanceData const& x);

/// Print CSV tables with the values.
/** \relates PerformanceData */
void
printCsv(std::ostream& out, PerformanceData const& x);


} // namespace performance
} // namespace stlib

#define __performance_PerformanceDataSerial_tcc__
#include "stlib/performance/PerformanceDataSerial.tcc"
#undef __performance_PerformanceDataSerial_tcc__

#endif
