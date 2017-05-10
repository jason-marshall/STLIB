// -*- C++ -*-

#if !defined(__performance_PerformanceMpi_h__)
#define __performance_PerformanceMpi_h__

/*!
  \file performance/PerformanceMpi.h
  \brief Measure the performance by recording quantities and timing events.
*/

#include "stlib/performance/Performance.h"
#include "stlib/performance/PerformanceDataMpi.h"

namespace stlib
{
namespace performance
{


/// Print information about the %performance.
/** \ingroup PerformanceFunctions */
void
print(std::ostream& out = std::cout, MPI_Comm comm = MPI_COMM_WORLD);


/// Print CSV tables with the maximum values.
/** \ingroup PerformanceFunctions */
void
printCsv(std::ostream& out = std::cout, MPI_Comm comm = MPI_COMM_WORLD);


} // namespace performance
} // namespace stlib

#define __performance_PerformanceMpi_tcc__
#include "stlib/performance/PerformanceMpi.tcc"
#undef __performance_PerformanceMpi_tcc__

#endif
