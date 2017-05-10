// -*- C++ -*-

#if !defined(__performance_PerformanceDataMpi_h__)
#define __performance_PerformanceDataMpi_h__

/*!
  \file performance/PerformanceDataMpi.h
  \brief Performance data within a scope.
*/

#include "stlib/performance/PerformanceData.h"

#include "stlib/mpi/statistics.h"

namespace stlib
{
namespace performance
{


/// Print information about the performance.
/** \relates PerformanceData */
void
print(std::ostream& out, PerformanceData const& x, MPI_Comm comm);


/// Print CSV tables with the maximum values.
/** \relates PerformanceData */
void
printCsv(std::ostream& out, PerformanceData const& x, MPI_Comm comm);


} // namespace performance
} // namespace stlib

#define __performance_PerformanceDataMpi_tcc__
#include "stlib/performance/PerformanceDataMpi.tcc"
#undef __performance_PerformanceDataMpi_tcc__

#endif
