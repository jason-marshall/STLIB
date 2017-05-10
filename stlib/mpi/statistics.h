// -*- C++ -*-

/**
  \file mpi/statistics.h
  \brief Compute statistics for distributed values.
*/

#if !defined(__mpi_statistics_h__)
#define __mpi_statistics_h__

#include "stlib/mpi/wrapper.h"

#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>
#include <string>

namespace stlib
{
namespace mpi
{


/** \defgroup mpiStatistics Compute statistics for distributed values
    The functions have one peculiarity: they were designed for distributed 
    values, but also work for serial applications. For a serial application,
    just pass \c MPI_COMM_NULL as the argument for the MPI communicator.
    This feature can simplify the programming logic.

    @{
*/


/// Calculate statistics for the distributed values.
/**
  \param value The local value.
  \param sum Set to the sum of the values.
  \param mean Set to the mean of the values.
  \param min Set to the minima of the values.
  \param max Set to the maxima of the values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.
*/
template<typename _T, typename _Float>
void
gatherStatistics(_T value, _T* sum, _Float* mean, _T* min, _T* max,
                 MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/// Print statistics for the distributed values at the root.
/**
  \param out The output stream.
  \param name The name for the value will be printed first.
  \param value The local value.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.
*/
template<typename _T>
void
printStatistics(std::ostream& out, std::string const& name, _T value,
                MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/// Calculate statistics for the distributed values.
/**
  \param values The local values.
  \param sums Set to the sums of the values.
  \param means Set to the means of the values.
  \param minima Set to the minima of the values.
  \param maxima Set to the maxima of the values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.

  When you have multiple values, using this function is more efficient than
  calling gatherStatistics() for each value because it uses a single MPI
  gather operation.
*/
template<typename _T, std::size_t N, typename _Float>
void
gatherStatistics(std::array<_T, N> const& values, std::array<_T, N>* sums,
                 std::array<_Float, N>* means, std::array<_T, N>* minima,
                 std::array<_T, N>* maxima, MPI_Comm comm = MPI_COMM_WORLD,
                 int root = 0);


/// Print statistics for the distributed values at the root.
/**
  \param out The output stream.
  \param names The name for the values will be printed first.
  \param values The local values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.

  When you have multiple values, using this function is more efficient than
  calling printStatistics() for each value because it uses a single MPI
  gather operation.
*/
template<typename _T, std::size_t N>
void
printStatistics(std::ostream& out, std::array<std::string, N> const& names,
                std::array<_T, N> const& values,
                MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/// Calculate statistics for the distributed values.
/**
  \param values The local values.
  \param sum Set to the sum of the values.
  \param mean Set to the mean of the values.
  \param min Set to the minima of the values.
  \param max Set to the maxima of the values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.
*/
template<typename _T, typename _Float>
void
gatherStatistics(std::vector<_T> const& values, _T* sum, _Float* mean, _T* min,
                 _T* max, MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/// Print statistics for the distributed values at the root.
/**
  \param out The output stream.
  \param name The name for the value will be printed first.
  \param values The local values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.
*/
template<typename _T>
void
printStatistics(std::ostream& out, std::string const& name,
                std::vector<_T> const& values, MPI_Comm comm = MPI_COMM_WORLD,
                int root = 0);


/// Calculate statistics for the distributed values.
/**
  \param first The beginning of the local values.
  \param last One past the end of the local values.
  \param sum Set to the sum of the values.
  \param mean Set to the mean of the values.
  \param min Set to the minima of the values.
  \param max Set to the maxima of the values.
  \param comm The MPI intra-communicator.
  \param root The process at which the statistics will be set. The default is 0.
*/
template<typename _ForwardIterator, typename _T, typename _Float>
void
gatherStatistics(_ForwardIterator first, _ForwardIterator last, _T* sum,
                 _Float* mean, _T* min, _T* max,
                 MPI_Comm comm = MPI_COMM_WORLD, int root = 0);


/** @} */ // End of mpiStatistics group.


} // namespace mpi
} // namespace stlib

#define __mpi_statistics_tcc__
#include "stlib/mpi/statistics.tcc"
#undef __mpi_statistics_tcc__

#endif
