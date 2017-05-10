// -*- C++ -*-

/**
  \file mpi/sort.h
  \brief Sort distributed data.
*/

#if !defined(__mpi_sort_h__)
#define __mpi_sort_h__

#include "stlib/mpi/wrapper.h"

#include <string>
#include <type_traits>

namespace stlib
{
namespace mpi
{


/** \defgroup mpiSort Sort distributed data.
    @{
*/


/// Sort the distributed values.
/**
  \param input The input vector of sorted elements.
  \param output The merged, sorted elements. Only valid at the root process.
  \param comm The MPI intra-communicator.

  When merging two vectors, use the sequential scan algorithm.
*/
template<typename _T>
void
mergeSortedSequentialScan(std::vector<_T> const& input, std::vector<_T>* output,
                          MPI_Comm comm = MPI_COMM_WORLD);


/// Sort the distributed values.
/**
  \param input The input vector of sorted elements.
  \param output The merged, sorted elements. Only valid at the root process.
  \param comm The MPI intra-communicator.

  When merging two vectors, use binary searches to find the crossover points.
*/
template<typename _T>
void
mergeSortedBinarySearch(std::vector<_T> const& input, std::vector<_T>* output,
                        MPI_Comm comm = MPI_COMM_WORLD);


/// Sort the distributed values.
/**
  \param input The input vector of sorted elements.
  \param output The merged, sorted elements. Only valid at the root process.
  \param comm The MPI communicator.

  When merging two vectors, use the sequential scan algorithm.
*/
template<typename _T>
inline
void
mergeSorted(std::vector<_T> const& input, std::vector<_T>* output,
            MPI_Comm const comm = MPI_COMM_WORLD)
{
  mergeSortedSequentialScan(input, output, comm);
}


/// Sort the distributed values.
/**
  \param input The input vector of sorted value/count pairs.
  \param output The merged, sorted pairs. Only valid at the root process.
  \param comm The MPI communicator.
*/
template<typename _T>
void
mergeSorted(std::vector<std::pair<_T, std::size_t> > const& input,
            std::vector<std::pair<_T, std::size_t> >* output,
            MPI_Comm const comm = MPI_COMM_WORLD);


/** @} */ // End of mpiSort group.


} // namespace mpi
}

#define __mpi_sort_tcc__
#include "stlib/mpi/sort.tcc"
#undef __mpi_sort_tcc__

#endif
