// -*- C++ -*-

/**
  \file mpi/partition.h
  \brief Partition vectors of objects.
*/

#if !defined(__mpi_partition_h__)
#define __mpi_partition_h__

#include "stlib/mpi/wrapper.h"
#include "stlib/mpi/allToAll.h"

#include "stlib/numerical/partition.h"

namespace stlib
{
namespace mpi
{

/** Partition a vector of objects.
Retain the global order of the distributed objects.

\param objects The vector of local objects will be replaced with 
redistributed objects.
\param comm The MPI communicator.
\param maxImbalance The objects will be partitioned if the maximum allowed 
load imbalance is exceeded.
\return \c true if the objects were partitioned.
*/
template<typename _T>
bool
partitionOrdered(std::vector<_T>* objects, MPI_Comm comm = MPI_COMM_WORLD,
                 double maxImbalance = 0.1);

/** Partition a vector of objects.
Send objects from processes with an excess to those with deficits.

\param objects The vector of local objects will be replaced with 
redistributed objects.
\param comm The MPI communicator.
\param maxImbalance The objects will be partitioned if the maximum allowed 
load imbalance is exceeded.
\return \c true if the objects were partitioned.
*/
template<typename _T>
bool
partitionExcess(std::vector<_T>* objects, MPI_Comm comm = MPI_COMM_WORLD,
                double maxImbalance = 0.1);

} // namespace mpi
}

#define __mpi_partition_tcc__
#include "stlib/mpi/partition.tcc"
#undef __mpi_partition_tcc__

#endif
