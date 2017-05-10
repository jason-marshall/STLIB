// -*- C++ -*-

#if !defined(__stlib_sfc_AdaptiveCellsMpi_h__)
#define __stlib_sfc_AdaptiveCellsMpi_h__

/**
  \file
  \brief Distributed algorithms for AdaptiveCells.
*/

#include "stlib/sfc/AdaptiveCells.h"
#include "stlib/sfc/NonOverlappingCellsMpi.h"
#include "stlib/sfc/RefinementSortCodes.h"
#include "stlib/sfc/BuildFromBlockCodeSizePairs.h"
#include "stlib/sfc/Partition.h"

#include "stlib/mpi/BBox.h"
#include "stlib/mpi/sort.h"
#include "stlib/mpi/statistics.h"
#include "stlib/performance/PerformanceMpi.h"

namespace stlib
{
namespace sfc
{


/// Build local cells. Reduce to build global cells, then broadcast them.
/**
   \param grid The virtual grid geometry.
   \param objects The sequence of objects will be reordered in building the
   local cells.
   \param maxObjectsPerCell The maximum number of objects per cell.
   \param localCells The local cells are built using the local objects.
   \param globalCells The local cells are reduced to form the global cells.
   \param codePartition A fair partitioning of the global cells. Note that
   if you pass a null pointer for the partition, it will not be computed.
   \param comm The MPI communicator.
*/
template<typename _AdaptiveCells, typename _Object>
_AdaptiveCells
adaptiveCells(typename _AdaptiveCells::Grid const& grid,
                std::vector<_Object>* objects,
                std::size_t maxObjectsPerCell,
                _AdaptiveCells* localCells,
                Partition<typename _AdaptiveCells::Traits>* codePartition,
                MPI_Comm comm);


/// Build local cells. Reduce to build global cells, then broadcast them.
/**
   \param grid The virtual grid geometry.
   \param objects The sequence of objects will be reordered in building the
   local cells.
   \param maxObjectsPerCell The maximum number of objects per cell.
   \param codePartition A fair partitioning of the global cells. Note that
   if you pass a null pointer for the partition, it will not be computed.
   \param comm The MPI communicator.

   Use this interface when you don't need to use the local cells that are
   built in the process of assembling the global cells.
*/
template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
adaptiveCells(typename _AdaptiveCells::Grid const& grid,
                std::vector<_Object>* objects,
                std::size_t const maxObjectsPerCell,
                Partition<typename _AdaptiveCells::Traits>* codePartition,
                MPI_Comm const comm)
{
  _AdaptiveCells localCells;
  return adaptiveCells(grid, objects, maxObjectsPerCell, &localCells,
                         codePartition, comm);
}


/// Build local cells. Reduce to build global cells, then broadcast them.
/**
   \param grid The virtual grid geometry.
   \param objects The sequence of objects will be reordered in building the
   local cells.
   \param maxObjectsPerCell The maximum number of objects per cell.
   \param comm The MPI communicator.
*/
template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
adaptiveCells(typename _AdaptiveCells::Grid const& grid,
                std::vector<_Object>* objects,
                std::size_t const maxObjectsPerCell,
                MPI_Comm const comm)
{
  Partition<typename _AdaptiveCells::Traits>* codePartition = nullptr;
  return adaptiveCells<_AdaptiveCells>(grid, objects, maxObjectsPerCell,
                                           codePartition, comm);
}


/// Determine an appropriate maximum number of objects per cell for MPI applications.
/**
\relates AdaptiveCells
\param numLocal The number of local objects.
\param comm The MPI communicator.
\param minimum The minimum permissible number of objects per cell. This value
is typically dictated by maximum objects per cell in data structures that are
used in serial algorithms.
\param targetCellsPerProcess The target number of cells per process. This 
defines the target accuracy for algorithms that use the SFC data structure.
*/
template<std::size_t _Dimension>
inline
std::size_t
maxObjectsPerCell
(std::size_t const numLocal,
 MPI_Comm const comm,
 std::size_t const minimum =
 AdaptiveCells<Traits<_Dimension>, void, false>::DefaultMaxObjectsPerCell,
 std::size_t const targetCellsPerProcess =
 AdaptiveCells<Traits<_Dimension>, void, false>::DefaultTargetCellsPerProcess)
{
  return maxObjectsPerCellDistributed<_Dimension>
    (mpi::allReduce(numLocal, MPI_SUM, comm), stlib::mpi::commSize(comm),
     minimum, targetCellsPerProcess);
}


/// Distribute the objects according to adaptive Morton blocks.
/**
  \param objects The distributed objects.
  \param maxObjectsPerCell The maximum allowed number of objects per cell.
  Only cells at the highest level of refinement can have more that this
  number of objects.
   \param codePartition A fair partitioning of the global cells.
  \param comm The MPI communicator.

  First local cells are built from the initial objects.
  Then the local cells are reduced to form the global cells.
  The objects will be redistributed according to the partitioning. On a given
  process, the objects are not ordered. The global cells data structure 
  and the partitioning are output on all processes.
*/
template<typename _AdaptiveCells, typename _Object>
_AdaptiveCells
distribute(typename _AdaptiveCells::Grid const& grid,
           std::vector<_Object>* objects,
           std::size_t maxObjectsPerCell,
           Partition<typename _AdaptiveCells::Traits>* codePartition,
           MPI_Comm comm);


/// Determine an appropriate maximum number of objects per cell given the accuracy goal.
/**
   \param numLocal The number of local objects.
   \param comm The MPI communicator.
   \param accuracyGoal The accuracy goal for partitioning the objects.
*/
template<typename _Traits>
std::size_t
maxObjectsPerCellForPartitioning(std::size_t numLocal,
                                 MPI_Comm comm = MPI_COMM_WORLD,
                                 double accuracyGoal = 0.01);


/// Distribute the objects according to adaptive Morton blocks.
/**
  \param objects The distributed objects.
  \param codePartition A fair partitioning of the global cells.
  \param accuracyGoal The accuracy goal for partitioning the objects.
  \param comm The MPI communicator.

  First local cells are built from the initial objects.
  Then the local cells are reduced to form the global cells.
  This function uses the accuracy goal to determine an appropriate maximum
  number of objects per cell. Then it calls the above distribute function.
*/
template<typename _AdaptiveCells, typename _Object>
_AdaptiveCells
distribute(typename _AdaptiveCells::Grid const& grid,
           std::vector<_Object>* objects,
           Partition<typename _AdaptiveCells::Traits>* codePartition,
           MPI_Comm comm,
           double accuracyGoal = 0.01);


/// Distribute the objects according to adaptive Morton blocks.
/**
   Use this interface when you have a specific domain that you want to use 
   for the SFC cell data structures, but you don't need the global cell data
   structure or the partitioning of its codes.
*/
template<typename _Object, typename _Float, std::size_t _Dimension>
void
distribute(std::vector<_Object>* objects,
           geom::BBox<_Float, _Dimension> const& domain, MPI_Comm comm,
           double accuracyGoal = 0.01);


/// Distribute the objects according to adaptive Morton blocks.
/**
   Use this interface when you don't need the global cell data structure or
   the partitioning of its codes.
*/
template<typename _Float, std::size_t _Dimension, typename _Object>
void
distribute(std::vector<_Object>* objects, MPI_Comm comm,
           double accuracyGoal = 0.01);


} // namespace sfc
} // namespace stlib

#define __stlib_sfc_AdaptiveCellsMpi_tcc__
#include "stlib/sfc/AdaptiveCellsMpi.tcc"
#undef __stlib_sfc_AdaptiveCellsMpi_tcc__

#endif
