// -*- C++ -*-

#if !defined(__sfc_NonOverlappingCellsMpi_h__)
#define __sfc_NonOverlappingCellsMpi_h__

/*!
  \file
  \brief Distributed algorithms for NonOverlappingCells.
*/

#include "stlib/sfc/NonOverlappingCells.h"
#include "stlib/sfc/Partition.h"

#include "stlib/mpi/allToAll.h"
#include "stlib/mpi/wrapper.h"

namespace stlib
{
namespace sfc
{


//! Send the cells.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
void
send(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order> const& input,
     int dest, int tag, MPI_Comm comm);


//! Receive the cells.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
void
recv(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* output,
     int source, int tag, MPI_Comm comm);


//! Broadcast the cells data structure from the root.
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
void
bcast(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* cells,
      MPI_Comm comm, int root = 0);


//! Reduce the compatible cell data structures.
/*!
  On process 0, the output is the merged cells.
*/
template<typename _Traits, typename _Cell, bool _StoreDel,
         template<typename> class _Order>
void
reduceCompatible
(NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order> const& input,
 NonOverlappingCells<_Traits, _Cell, _StoreDel, _Order>* output,
 MPI_Comm comm);


//! Redistribute the objects, but don't sort the result on each local process.
/*!
  \param localCells The cell data structure defines codes and object delimiters.
  \param objects The vector of objects. Both input and output.
  \param codePartition Defines the partitioning of the codes for the MPI
  processes.
  \param comm The MPI communicator.

  On input, the objects are sorted according the the cells in localCells. 
  On output, each object is stored on the correct MPI process, but is not 
  in any particular order.
*/
template<typename _Traits, typename _Cell, template<typename> class _Order,
         typename _Object>
inline
void
distributeNoSort
(NonOverlappingCells<_Traits, _Cell, true, _Order> const& localCells,
 std::vector<_Object>* objects,
 Partition<_Traits> const& codePartition, MPI_Comm comm);


} // namespace sfc
} // namespace stlib

#define __sfc_NonOverlappingCellsMpi_tcc__
#include "stlib/sfc/NonOverlappingCellsMpi.tcc"
#undef __sfc_NonOverlappingCellsMpi_tcc__

#endif
