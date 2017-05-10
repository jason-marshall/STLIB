// -*- C++ -*-

#if !defined(__sfc_UniformCellsMpi_h__)
#define __sfc_UniformCellsMpi_h__

/*!
  \file
  \brief Distributed algorithms for UniformCells.
*/

#include "stlib/sfc/NonOverlappingCellsMpi.h"
#include "stlib/sfc/UniformCells.h"

#include "stlib/performance/PerformanceMpi.h"

namespace stlib
{
namespace sfc
{


//! Merge the distributed groups of cells. Coarsen as needed.
/*! On the process with rank 0, the output is the merged, and possibly
  coarsened, cells. The number of cells in the output will not exceed
  the specified maximum. On all other processes, output is not generated.
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
void
mergeCoarsen
(UniformCells<_Traits, _Cell, _StoreDel> const& input,
 std::size_t maxCells,
 UniformCells<_Traits, _Cell, _StoreDel>* output,
 MPI_Comm comm, const double targetSizeFactor = 1);


//! Merge the distributed groups of cells. Coarsen as needed.
/*! On the process with rank 0, the output is the merged, and possibly
  coarsened, cells. The number of cells in the output will not exceed
  the specified maximum. On all other processes, output is not generated.
  This function is designed for distributions that are mostly disjoint.
  This assumption allows us to apply coarsening earlier in the procedure, which 
  results in a more efficient algorithm.
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
void
mergeCoarsenDisjoint
(const UniformCells<_Traits, _Cell, _StoreDel>& input,
 std::size_t maxCells,
 UniformCells<_Traits, _Cell, _StoreDel>* output,
 MPI_Comm comm);


//! Merge the distributed groups of cells.
/*! On the process with rank 0, the output is the merged cells. On all other
  processes, output is not generated. The number of levels of refinement in
  the output is the minimum of the inputs.
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
merge(const UniformCells<_Traits, _Cell, _StoreDel>& input,
      UniformCells<_Traits, _Cell, _StoreDel>* output,
      MPI_Comm comm)
{
  // Merge and coarsen with no limit on the number of cells.
  mergeCoarsen(input, std::size_t(-1), output, comm);
}


//! Distribute the cells and objects according to the partition.
template<typename _Traits, typename _Cell, typename _Object>
void
distribute(UniformCells<_Traits, _Cell, true>* cells,
           std::vector<_Object>* objects,
           Partition<_Traits> const& codePartition, MPI_Comm comm);


//! Partition and distribute the cells and objects. Coarsen if necessary.
/*! We use mergeCoarsen() to obtain coarsened cells at the root. The maximum
  allowed number of cells in this data structure is the number of processes
  divided by the imbalance goal. 

  \note If the distribution of objects is inhomogeneous, decrease the
  imbalance goal to account for the inhomogenous distribution in the
  number of objects per cell.
*/
template<typename _Traits, typename _Cell, typename _Object>
void
partitionCoarsen
(UniformCells<_Traits, _Cell, true>* cells,
 std::vector<_Object>* objects,
 Partition<_Traits>* codePartition, MPI_Comm comm,
 double imbalanceGoal = 0.01);


//! Partition and distribute the objects.
/*! The partitioning respects cell boundaries. A cell data structure that
  records element counts is built and merged. (The discrete coordinates should
  be such that this merged data structure has reasonable storage requirements.)
*/
template<typename _Traits, typename _Object>
void
partition(DiscreteCoordinates<_Traits> const& discreteCoordinates,
          std::vector<_Object>* objects,
          Partition<_Traits>* codePartition, MPI_Comm comm);


} // namespace sfc
} // namespace stlib

#define __sfc_UniformCellsMpi_tcc__
#include "stlib/sfc/UniformCellsMpi.tcc"
#undef __sfc_UniformCellsMpi_tcc__

#endif
