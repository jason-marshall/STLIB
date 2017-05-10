// -*- C++ -*-

#if !defined(__sfc_UniformCellsMpi_tcc__)
#error This file is an implementation detail of UniformCellsMpi.
#endif

namespace stlib
{
namespace sfc
{


inline
std::size_t
_targetSize(std::size_t const maxCells, double const targetSizeFactor,
            int numProcesses)
{
  // Note the following unexpected casting behavior.
  // std::size_t(double(std::numeric_limits<std::size_t>::max())) == 0.
  // If maxCells == std::numeric_limits<std::size_t>::max() then there 
  // is no limit on the number of cells.
  if (maxCells == std::numeric_limits<std::size_t>::max() ||
      targetSizeFactor == 1) {
    return maxCells;
  }
  double sizeFraction = 1;
  while (numProcesses > 1) {
    numProcesses /= 2;
    sizeFraction *= targetSizeFactor;
  }
  return std::max(std::size_t(maxCells * sizeFraction), std::size_t(1));
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
mergeCoarsen
(const UniformCells<_Traits, _Cell, _StoreDel>& input,
 const std::size_t maxCells,
 UniformCells<_Traits, _Cell, _StoreDel>* output,
 const MPI_Comm comm,
 const double targetSizeFactor)
{
  assert(maxCells != 0);

  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);

  // Copy the input.
  *output = input;

  // A power of two that is at least the number of processes.
  int n = 1;
  int cellsTag = 0;
  while (n < commSize) {
    n *= 2;
    ++cellsTag;
  }

  // Perform initial coarsening.
  {
    // Coarsen if necessary to not exceed the target number of cells.
    std::size_t const targetSize = _targetSize(maxCells, targetSizeFactor, n);
    while (output->size() > targetSize) {
      output->coarsen();
    }
    // Determine the minimum number of levels across processes.
    std::size_t const numLevels = mpi::allReduce(output->numLevels(), MPI_MIN,
                                                 comm);
    // The result will be at least that coarse, so perform the coarsening
    // now for best efficiency.
    while (output->numLevels() > numLevels) {
      output->coarsen();
    }
  }

  UniformCells<_Traits, _Cell, _StoreDel>
    received(input.lowerCorner(), input.lengths(), input.numLevels());
  int numLevelsTag = 0;
  for (; n > 1; n /= 2) {
    // If this process is still active (sender or receiver).
    if (commRank < n) {
      // Coarsen if necessary to not exceed the target number of cells.
      std::size_t const targetSize = _targetSize(maxCells, targetSizeFactor, n);
      while (output->size() > targetSize) {
        output->coarsen();
      }
    }

    // If this process is a receiver.
    if (commRank < n / 2) {
      int const sender = commRank + n / 2;
      // If there is a sender.
      if (sender < commSize) {
        // Get the number of levels in the cells that we will receive.
        std::size_t numLevels;
        mpi::sendRecv(output->numLevels(), &numLevels, sender, numLevelsTag,
                      comm);
        // Coarsen if necessary to match the number of levels of refinement.
        while (output->numLevels() > numLevels) {
          output->coarsen();
        }
        // Receive a group of cells. Note that trying to overlap this
        // communication with the preceding coarsening operations does
        // not improve performance.
        recv(&received, sender, cellsTag, comm);
        assert(received.numLevels() == output->numLevels());
        // Merge into the output.
        *output += received;
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      int const receiver = commRank - n / 2;
      // Get the number of levels in the process to which we will send.
      std::size_t numLevels;
      mpi::sendRecv(output->numLevels(), &numLevels, receiver, numLevelsTag,
                    comm);
      // Coarsen if necessary to match the number of levels of refinement.
      while (output->numLevels() > numLevels) {
        output->coarsen();
      }
      // Send the cells.
      send(*output, receiver, cellsTag, comm);
    }
    ++numLevelsTag;
    ++cellsTag;
  }

  // Only the root process holds the merged cells.
  if (commRank == 0) {
    while (output->size() > maxCells) {
      output->coarsen();
    }
  }
  else {
    output->clear();
  }
}


template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
mergeCoarsenDisjoint
(const UniformCells<_Traits, _Cell, _StoreDel>& input,
 const std::size_t maxCells,
 UniformCells<_Traits, _Cell, _StoreDel>* output,
 const MPI_Comm comm)
{
  assert(maxCells != 0);

  int const commSize = mpi::commSize(comm);
  int const commRank = mpi::commRank(comm);

  // Copy the input.
  *output = input;

  // A power of two that is at least the number of processes.
  int n = 1;
  while (n < commSize) {
    n *= 2;
  }

  // Perform initial coarsening.
  // Coarsen if necessary to not exceed the maximum number of cells.
  while (output->size() > maxCells) {
    output->coarsen();
  }
  {
    // Determine the minimum number of levels across processes.
    std::size_t const numLevels = mpi::allReduce(output->numLevels(), MPI_MIN,
                                                 comm);
    // Synchronize the levels.
    while (output->numLevels() > numLevels) {
      output->coarsen();
    }
  }

  // Get total number of distributed cells. While it exceeds twice the maximum 
  // number of allowed cells, coarsen on all processes. We use a factor
  // of two because the distributed cells are only roughly disjoint.
  while (output->numLevels() != 0 &&
         mpi::allReduce(output->size(), MPI_SUM, comm) > 2 * maxCells) {
    output->coarsen();
  }

  int tag = 0;
  UniformCells<_Traits, _Cell, _StoreDel>
    received(input.lowerCorner(), input.lengths(), input.numLevels());
  for (; n > 1; n /= 2) {
    // If this process is a receiver.
    if (commRank < n / 2) {
      int const sender = commRank + n / 2;
      // If there is a sender.
      if (sender < commSize) {
        // Receive a group of cells.
        recv(&received, sender, tag, comm);
        assert(received.numLevels() == output->numLevels());
        // Merge into the output.
        *output += received;
      }
    }
    // If this process is a sender.
    else if (commRank < n) {
      int const receiver = commRank - n / 2;
      // Send the cells.
      send(*output, receiver, tag, comm);
      output->clear();
    }
    ++tag;
  }

  // Only the root process holds the merged cells.
  if (commRank == 0) {
    while (output->size() > maxCells) {
      output->coarsen();
    }
  }
}


//! Partition the cells without distributing them. Coarsen if necessary.
/*! We use mergeCoarsen() to obtain coarsened cells at the root. The maximum
  allowed number of cells in this data structure is the number of processes
  divided by the imbalance goal. If necessary, coarsen the local set of cells
  to match the number of levels of refinement in the global set. (Without
  this step, the partition code delimiters would not match the codes in
  the local data structure.)

  \note If the distribution of objects is inhomogeneous, decrease the
  imbalance goal to account for the inhomogenous distribution in the
  number of objects per cell.
*/
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
_partitionCoarsen
(UniformCells<_Traits, _Cell, _StoreDel>* local,
 Partition<_Traits>* codePartition, MPI_Comm comm,
 const double imbalanceGoal = 0.01)
{
  assert(0 < imbalanceGoal && imbalanceGoal < 1);

  int const commSize = mpi::commSize(comm);
  assert(codePartition->size() == std::size_t(commSize));
  int const commRank = mpi::commRank(comm);

  // Obtained coarsened, merged cells on the root process.
  const std::size_t maxCells =
    commSize * (std::size_t(1) << _Traits::Dimension) / imbalanceGoal;
  UniformCells<_Traits, _Cell, _StoreDel>
    coarsened(local->lowerCorner(),
      local->lengths(),
      local->numLevels());
  mergeCoarsen(*local, maxCells, &coarsened, comm);

  // Set the number of levels of refinement to that in the merged, coarsened
  // set of cells.
  {
    std::size_t numLevels = coarsened.numLevels();
    // Broadcast the number of levels of refinement from the root process.
    mpi::bcast(&numLevels, comm);
    // Coarsen if necessary.
    local->setNumLevels(numLevels);
  }

  // Partition the cells on the root process.
  if (commRank == 0) {
    (*codePartition)(coarsened);
  }
  // Broadcast the partitioning.
  mpi::bcastNoResize(&codePartition->delimiters, comm);
}


//! Calculate a partitioning of the cells.
/*! Partition the cells on the root process and then broadcast the
  partitioning. The cell type must have a \c size() accessor.
 */
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
_calculatePartitionFromGlobalCells
(UniformCells<_Traits, _Cell, _StoreDel> const& global,
 Partition<_Traits>* codePartition, MPI_Comm const comm)
{
  assert(codePartition->size() == std::size_t(mpi::commSize(comm)));

  // Partition the cells on the root process.
  if (mpi::commRank(comm) == 0) {
    (*codePartition)(global);
  }
  // Broadcast the partitioning.
  mpi::bcastNoResize(&codePartition->delimiters, comm);
}


//! Partition the cells without distributing them.
/*! We use merge() to obtain cells at the root. */
template<typename _Traits, typename _Cell, bool _StoreDel>
inline
void
_partition
(UniformCells<_Traits, _Cell, _StoreDel> const& local,
 Partition<_Traits>* codePartition, MPI_Comm const comm)
{
  assert(codePartition->size() == std::size_t(mpi::commSize(comm)));

  // Obtained merged cells on the root process.
  UniformCells<_Traits, _Cell, _StoreDel>
    merged(local.lowerCorner(), local.lengths(), local.numLevels());
  merge(local, &merged, comm);

  _calculatePartitionFromGlobalCells(merged, codePartition, comm);
}


template<typename _Traits, typename _Cell, typename _Object>
inline
void
distribute(UniformCells<_Traits, _Cell, true>* localCells,
           std::vector<_Object>* objects,
           Partition<_Traits> const& codePartition, MPI_Comm comm)
{
  // Distribute the objects without sorting the new local objects.
  distributeNoSort(*localCells, objects, codePartition, comm);
  // Rebuild to sort the objects and get the new local cells.
  localCells->buildCells(objects);
}


template<typename _Traits, typename _Cell, typename _Object>
inline
void
distribute(UniformCells<_Traits, _Cell, true> const& localCells,
           std::vector<_Object>* objects,
           Partition<_Traits> const& codePartition, MPI_Comm comm)
{
  // Distribute the objects without sorting the new local objects.
  distributeNoSort(localCells, objects, codePartition, comm);
  // Sort the objects.
  sortByCodes(localCells.order(), objects);
}


template<typename _Traits, typename _Cell, typename _Object>
inline
void
partitionCoarsen
(UniformCells<_Traits, _Cell, true>* cells,
 std::vector<_Object>* objects,
 Partition<_Traits>* codePartition, MPI_Comm const comm,
 double const imbalanceGoal)
{
  _partitionCoarsen(cells, codePartition, comm, imbalanceGoal);
  distribute(cells, objects, *codePartition, comm);
}


template<typename _Traits, typename _Object>
inline
void
partition(DiscreteCoordinates<_Traits> const& discreteCoordinates,
          std::vector<_Object>* objects,
          Partition<_Traits>* codePartition, MPI_Comm const comm)
{
  // Build a cell data structure with element counts.
  UniformCells<_Traits, void, true>
    cells(discreteCoordinates.lowerCorner(), discreteCoordinates.lengths(),
          discreteCoordinates.numLevels());
  // Note that this first sorts the local objects.
  cells.buildCells(objects);
  // Determine the partitioning.
  _partition(cells, codePartition, comm);
  // Distribute and sort the objects.
  distribute(&cells, objects, *codePartition, comm);
}


template<typename _Traits, typename _Cell, typename _Object>
inline
void
partitionOrderedObjects
(UniformCells<_Traits, _Cell, true> const& globalCells,
 std::vector<_Object>* objects,
 Partition<_Traits>* codePartition, MPI_Comm const comm)
{
  performance::Performance& perf = performance::getInstance();
  perf.beginScope("partitionOrderedObjects()");

  perf.start("Build a cell data structure with element counts");
  UniformCells<_Traits, void, true>
    localCells(globalCells.lowerCorner(), globalCells.lengths(),
               globalCells.numLevels());
  localCells.buildCells(*objects);
  perf.stop();

  perf.start("Determine the partitioning");
  _calculatePartitionFromGlobalCells(globalCells, codePartition, comm);
  perf.stop();

  perf.start("Distribute and sort the objects");
  distribute(localCells, objects, *codePartition, comm);
  perf.stop();

  perf.endScope();
}


} // namespace sfc
} // namespace stlib
