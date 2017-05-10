// -*- C++ -*-

#if !defined(__stlib_sfc_AdaptiveCellsMpi_tcc__)
#error This file is an implementation detail of AdaptiveCellsMpi.
#endif

namespace stlib
{
namespace sfc
{


/// Build a AdaptiveCells from the distributed objects.
/**
  \param objects The distributed objects.
  \param maxObjectsPerCell The maximum allowed number of objects per cell.
  Only cells at the highest level of refinement can have more that this
  number of objects.
  \param output The output cells data structure is only built on process 0.
  \param comm The MPI communicator.
*/
template<typename _Traits, bool _StoreDel, typename _Object>
inline
AdaptiveCells<_Traits, void, _StoreDel>
buildAdaptiveBlocksFromDistributedObjects
(typename AdaptiveCells<_Traits, void, _StoreDel>::Grid const& grid,
 std::vector<_Object> const& objects,
 std::size_t const maxObjectsPerCell,
 MPI_Comm const comm)
{
  typedef typename _Traits::Code Code;
  typedef std::pair<Code, std::size_t> Pair;
  typedef typename _Traits::BBox BBox;

  // Make a vector of the object codes.
  std::vector<Code> objectCodes(objects.size());
  for (std::size_t i = 0; i != objectCodes.size(); ++i) {
    objectCodes[i] = grid.code(centroid(geom::specificBBox<BBox>(objects[i])));
  }
  // Determine the highest required level given the maximum elements per
  // cell. First we calculate the value locally. Note that this partially
  // sorts the object codes, but does not modify the code values.
  std::size_t level =
    refinementSortCodes(grid, &objectCodes, maxObjectsPerCell);
  // Then take the maximum of the distributed levels.
  level = mpi::allReduce(level, MPI_MAX, comm);

  // Set the level in the object codes.
  for (auto&& code : objectCodes) {
    code = grid.atLevel(code, level);
  }
  // Sort the result so that we can effectively convert it to a cell-based
  // representation.
  std::sort(objectCodes.begin(), objectCodes.end());
  // Produce a sorted vector of code/count pairs on process 0.
  std::vector<Pair> allPairs;
  {
    // Convert from codes to code-count pairs.
    std::vector<Pair> pairs;
    objectCodesToCellCodeCountPairs<_Traits>(objectCodes, &pairs);
    // Merge the sorted sequences to obtain the full list on process 0.
    mpi::mergeSorted(pairs, &allPairs, comm);
  }

  AdaptiveCells<_Traits, void, _StoreDel> output(grid);
  // Determine the codes for a AdaptiveCells on process 0.
  if (mpi::commRank(comm) == 0) {
    std::vector<Pair> cellCodeSizePairs;
    buildFromBlockCodeSizePairs(grid, allPairs, maxObjectsPerCell, 
                                &cellCodeSizePairs);
    output.buildCells(cellCodeSizePairs);
  }
  return output;
}


#if 0
// Not currently used.
/// Build local cells. Reduce to build global cells, then broadcast them.
/**
   \param objects The sequence of objects will be reordered in building the
   local cells.
   \param globalCells The local cells are reduced to form the global cells.
   \param comm The MPI communicator.
   \param accuracyGoal The accuracy goal for partitioning the objects.
*/
template<typename _Object, typename _Traits, typename _Cell, bool _StoreDel>
inline
void
adaptiveCells(std::vector<_Object>* objects,
                     AdaptiveCells<_Traits, _Cell, _StoreDel>* globalCells,
                     MPI_Comm comm,
                     double accuracyGoal = 0.01)
{
  // Determine an appropriate maximum number of objects per cell given 
  // the number of objects and the number of MPI processes.
  std::size_t const maxObjectsPerCell = 
    maxObjectsPerCellForPartitioning(objects->size(), comm, accuracyGoal);
  // Build the global cell data structure.
  Partition<_Traits> codePartition(mpi::commSize(comm));
  adaptiveCells(objects, maxObjectsPerCell, globalCells, &codePartition,
                       comm);
}
#endif


// Note: If you pass a null pointer for the partition, it will not be computed.
template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
adaptiveCells(typename _AdaptiveCells::Grid const& grid,
                std::vector<_Object>* objects,
                std::size_t const maxObjectsPerCell,
                _AdaptiveCells* localCells,
                Partition<typename _AdaptiveCells::Traits>* codePartition,
                MPI_Comm const comm)
{
  typedef typename _AdaptiveCells::Traits Traits;
  typedef typename Traits::Code Code;

  assert(maxObjectsPerCell > 0);
  assert(localCells);

  // The adaptive blocks define the block codes for the global cells.
  // Build the adaptive blocks. The result is only valid on process 0.
  // We store the delimiters because we may need them for calculating the 
  // partitioning.
  AdaptiveCells<Traits, void, true> adaptiveBlocks =
    buildAdaptiveBlocksFromDistributedObjects<Traits, true>
    (grid, *objects, maxObjectsPerCell, comm);
  // Broadcast the adaptive blocks.
  bcast(&adaptiveBlocks, comm);

  if (codePartition) {
    // Use the blocks and associated cell size counts to determine a 
    // partitioning of the blocks.
    if (stlib::mpi::commRank(comm) == 0) {
      (*codePartition)(adaptiveBlocks);
    }
    // Broadcast the partitioning.
    stlib::mpi::bcastNoResize(&codePartition->delimiters, comm);
  }

  // Calculate codes and sort the local objects.
  std::vector<Code> localCodes;
  sort(grid, objects, &localCodes);
  // Build the local cells.
  *localCells = _AdaptiveCells(grid);
  localCells->buildCells(adaptiveBlocks.codesWithGuard(), localCodes, *objects);

  // Perform a reduction to build the coarse global boundary on process 0.
  _AdaptiveCells globalCells = _AdaptiveCells(grid);
  reduceCompatible(*localCells, &globalCells, comm);
  // Broadcast to obtain the coarse global boundary on all processes.
  bcast(&globalCells, comm);
  return globalCells;
}


// CONTINUE Specify the grid, return the global cells.
template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
distribute(typename _AdaptiveCells::Grid const& grid,
           std::vector<_Object>* objects,
           std::size_t const maxObjectsPerCell,
           Partition<typename _AdaptiveCells::Traits>* codePartition,
           MPI_Comm const comm)
{
  assert(codePartition);

  performance::Scope _("sfc::distribute() for AdaptiveCells");
  performance::record("Initial objects", objects->size());
  performance::start("adaptiveCells()");

  // Build the local and global cells. In the process, determine a partitioning
  // of the global cells.
  _AdaptiveCells localCells;
  _AdaptiveCells const globalCells =
    adaptiveCells(grid, objects, maxObjectsPerCell, &localCells,
                    codePartition, comm);

  performance::stop();
  performance::start("distributeNoSort()");

  // Distribute the objects.
  distributeNoSort(localCells, objects, *codePartition, comm);

  performance::stop();
  performance::record("maxObjectsPerCell", maxObjectsPerCell);
  performance::record("Redistributed objects", objects->size());
  performance::record("Local cells", localCells.size());
  performance::record("Global cells", globalCells.size());

  return globalCells;
}


template<typename _Traits>
inline
std::size_t
maxObjectsPerCellForPartitioning(std::size_t const numLocal,
                                 MPI_Comm const comm,
                                 double const accuracyGoal)
{
  std::size_t maxObjectsPerCell;
  // First determine the result on process 0.
  {
    // Determine the total number of objects on process 0.
    std::size_t const totalNumberOfObjects =
      mpi::reduce(numLocal, MPI_SUM, comm);
    if (mpi::commRank(comm) == 0) {
      std::size_t const targetCells =
        std::max(std::size_t(double(mpi::commSize(comm)) / accuracyGoal),
                 std::size_t(1));
      std::size_t const averageObjectsPerCell =
        totalNumberOfObjects / targetCells;
      maxObjectsPerCell = std::max(averageObjectsPerCell *
                                   (1 << _Traits::Dimension),
                                   std::size_t(1));
    }
  }
  // Then broadcast it.
  mpi::bcast(&maxObjectsPerCell, comm);
  return maxObjectsPerCell;
}


// This algorithm for computing the max objects per cell is no longer used.
#if 0
/// Determine an appropriate maximum number of objects per cell.
/**
   \param numLocal The number of local objects.
   \param comm The MPI communicator.
   \param targetObjectsPerCell The number of objects per cell that would result
   in the desired resolution.
   \param minCells The (approximate) minimum number of cells that are 
   acceptable.
   \param maxCells The (approximate) maximum number of cells that are 
   acceptable.
   
   The purpose of this function is to balance the competing concerns of 
   accuracy and speed. As the number of objects per cell decreases, the 
   accuracy of the cell data structure increases, but the costs of building
   and working with it also increase. You specify the desired accuracy with 
   targetObjectsPerCell. You specify the maximum acceptable cost with maxCells.
   If the estimate of the number of cells does not exceed maxCells, then 
   this function simply returns targetObjectsPerCell. Otherwise, it 
   calculates the number of objects per cell that would result in 
   approximately the maximum allowed number of cells.
*/
template<typename _Traits>
inline
std::size_t
maxObjectsPerCell(std::size_t const numLocal,
                  MPI_Comm const comm,
                  std::size_t const targetObjectsPerCell = 128,
                  std::size_t const minCells = 1 << 10,
                  std::size_t const maxCells = 1 << 19)
{
  // Determine the total number of objects.
  std::size_t const totalNumberOfObjects =
    mpi::allReduce(numLocal, MPI_SUM, comm);
  // Lower the maximum objects per cell if necessary to give a reasonable number
  // of cells for simple problems.
  std::size_t const lower =
    totalNumberOfObjects / minCells * (1 << _Traits::Dimension);
  if (lower < targetObjectsPerCell) {
    return std::max(lower, std::size_t(1));
  }
  // Raise the maximum objects per cell if necessary to limit the total 
  // number of cells.
  return std::max(totalNumberOfObjects / maxCells * (1 << _Traits::Dimension),
                  targetObjectsPerCell);
}
#endif


template<typename _AdaptiveCells, typename _Object>
inline
_AdaptiveCells
distribute(typename _AdaptiveCells::Grid const& grid,
           std::vector<_Object>* objects,
           Partition<typename _AdaptiveCells::Traits>* codePartition,
           MPI_Comm const comm,
           double const accuracyGoal)
{
  // Determine an appropriate maximum number of objects per cell given the 
  // accuracy goal.
  std::size_t const maxObjectsPerCell =
    maxObjectsPerCellForPartitioning<typename _AdaptiveCells::Traits>
    (objects->size(), comm, accuracyGoal);
  // Distribute using the maximum number of objects per cell parameter.
  return distribute<_AdaptiveCells>(grid, objects, maxObjectsPerCell,
                                      codePartition, comm);
}


template<typename _Object, typename _Float, std::size_t _Dimension>
inline
void
distribute(std::vector<_Object>* objects,
           geom::BBox<_Float, _Dimension> const& domain, MPI_Comm const comm,
           double const accuracyGoal)
{
  typedef Traits<_Dimension, _Float> _Traits;
  typedef AdaptiveCells<_Traits, void, true> _AdaptiveCells;
  assert(! isEmpty(domain));
  typename _AdaptiveCells::Grid const grid(domain, 0);
  Partition<_Traits> codePartition(mpi::commSize(comm));
  // Note that we ignore the returned global cells.
  distribute<_AdaptiveCells>(grid, objects, &codePartition, comm,
                               accuracyGoal);
}


// CONTINUE Infer the floating-point number type and space dimension from the
// object.
template<typename _Float, std::size_t _Dimension, typename _Object>
inline
void
distribute(std::vector<_Object>* objects, MPI_Comm const comm,
           double const accuracyGoal)
{
  typedef geom::BBox<_Float, _Dimension> BBox;

  // Calculate a domain that contains the objects.
  BBox const domain =
    mpi::allReduce(geom::specificBBox<BBox>(objects->begin(), objects->end()),
                   comm);
  // Check for the trivial case of no objects before trying to distribute them.
  if (! isEmpty(domain)) {
    // Pass on the task of distributing the objects.
    distribute(objects, domain, comm, accuracyGoal);
  }
}


} // namespace sfc
} // namespace stlib
