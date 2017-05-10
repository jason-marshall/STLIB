// -*- C++ -*-

/*!
  \file particle/order.h
  \brief Use Morton codes to order particles.
*/

#if !defined(__particle_orderMpi_h__)
#define __particle_orderMpi_h__

#include "stlib/particle/cellData.h"
#include "stlib/particle/order.h"
#include "stlib/particle/set.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/MultiIndexRangeIterator.h"
#include "stlib/mpi/wrapper.h"
#if 0
// CONTINUE REMOVE
#include "stlib/ext/pair.h"
#include "stlib/ext/vector.h"
#endif

#include <list>
#include <set>
#include <map>
#include <unordered_set>

#include <mpi.h>

namespace stlib
{
namespace particle
{

//! Use Morton codes to order distributed particles.
/*!
  \param _Traits The traits class. Use PlainTraits, PeriodicTraits, or a
  class with equivalent functionality.
*/
template<typename _Traits>
class MortonOrderMpi :
  public MortonOrder<_Traits>
{
  //
  // Constants.
  //
public:

  //! The Dimension of the space.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;
  //! Whether the domain is periodic.
  BOOST_STATIC_CONSTEXPR bool Periodic = _Traits::Periodic;

  //
  // Types.
  //
public:

  //! The particle type.
  typedef typename _Traits::Particle Particle;
  //! The floating-point number type.
  typedef typename _Traits::Float Float;
  //! A Cartesian point.
  typedef typename TemplatedTypes<Float, Dimension>::Point Point;

private:

  //! The base class.
  typedef MortonOrder<_Traits> Base;

protected:

  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;
  //! A discrete coordinate.
  typedef IntegerTypes::DiscreteCoordinate DiscreteCoordinate;
  //! A discrete point with integer coordinates.
  typedef typename TemplatedTypes<Float, Dimension>::DiscretePoint
  DiscretePoint;

  //
  // Enumerations.
  //
private:

  enum {ReorderParticleTag, PartitionShiftTag, ReduceTableTag,
        DuCountTag, DuParticleTag, BepCellCountTag, BepCodeSizeTag, BepCodeTag,
        BepCountTag, EcCodeTag, EpParticleTag
       };

  //
  // Nested classes.
  //
private:

  //! Information about the particles we receive from a process.
  struct ReceivingInfo {
    //! The source process index.
    std::size_t source;
    //! The position in the particles vector to put the particles.
    std::size_t position;
    //! The number of particles that we will receive.
    std::size_t size;
  };

  //
  // Member data.
  //
public:

  // These are set in _buildExchangePattern().
  //! The cells that are not exchanged with other processes.
  std::vector<std::size_t> interior;
  //! The cells that are exchanged with other processes.
  std::vector<std::size_t> exchanged;
  //! The cells that belong to other processes.
  std::vector<std::size_t> shadow;

  //! The maximum allowed load imbalance.
  /*! The default value is 0.01. If modified, it must be set to the same value
    on all processes. */
  Float maxLoadImbalance;

private:

  // The communicator and partition.

  //! The intra-communicator for the particles.
  MPI_Comm _comm;
  //! The sequence of code delimiters.
  /*! The nth and (n+1)th values define the range for the nth process.
    The lower bound is closed, while the upper is open.*/
  std::vector<Code> _delimiters;
  //! The total number of adjacent neighbors for the local cells.
  std::size_t _totalNumAdjacentNeighbors;

  // Data structures for exchanging particles.

  MPI_Datatype _mpiCodeType;
  MPI_Datatype _mpiSizeType;
  MPI_Datatype _mpiFloatType;
  //! The width of the shadow region for particles that are exchanged.
  Float _shadowWidth;
  //! The index offsets for cells that are within the shadow width.
  std::vector<std::array<std::ptrdiff_t, Dimension> >
  _shadowIndexOffsets;
  //! Whether the exchange pattern is defined.
  bool _isExchangePatternDefined;
  // The index of the first local cell.
  std::size_t _localCellsBegin;
  // One past the index of the last local cell.
  std::size_t _localCellsEnd;
  //! For each process from which we receive particles, the source, position, and size.
  std::vector<ReceivingInfo> _receiving;
  //! For each process, the cells to send.
  container::PackedArrayOfArrays<std::size_t> _processCellLists;
  //! The processes to which we will send a nonzero number of particles.
  std::vector<std::size_t> _sendProcessIds;
  //! The particle send buffers. One buffer for each non-empty send.
  container::PackedArrayOfArrays<Particle> _particleSendBuffers;


  //! The load imbalance following the last partition.
  /*! Note that this is only defined on the root process. */
  Float _startingImbalance;
  //! The number of times the particles have been partitioned.
  std::size_t _partitionCount;
  //! The number of times the particles have been reordered.
  std::size_t _reorderCount;
  //! The number of times the data structure has been repaired.
  std::size_t _repairCount;
  //! A timer for measuring time spent in various functions.
  performance::SimpleTimer _timer;
  //! The time spent reordering.
  double _timeReorder;
  //! The time spent partitioning.
  double _timePartition;
  //! The time spent distributing particles.
  double _timeDistributeUnordered;
  //! The number of particles sent in exchanges.
  double _numDistributeSent;
  //! The time spent building the exchange pattern.
  double _timeBuildExchangePattern;
  //! The time spent posting the sends and receives for exchanging particles.
  double _timeExchangePost;
  //! The time spent waiting for the sends and receives to complete.
  double _timeExchangeWait;
  //! The number of neighbors to whom we send particles.
  double _numNeighborsSend;
  //! The number of neighbors to whom we send particles.
  double _numNeighborsReceive;
  //! The number of particles sent in exchanges.
  double _numExchangeSent;
  //! The number of particles received in exchanges.
  double _numExchangeReceived;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the communicator and the geometry.
  /*! If the value for the padding is omitted, a suitable value will be
    chosen. */
  MortonOrderMpi(MPI_Comm comm,
                 const geom::BBox<Float, Dimension>& domain,
                 Float interactionDistance, Float shadowWidth,
                 Float padding = std::numeric_limits<Float>::quiet_NaN());

  //! Default constructor invalidates the data members relating to geometry.
  MortonOrderMpi(MPI_Comm comm);

  //! Initialize from the domain, the interaction distance, and the padding.
  /*! If the value for the padding is omitted, a suitable value will be
    chosen. */
  void
  initialize(geom::BBox<Float, Dimension> domain,
             const Float interactionDistance, const Float shadowWidth,
             const Float padding = std::numeric_limits<Float>::quiet_NaN())
  {
    Base::initialize(domain, interactionDistance, padding);
    _shadowWidth = shadowWidth;
    _buildShadowIndexOffsets();
    _checkGeometry();
  }

  //! Check that the data structure is valid.
  void
  checkValidity() const;

private:

  void
  _defineMpiTypes();

  void
  _buildShadowIndexOffsets();

  // Check that the domains, interaction distances, etc. are the same
  // across processes.
  void
  _checkGeometry();

  //@}
  //--------------------------------------------------------------------------
  /*! \name Order particles.
    When used, all of these functions must be called on all processes.
  */
  //@{
public:

  //! Set the particles.
  /*! \param begin The beginning of a sequence of particles.
    \param end One past the end of a sequence of particles.

    Set the particles in the local process. Determine a partitioning.
    Distribute the particles. Call exchangeParticles() before using
    neighbors. */
  template<typename _InputIterator>
  void
  setParticles(_InputIterator begin, _InputIterator end);

  //! Repair the data structure if necessary. Rebalance the load if necessary.
  /*! Return true if it is repaired.
    If repaired, call exchangeParticles() before using neighbors. */
  bool
  repair();

  //! Exchange particles with the neighboring processes.
  void
  exchangeParticles();

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors/Manipulators.
  //@{
public:

  //! The number of non-empty, local cells.
  std::size_t
  localCellsSize() const
  {
    return _localCellsEnd - _localCellsBegin;
  }

  //! The index of the first local cell.
  std::size_t
  localCellsBegin() const
  {
    return _localCellsBegin;
  }

  //! One past the index of the last local cell.
  std::size_t
  localCellsEnd() const
  {
    return _localCellsEnd;
  }

  //! The first index of the local particles.
  std::size_t
  localParticlesBegin() const
  {
    return Base::cellBegin(_localCellsBegin);
  }

  //! One past the last index of the local particles.
  std::size_t
  localParticlesEnd() const
  {
    return Base::cellBegin(_localCellsEnd);
  }

  //! The number of times the particles have been partitioned.
  std::size_t
  partitionCount() const
  {
    return _partitionCount;
  }

  //! The number of times the particles have been reordered.
  std::size_t
  reorderCount() const
  {
    return _reorderCount;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  //! Print information about the data structure.
  void
  printInfo(std::ostream& out) const;

  //! Print performance information.
  void
  printPerformanceInfo(std::ostream& out) const;

  //! Print the cell data to a VTK file.
  void
  printCellDataVtk(std::ostream& out, const std::size_t MaxImageExtent = 128)
  const
  {
    _printCellDataVtk(out, MaxImageExtent,
                      std::integral_constant<std::size_t, Dimension>());
  }

private:

  //! Print the cell data to a VTK file.
  /*! Specialization for 3-D. Other dimensions are not supported. */
  void
  _printCellDataVtk(std::ostream& out, std::size_t MaxImageExtent,
                    std::integral_constant<std::size_t, 3> /*Dimension*/) const;

  //! Make a table of the cell data.
  void
  _cellDataTable(std::vector<std::pair<Code, CellData> >* table) const;

  //@}
private:

  //
  // Order particles.
  //

  //! Return true if no particle has moved farther than the allowed padding.
  bool
  isOrderValid() const;

  //! Return true if the load is balanced to within maxLoadImbalance.
  /*! \note This function is no longer used. */
  bool
  isParticleLoadBalanced();

  //! Return true if the load is balanced to within maxLoadImbalance.
  bool
  isNeighborsLoadBalanced();

  //! Rebalance the load if necessary.
  /*! Return true if it is rebalanced.
    If rebalanced, call exchangeParticles() before using neighbors.*/
  bool
  _balanceLoad();

  //! Reorder the particles.
  /*!
    Recalculate the codes. Reorder the particles. Record the new starting
    positions.
  */
  void
  reorder();

  //! Perform a binary reduction of the tables of cell data.
  /*! \return The number of levels the table was shifted. */
  template<typename _T>
  std::size_t
  _reduce(std::vector<std::pair<Code, _T> >* table,
          std::size_t maxBufferSize) const;

  // Partition using the number of particles. Note that this is only used
  // for an initial partitioning before we have neighbor information.
  void
  _partitionByParticles(Float accuracyGoal = 0.01);

  // Partition using the number of potential neighbors.
  void
  _partitionByNeighbors(Float accuracyGoal = 0.01);

  // Partition using the provided costs per cell.
  void
  _partition(std::vector<std::pair<Code, std::size_t> >* costs,
             Float accuracyGoal = 0.01);

  //! Distribute particles, and determine the exchange pattern.
  void
  _distributePattern();

  //! Distribute the particles according to the partition defined by the code delimiters.
  void
  _distributeUnordered();

  //! Calculate the list of codes for the neighboring cells.
  /*! Cells within the shadow width will be added. */
  void
  _neighborCellCodes(const std::size_t cellIndex,
                     std::vector<Code>* neighborCodes)
  const
  {
    _neighborCellCodes(cellIndex, neighborCodes,
                       std::integral_constant<bool, Periodic>());
  }

  //! Calculate the list of codes for the neighboring cells.
  void
  _neighborCellCodes(std::size_t cellIndex, std::vector<Code>* neighborCodes,
                     std::false_type /*Periodic*/) const;

  //! Calculate the list of codes for the neighboring cells.
  void
  _neighborCellCodes(std::size_t cellIndex, std::vector<Code>* neighborCodes,
                     std::true_type /*Periodic*/) const;

  //! Record the processes to which we should send the cell.
  /*!
    \param cellIndex The index of the cell.
    \param processes The output sequence of process indices.
  */
  void
  _sendToProcessList(std::size_t cellIndex,
                     std::vector<std::size_t>* processes) const;

  //! Record the processes to which we should send the cell.
  /*!
    \param neighborCodes The codes for the neighboring cells.
    \param processes The output sequence of process indices.
  */
  void
  _sendToProcessList(std::vector<Code>* neighborCodes,
                     std::vector<std::size_t>* processes) const;

  //! Determine the exchange pattern for sharing neighboring particles.
  void
  _buildExchangePattern();

  //! Clear the exchange pattern for sharing neighboring particles.
  void
  _clearExchangePattern();

  //
  // Accessors.
  //

  //! Return the index of the process that holds the specified code.
  std::size_t
  _process(const Code code) const
  {
    return std::distance(_delimiters.begin(),
                         std::upper_bound(_delimiters.begin(),
                                          _delimiters.end(), code)) - 1;
  }

};


} // namespace particle
}

#define __particle_partition_tcc__
#include "stlib/particle/partition.tcc"
#undef __particle_partition_tcc__

#define __particle_orderMpi_tcc__
#include "stlib/particle/orderMpi.tcc"
#undef __particle_orderMpi_tcc__

#endif
