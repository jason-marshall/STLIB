// -*- C++ -*-

/*!
  \file particle/subset.h
  \brief Neighbors are stored as a subset of the common neighbors of a cell.
*/

#if !defined(__particle_subset_h__)
#define __particle_subset_h__

#include "stlib/particle/types.h"
#include "stlib/particle/neighbors.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/numerical/integer/bits.h"
#include "stlib/numerical/partition.h"
#include "stlib/simd/shuffle.h"

#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace particle
{

//! Neighbors are stored as a subset of the common neighbors of a cell.
/*!
  \param _Order The class for ordering the particles.
*/
template<typename _Order>
class SubsetUnionNeighbors :
  public NeighborsBase<_Order>, NeighborsPerformance
{
private:

  //! The base class.
  typedef NeighborsBase<_Order> Base;

  //
  // Constants.
  //
private:

  //! The Dimension of the space.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Order::Dimension;
  //! The number of cells in a 3^Dimension block.
  BOOST_STATIC_CONSTEXPR std::size_t NumAdjacentCells =
    numerical::Exponentiation<std::size_t, 3, Dimension>::Result;
  //! Whether the domain is periodic.
  BOOST_STATIC_CONSTEXPR bool Periodic = _Order::Periodic;

  //
  // Types.
  //
public:

  //! The unsigned integer type for storing bit masks.
  typedef unsigned char Mask;
  //! The number of digits in the mask type.
  BOOST_STATIC_CONSTEXPR std::size_t MaskDigits =
    std::numeric_limits<Mask>::digits;

private:

  //! The floating-point number type.
  typedef typename _Order::Float Float;
  //! A Cartesian point.
  typedef typename TemplatedTypes<Float, Dimension>::Point Point;
  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;
  //! A discrete point with integer coordinates.
  typedef typename TemplatedTypes<Float, Dimension>::DiscretePoint
  DiscretePoint;
  //! The representation of a neighbor.
  typedef particle::Neighbor<Periodic> Neighbor;
  //! The representation of a contiguous range of neighbors.
  typedef particle::NeighborRange<Periodic> NeighborRange;
  //! A cell's list of adjacent cells.
  typedef std::array<NeighborRange, NumAdjacentCells> AdjacentList

  //
  // Member data.
  //
public:

  //! For each cell, the union of interacting neighbors.
  container::PackedArrayOfArrays<Neighbor> unionNeighbors;
  //! For each particle, the bit masks that record actual neighbors.
  container::PackedArrayOfArrays<Mask> neighborMasks;

private:

  using Base::_order;
  using Base::_cellsBegin;
  using Base::_cellsEnd;
  //! For each cell, represent the particles in the adjacent cells.
  std::vector<AdjacentList> _adjacentCells;
  //! For each particle, the index of the list of potential neighbors.
  std::vector<std::size_t> _cellListIndices;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the data structure that orders the particles.
  SubsetUnionNeighbors(const _Order& order);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Count the total number of actual neighbors using the bit masks.
  std::size_t
  numNeighbors() const;

  //! Count the number of actual neighbors for the specified particle.
  /*! Use the bit masks. */
  std::size_t
  numNeighbors(std::size_t i) const;

  //! Return the number of potential neighbors for the specified particle.
  /*! \note The potential neighbors are padded to be a multiple of the number
    of bit mask digits. */
  std::size_t
  numPotentialNeighbors(const std::size_t particle) const
  {
    return neighborMasks.size(particle) * MaskDigits;
  }

  //! Get the potential neighbor indices and positions for the specified particle.
  /*! \note The particle itself is included in the output. */
  template<typename _Allocator1, typename _Allocator2>
  void
  potentialNeighbors(std::size_t particle,
                     std::vector<std::size_t, _Allocator1>* indices,
                     std::vector<Point, _Allocator2>* positions) const;

  //! Return true if the potential neighbor is actually a neighbor.
  /*! Use the bit-masks that record the actual neighbors. */
  bool
  isNeighbor(std::size_t particle, std::size_t index) const;

private:

  //! Count the number of neighbors in the range of masks.
  std::size_t
  popCount(const Mask* masks, std::size_t size) const;

  //! Use unionNeighbors to count the neighbors.
  /*! \note The potential neighbors are not padded. */
  std::size_t
  _countPotentialNeighbors(const std::size_t particle) const
  {
    return unionNeighbors.size(_cellListIndices[particle]);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Calculate neighbors.
  //@{
public:

  //! Find the potential neighbors for each particle.
  void
  findPotentialNeighbors()
  {
    findPotentialNeighbors(0, _order.cellsSize());
  }

  //! Find the potential neighbors for each particle.
  /*!
    \param localCellsBegin The index of the first local cell.
    \param localCellsEnd One past the index of the last local cell.

    Store the particles that are in the adjacent cells. This function sets
    values for the _adjacentCells and _cellListIndices data members.
  */
  void
  findPotentialNeighbors(std::size_t localCellsBegin,
                         std::size_t localCellsEnd);

  //! Use the potential neighbors to find the actual neighbors.
  void
  findNeighbors();

private:

  //! Use the potential neighbors to find the actual neighbors.
  void
  _findNeighbors(std::size_t begin, std::size_t end)
  {
    // Dispatch for generic and specialized implementations.
    _findNeighbors(begin, end, Float(0),
                   std::integral_constant<std::size_t, Dimension>());
  }

  // CONTINUE HERE
  //! Use the adjacent cells to find the actual neighbors.
  /*! Generic implementation. */
  template<typename _Float, typename _Dimension>
  void
  _findNeighbors(std::size_t begin, std::size_t end,
                 _Float /*dummy*/, _Dimension /*dummy*/);

  // CONTINUE: Implement.
#if 0
#ifdef __SSE__
  //! Use the potential neighbors to find the actual neighbors.
  /*! Specialization for float type in 3D. */
  void
  _findNeighbors(std::size_t begin, std::size_t end,
                 float /*dummy*/,
                 std::integral_constant<std::size_t, 3> /*3D*/);
#endif
#endif

  //! Determine a fair partitioning that respects cell boundaries.
  void
  _partitionByCells(std::size_t localBegin, std::size_t localEnd,
                    std::vector<std::size_t>* delimiters) const;

  //! Determine the neighbor index of the particle itself.
  std::size_t
  _selfNeighborIndex(std::size_t i) const;

  //! Extract the positions for the potential neighbors of a cell.
  /*! \note This reads from _order.positions, so one must call
    _order.recordPositions() first. */
  void
  _extractPotentialNeighborPositions(std::size_t particle,
                                     std::vector<Point>* positions) const;

  //! Calculate the distance to determine if the potential neighbor is actually a neighbor.
  bool
  _isNeighbor(const std::size_t particle, const std::vector<Point>& positions,
              const std::size_t n) const
  {
    return squaredDistance(_order.position(particle), positions[n]) <=
           _order.squaredInteractionDistance();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  void
  printMemoryUsageTable(std::ostream& out) const;

  //! Print information about the data structure.
  void
  printInfo(std::ostream& out) const;

  //! Print performance information.
  void
  printPerformanceInfo(std::ostream& out) const
  {
    NeighborsPerformance::printPerformanceInfo(out);
    out << "\nMemory Usage:\n";
    printMemoryUsageTable(out);
  }

  //@}
};


} // namespace particle
}

#define __particle_subset_tcc__
#include "stlib/particle/subset.tcc"
#undef __particle_subset_tcc__

#endif
