// -*- C++ -*-

/*!
  \file particle/unionMask.h
  \brief Use the union of cell neighbors and bit masks to store particle neighbors.
*/

#if !defined(__particle_unionMask_h__)
#define __particle_unionMask_h__

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

//! Use the union of cell neighbors and bit masks to store particle neighbors.
/*!
  \param _Order The class for ordering the particles.
*/
template<typename _Order>
class UnionMask : public NeighborsPerformance
{
  //
  // Constants.
  //
private:

  //! The Dimension of the space.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Order::Dimension;
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

  //
  // Member data.
  //
public:

  //! For each cell, the union of interacting neighbors.
  /*! Note that the lists are not padded to be multiples of the mask size. */
  container::PackedArrayOfArrays<Neighbor> unionOfNeighbors;
  //! For each particle, the bit masks that record actual neighbors.
  container::PackedArrayOfArrays<Mask> neighborMasks;

private:

  //! The data structure that orders the particles.
  const _Order& _order;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the data structure that orders the particles.
  UnionMask(const _Order& order);

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

  //! Return true if the potential neighbor is actually a neighbor.
  /*! Use the bit-masks that record the actual neighbors. */
  bool
  isNeighbor(std::size_t particle, std::size_t index) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Calculate neighbors.
  //@{
public:

  //! Find the neighbors for each particle.
  /*!
    \note Particles are not listed as their own neighbors.
  */
  void
  findAllNeighbors()
  {
    _findNeighbors(0, _order.cellsSize());
  }

  //! Find the neighbors for each local particle.
  /*!
    The shadow particles will be assigned empty lists of neighbors.
    \note Particles are not listed as their own neighbors.
  */
  void
  findLocalNeighbors()
  {
    _findNeighbors(_order.localCellsBegin(), _order.localCellsEnd());
  }

  //! Get the positions of the union of neighbors.
  /*! The positions are padded with NaN's to match the length of the bit
    masks. */
  void
  positionsInUnion(std::size_t cell, std::vector<Point>* positions) const;

  //! Get the indices and the positions of the union of neighbors.
  /*! The indices are padded with 0 and the positions are padded with NaN's
    to match the length of the bit masks. */
  void
  positionsInUnion(std::size_t cell, std::vector<std::size_t>* indices,
                   std::vector<Point>* positions) const;

private:

  //! Pad the positions with NaN's to match the length of the bit masks.
  void
  _pad(std::size_t cell, std::vector<Point>* positions) const;

  //! Pad the indices with zeros and the positions with NaN's.
  void
  _pad(std::size_t cell, std::vector<std::size_t>* indices,
       std::vector<Point>* positions)
  const;

  //! Get the positions of the union of neighbors.
  /*! The positions are padded with NaN's to match the length of the bit
    masks. */
  void
  _positionsInUnion(const std::vector<Point>& cachedPositions,
                    std::size_t cell, std::vector<Point>* positions) const;

  //! Calculate the union of neighbors.
  void
  _getUnionOfNeighbors(const std::vector<Point>& cachedPositions,
                       std::size_t cell, std::vector<Neighbor>* neighbors)
  const;

  //! Find the neighbors for the specified range of cells.
  void
  _findNeighbors(std::size_t cellsBegin, std::size_t cellsEnd);

  //! Find the neighbors for the specified range of cells.
  /*! Generic implementation. */
  template<typename _Float, typename _Dimension>
  void
  _findNeighbors(const std::vector<Point>& cachedPositions,
                 std::size_t cellsBegin, std::size_t cellsEnd,
                 container::PackedArrayOfArrays<Neighbor>* unionOfNeighbors_,
                 container::PackedArrayOfArrays<Mask>* neighborMasks_,
                 _Float /*dummy*/, _Dimension /*dummy*/);

// CONTINUE: Implement this.
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

  //! Determine a fair partitioning of the cells.
  void
  _partitionCells(std::size_t localBegin, std::size_t localEnd,
                  std::vector<std::size_t>* delimiters) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  //! Print information about the memory usage.
  void
  printMemoryUsageTable(std::ostream& out) const;

  //! Print information about the density of neighbors in the neighbor masks.
  void
  printNeighborDensity(std::ostream& out) const;

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
    out << '\n';
    printNeighborDensity(out);
  }

  //@}
};


} // namespace particle
}

#define __particle_unionMask_tcc__
#include "stlib/particle/unionMask.tcc"
#undef __particle_unionMask_tcc__

#endif
