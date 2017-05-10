// -*- C++ -*-

/*!
  \file particle/verletPotential.h
  \brief Use Verlet lists to store particle neighbors.
*/

#if !defined(__particle_verletPotential_h__)
#define __particle_verletPotential_h__

#include "stlib/particle/types.h"
#include "stlib/particle/neighbors.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/numerical/partition.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace particle
{

//! Use Verlet lists to store particle neighbors.
/*!
  \param _Order The class for ordering the particles.
*/
template<typename _Order>
class VerletListsPotential : public NeighborsPerformance
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
  /*! For plain domains, this is just an index. For periodic domains, it
    is a pair of the particle index and an index into the periodic offsets
    array. */
  typedef particle::Neighbor<Periodic> Neighbor;

  //
  // Member data.
  //
private:

  //! The data structure that orders the particles.
  const _Order& _order;
  //! The beginning of the cells for which we calculate neighbors.
  std::size_t _cellsBegin;
  //! The end of the cells for which we calculate neighbors.
  std::size_t _cellsEnd;
  //! The packed array of potential neighbors.
  /*! "Potential" means within the interaction distance plus the padding.
    For plain domains, we store the indices of the neighbors. For periodic
    domains, we store a pair of the neighbor index and an index into
    the periodic offsets array. This is because the neighbor may be a
    virtual periodic extension of an explicitly represented particle.
    In this case, the position must be offset before being used. */
  container::PackedArrayOfArrays<Neighbor> _potentialNeighbors;
  //! The packed array of neighbors.
  container::PackedArrayOfArrays<Neighbor> _neighbors;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the data structure that orders the particles.
  VerletListsPotential(const _Order& order);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the total number of neighbors.
  std::size_t
  numNeighbors() const
  {
    return _neighbors.size();
  }

  //! Return the number of neighbors for the specified particle.
  std::size_t
  numNeighbors(const std::size_t i) const
  {
    return _neighbors.size(i);
  }

  //! Return the total number of potential neighbors.
  std::size_t
  numPotentialNeighbors() const
  {
    return _potentialNeighbors.size();
  }

  //! Return the number of potential neighbors for the specified particle.
  std::size_t
  numPotentialNeighbors(const std::size_t i) const
  {
    return _potentialNeighbors.size(i);
  }

private:

  //! Return true if the particles are neighbors.
  /*! \pre <code>particle < particles.size()</code>
   \pre <code>index < numPotentialNeighbors(particle)</code> */
  bool
  _isNeighbor(std::size_t particle, std::size_t index) const;

public:

  //! Return the index of the specified neighbor.
  std::size_t
  neighborIndex(const std::size_t particle, const std::size_t index) const
  {
    return _neighbors(particle, index).particle;
  }

public:

  //! Return the position for the specified neighbor.
  Point
  neighborPosition(const std::size_t particle, const std::size_t index) const
  {
    return _order.neighborPosition(_neighbors(particle, index));
  }

private:

  //! Return the position for the specified potential neighbor.
  Point
  potentialNeighborPosition(const std::size_t particle,
                            const std::size_t index) const
  {
    return _order.neighborPosition(_potentialNeighbors(particle, index));
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Calculate neighbors.
  //@{
public:

  //! Find the potential neighbors for each particle.
  /*!
    \note Particles are not listed as their own neighbors.
  */
  void
  findPotentialNeighbors()
  {
    findPotentialNeighbors(0, _order.cellsSize());
  }

  //! Find the potential neighbors for each particle.
  /*!
    \param localCellsBegin The index of the first local cell.
    \param localCellsEnd One past the index of the last local cell.

    \note Particles are not listed as their own neighbors.
  */
  void
  findPotentialNeighbors(const std::size_t localCellsBegin,
                         const std::size_t localCellsEnd)
  {
    _timer.start();
    assert(localCellsBegin <= _order.cellsSize() &&
           localCellsBegin <= localCellsEnd &&
           localCellsEnd <= _order.cellsSize());
    ++_potentialNeighborsCount;
    _cellsBegin = localCellsBegin;
    _cellsEnd = localCellsEnd;
    _findPotentialNeighbors(localCellsBegin, localCellsEnd);
    _timer.stop();
    _timePotentialNeighbors += _timer.elapsed();
  }

private:

  //! Find the neighbors for each particle.
  void
  _findPotentialNeighbors(std::size_t localCellsBegin,
                          std::size_t localCellsEnd);

public:

  //! Use the potential neighbors to find the actual neighbors.
  void
  findNeighbors();

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

#define __particle_verletPotential_tcc__
#include "stlib/particle/verletPotential.tcc"
#undef __particle_verletPotential_tcc__

#endif
