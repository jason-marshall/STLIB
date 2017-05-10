// -*- C++ -*-

/*!
  \file particle/verlet.h
  \brief Use Verlet lists to store particle neighbors.
*/

#if !defined(__particle_verlet_h__)
#define __particle_verlet_h__

#include "stlib/particle/types.h"
#include "stlib/particle/neighbors.h"
#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/numerical/partition.h"
#include "stlib/simd/functions.h"
#include "stlib/simd/shuffle.h"

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
class VerletLists : public NeighborsPerformance
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

public:

  //! The packed array of neighbors.
  container::PackedArrayOfArrays<Neighbor> neighbors;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the data structure that orders the particles.
  VerletLists(const _Order& order);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the indices and positions of the neighbors for the specified particle.
  void
  getNeighbors(std::size_t particle, std::vector<std::size_t>* indices,
               std::vector<Point>* positions) const;

  //! Return the position for the specified neighbor.
  Point
  neighborPosition(const std::size_t particle, const std::size_t index) const
  {
    return _order.neighborPosition(neighbors(particle, index));
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the total number of neighbors.
  std::size_t
  numNeighbors() const
  {
    return neighbors.size();
  }

  //! Return the number of neighbors for the specified particle.
  std::size_t
  numNeighbors(const std::size_t i) const
  {
    return neighbors.size(i);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Calculate neighbors.
  //@{
public:

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

  //! Find the neighbors for each particle.
  /*!
    \note Particles are not listed as their own neighbors.
  */
  void
  findAllNeighbors()
  {
    _findNeighbors(0, _order.cellsSize());
  }

private:

  //! Find the neighbors for each particle.
  void
  _findNeighbors(std::size_t cellsBegin, std::size_t cellsEnd);

  //! Find the neighbors for each particle.
  /*! Generic implementation. */
  template<typename _Float, typename _Dimension>
  void
  _findNeighbors(std::size_t cellsBegin, std::size_t cellsEnd,
                 container::PackedArrayOfArrays<Neighbor>* neighborsRef,
                 _Float /*dummy*/, _Dimension /*dummy*/);

  //! Find the neighbors for each particle.
  /*! Specialization for single-precision in 3D. */
  void
  _findNeighbors(std::size_t cellsBegin, std::size_t cellsEnd,
                 container::PackedArrayOfArrays<Neighbor>* packedNeighbors,
                 float /*dummy*/, std::integral_constant<std::size_t, 3> /*3D*/);

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
  printPerformanceInfo(std::ostream& out) const
  {
    NeighborsPerformance::printPerformanceInfo(out);
    out << "\nMemory Usage:\n";
    _printMemoryUsageTable(out);
  }

private:

  void
  _printMemoryUsageTable(std::ostream& out) const;

  //@}
};


} // namespace particle
}

#define __particle_verlet_tcc__
#include "stlib/particle/verlet.tcc"
#undef __particle_verlet_tcc__

#endif
