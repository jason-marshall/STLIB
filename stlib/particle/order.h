// -*- C++ -*-

/*!
  \file particle/order.h
  \brief Use Morton codes to order particles.
*/

#if !defined(__particle_order_h__)
#define __particle_order_h__

#include "stlib/particle/codes.h"
#include "stlib/particle/lookup.h"

#include "stlib/container/PackedArrayOfArrays.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/numerical/constants/Exponentiation.h"
#include "stlib/numerical/partition.h"
#include "stlib/performance/SimpleTimer.h"

#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace particle
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

//! The representation of a neighbor.
/*! For plain domains, this is just an index. For periodic domains, it
  is a pair of the particle index and an index into a periodic offsets
  array. */
template<bool _Periodic>
struct Neighbor;

//! The representation of a neighbor for plain domains.
template<>
struct Neighbor<false> {
  //! The particle index.
  std::size_t particle;

  //! Equality comparison.
  bool
  operator==(const Neighbor& other) const
  {
    return particle == other.particle;
  }

  //! Increment operator. Move to the next particle in a cell.
  void
  operator++()
  {
    ++particle;
  }
};

//! The representation of a neighbor for periodic domains.
template<>
struct Neighbor<true> {
  //! The particle index.
  std::size_t particle;
  //! The periodic offset index.
  std::size_t offset;

  //! Equality comparison.
  bool
  operator==(const Neighbor& other) const
  {
    return particle == other.particle && offset == other.offset;
  }

  //! Increment operator. Move to the next particle in a cell.
  void
  operator++()
  {
    ++particle;
  }
};

//! Write the neighbor.
inline
std::ostream&
operator<<(std::ostream& out, const Neighbor<false>& neighbor)
{
  return out << neighbor.particle;
}

//! Write the neighbor.
inline
std::ostream&
operator<<(std::ostream& out, const Neighbor<true>& neighbor)
{
  return out << neighbor.particle << ' ' << neighbor.offset;
}


//! The representation of a neighboring cell.
/*! For plain domains, this is just a cell index. For periodic domains, it
  is a pair of the cell index and an index into a periodic offsets
  array. */
template<bool _Periodic_>
struct NeighborCell;

//! The representation of a neighbor for plain domains.
template<>
struct NeighborCell<false> {
  //! The cell index.
  std::size_t cell;

  //! Equality operator.
  bool
  operator==(const NeighborCell& other) const
  {
    return cell == other.cell;
  }

  //! Less than operator.
  bool
  operator<(const NeighborCell& other) const
  {
    return cell < other.cell;
  }
};

//! The representation of a neighbor for periodic domains.
template<>
struct NeighborCell<true> {
  //! The cell index.
  std::size_t cell;
  //! The periodic offset index.
  std::size_t offset;

  //! Equality operator.
  bool
  operator==(const NeighborCell& other) const
  {
    return cell == other.cell && offset == other.offset;
  }

  //! Less than operator.
  bool
  operator<(const NeighborCell& other) const
  {
    return cell < other.cell;
  }
};


//! Use Morton codes to order particles.
/*!
  \param _Traits The traits class. Use PlainTraits, PeriodicTraits, or a
  class with equivalent functionality.
*/
template<typename _Traits>
class MortonOrder
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
  //! A discrete coordinate.
  typedef IntegerTypes::DiscreteCoordinate DiscreteCoordinate;
  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;

protected:

  //! The class for computing spatial indices.
  typedef Morton<Float, Dimension, Periodic> SpatialIndex;
  //! A discrete point with integer coordinates.
  typedef typename TemplatedTypes<Float, Dimension>::DiscretePoint
  DiscretePoint;

private:

  //! A range of multi-indices.
  typedef container::SimpleMultiIndexRange<Dimension> IndexRange;

  //
  // Member data.
  //
public:

  //! The vector of particles.
  /*! While you are free to manipulate particles through this data member,
    do not alter the sequence, i.e. resize it. */
  std::vector<Particle> particles;
  //! The adjacent cells for each cell.
  container::PackedArrayOfArrays<NeighborCell<Periodic> > adjacentCells;
  //! The number of adjacent neighbors for each cell.
  /*! This is the product of the number of particles in the cell and the
   number of particles in the adjacent cells. This quantity is used in
   partitioning the cells. */
  std::vector<std::size_t> numAdjacentNeighbors;

protected:

  //! The data structure for computing spatial indices.
  SpatialIndex morton;
  //! The number of subcell levels.
  std::size_t _subcellLevels;
  //! Used for the sub-cell ordering of the particles.
  SpatialIndex _subcellMorton;

  //! The particle index delimiters for the cells.
  /*! While the vector of codes provides an implicit representation of the
    (non-empty) cells. This is an explicit representation that makes it easier
    to apply cell-based operations. The size of this vector is one more than
    the number of non-empty cells. */
  std::vector<std::size_t> _cellDelimiters;
  //! The codes for each cell.
  /*! Note that the length is one more than the number of cells due to
    a guard element, which has a value that is greater than any valid code. */
  std::vector<Code> _cellCodes;
  //! Lookup table for accelerating searches in the cell codes.
  LookupTable _lookupTable;

  //! The functor for extracting the position from a particle.
  typename _Traits::GetPosition _getPosition;

private:

  //! The functor for setting the position in a particle.
  typename _Traits::SetPosition _setPosition;

  //! Offsets for moving particles in neighboring cells in periodic domains.
  /*! This is a linear representation of an N-D multi-array, whose extent in
    each direction is 3. */
  std::array<Point, numerical::Exponentiation
  <std::size_t, 3, Dimension>::Result> _periodicOffsets;

  //! The interaction distance.
  Float _interactionDistance;
  //! The squared interaction distance.
  Float _squaredInteractionDistance;
  //! The padding for the interaction distance.
  /*! This is the amount that a particle can move before recomputing the
    order is necessary. */
  Float _padding;
  //! The starting positions of the particles when the order is set.
  /*! These are not recorded if the padding is zero. */
  std::vector<Point> _startingPositions;

  //! The number of times the particles have been reordered.
  std::size_t _reorderCount;
  //! The number of times the data structure has been repaired.
  std::size_t _repairCount;

  //! A timer for measuring time spent in various functions.
  /*! It's mutable so that it can be used in const member functions. */
  mutable performance::SimpleTimer _timer;
  //! The time spent checking if the order is valid.
  mutable double _timeIsOrderValid;
  //! The time spent ordering the particles.
  double _timeOrder;
  //! The time spent recording starting positions.
  double _timeRecordStartingPositions;
  //! The time spent building the lookup table.
  double _timeBuildLookupTable;
  //! The time spent calculating adjacent cells.
  double _timeAdjacentCells;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the domain, the interaction distance, and the padding.
  /*! If the value for the padding is omitted, a suitable value will be
    chosen. */
  MortonOrder(const geom::BBox<Float, Dimension>& domain,
              Float interactionDistance,
              Float padding = std::numeric_limits<Float>::quiet_NaN());

  //! Default constructor invalidates the data members.
  MortonOrder();

  //! Initialize from the domain, the interaction distance, and the padding.
  /*! If the value for the padding is omitted, a suitable value will be
    chosen. */
  void
  initialize(const geom::BBox<Float, Dimension>& domain,
             Float interactionDistance,
             Float padding = std::numeric_limits<Float>::quiet_NaN());

private:

  //! Calculate the periodic offsets for the neighboring cells.
  void
  _calculatePeriodicOffsets();

  //@}
  //--------------------------------------------------------------------------
  //! \name Order particles.
  //@{
public:

  //! Set the particles.
  /*!
    \param begin The beginning of a sequence of particles.
    \param end One past the end of a sequence of particles.

     Calculate the codes. Determine the order of the particles. Record
     the starting positions.
  */
  template<typename _InputIterator>
  void
  setParticles(_InputIterator begin, _InputIterator end);

  //! Repair the data structure if necessary.
  /*! Return true if it is repaired. */
  bool
  repair();

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the position for the i_th particle.
  /*! Note that the return type must be \c Point and not a constant reference
   because we don't know if the position functor return a constant reference.
   As this function should be inlined, there should be no performance penalty.

   \note We need this function so that the neighbor classes can access
   positions.
  */
  Point
  position(const std::size_t i) const
  {
    return _getPosition(particles[i]);
  }

  //! Cache the positions of the particles.
  void
  getPositions(std::vector<Point>* cachedPositions) const
  {
    cachedPositions->resize(particles.size());
    #pragma omp parallel for default(none) shared(cachedPositions)
    for (std::ptrdiff_t i = 0; i < std::ptrdiff_t(particles.size()); ++i) {
      (*cachedPositions)[i] = position(i);
    }
  }

  //! Return the interaction distance.
  Float
  interactionDistance() const
  {
    return _interactionDistance;
  }

  //! Return the squared interaction distance.
  Float
  squaredInteractionDistance() const
  {
    return _squaredInteractionDistance;
  }

  //! Return the padding.
  Float
  padding() const
  {
    return _padding;
  }

  //! Return the number of non-empty cells.
  std::size_t
  cellsSize() const
  {
    return _cellDelimiters.size() - 1;
  }

  //! The index of the first particle in the nth non-empty cell.
  std::size_t
  cellBegin(const std::size_t n) const
  {
#ifdef STLIB_DEBUG
    assert(n < _cellDelimiters.size());
#endif
    return _cellDelimiters[n];
  }

  //! The index of one past the last particle in the nth non-empty cell.
  std::size_t
  cellEnd(const std::size_t n) const
  {
#ifdef STLIB_DEBUG
    assert(n + 1 < _cellDelimiters.size());
#endif
    return _cellDelimiters[n + 1];
  }

  //! Return the first particle in a neighboring cell.
  Neighbor<false>
  cellBegin(const NeighborCell<false>& neighbor) const
  {
    return Neighbor<false>{cellBegin(neighbor.cell)};
  }

  //! Return the one past the last particle in a neighboring cell.
  Neighbor<false>
  cellEnd(const NeighborCell<false>& neighbor) const
  {
    return Neighbor<false>{cellEnd(neighbor.cell)};
  }

  //! Return the first particle in a neighboring cell.
  Neighbor<true>
  cellBegin(const NeighborCell<true>& neighbor) const
  {
    return Neighbor<true>{cellBegin(neighbor.cell), neighbor.offset};
  }

  //! Return the one past the last particle in a neighboring cell.
  Neighbor<true>
  cellEnd(const NeighborCell<true>& neighbor) const
  {
    return Neighbor<true>{cellEnd(neighbor.cell), neighbor.offset};
  }

  //! The first index of the local particles.
  /*! This is for compatibility with the MortonOrderMpi. */
  std::size_t
  localParticlesBegin() const
  {
    return 0;
  }

  //! One past the last index of the local particles.
  /*! This is for compatibility with the MortonOrderMpi. */
  std::size_t
  localParticlesEnd() const
  {
    return particles.size();
  }

  //! The number of non-empty, local cells.
  std::size_t
  localCellsSize() const
  {
    return cellsSize();
  }

  //! The index of the first local cell.
  std::size_t
  localCellsBegin() const
  {
    return 0;
  }

  //! One past the index of the last local cell.
  std::size_t
  localCellsEnd() const
  {
    return cellsSize();
  }

  //! Return the lower corner of the Cartesian domain.
  const Point&
  lowerCorner() const
  {
    return morton.lowerCorner();
  }

  //! The lengths of the Cartesian domain.
  const Point&
  lengths() const
  {
    return morton.lengths();
  }

  //! Return the index of the first cell with a code that is not less than the argument.
  std::size_t
  index(const Code code) const;

  //! Get the indices and positions of the particles in the adjacent cells.
  void
  positionsInAdjacent(std::size_t cell, std::vector<Point>* positions) const;

  //! Get the indices and positions of the particles in the adjacent cells.
  void
  positionsInAdjacent(const std::vector<Point>& cachedPositions,
                      std::size_t cell, std::vector<Point>* positions) const;

  //! Get the indices and positions of the particles in the adjacent cells.
  void
  positionsInAdjacent(std::size_t cell, std::vector<std::size_t>* indices,
                      std::vector<Point>* positions) const;

  //! Get the neighbor representation and positions of the particles in the adjacent cells.
  void
  positionsInAdjacent(std::size_t cell,
                      std::vector<Neighbor<Periodic> >* neighbors,
                      std::vector<Point>* positions) const;

  //! Return the number of particles that precede the center cell.
  std::size_t
  centerCellOffset(const std::size_t cell) const
  {
    std::size_t offset = 0;
    for (std::size_t i = 0; i != adjacentCells.size(cell); ++i) {
      const std::size_t c = adjacentCells(cell, i).cell;
      if (c == cell) {
        return offset;
      }
      offset += cellEnd(c) - cellBegin(c);
    }
    assert(false);
    return std::numeric_limits<std::size_t>::max();
  }

  //! Count the particles in the adjacent cells.
  std::size_t
  countAdjacentParticles(std::size_t cell) const;

  //! Return the position for the specified neighbor in a plain domain.
  Point
  neighborPosition(const Neighbor<false>& neighbor) const
  {
    return _getPosition(particles[neighbor.particle]);
  }

  //! Return the position for the specified neighbor in a periodic domain.
  Point
  neighborPosition(const Neighbor<true>& neighbor) const
  {
    return _getPosition(particles[neighbor.particle]) +
           _periodicOffsets[neighbor.offset];
  }

  //! Return the position for the specified neighbor in a plain domain.
  Point
  neighborPosition(const std::vector<Point>& cachedPositions,
                   const Neighbor<false>& neighbor) const
  {
    return cachedPositions[neighbor.particle];
  }

  //! Return the position for the specified neighbor in a periodic domain.
  Point
  neighborPosition(const std::vector<Point>& cachedPositions,
                   const Neighbor<true>& neighbor) const
  {
    return cachedPositions[neighbor.particle] +
           _periodicOffsets[neighbor.offset];
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
  printPerformanceInfo(std::ostream& out) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Protected functions.
  //@{
protected:

  //
  // Constructors etc.
  //

  //! Calculate an appropriate padding if none was specified.
  static
  Float
  _appropriatePadding(Float interactionDistance, Float padding);

  //
  // Order particles.
  //

  //! Return true if no particle has moved farther than the allowed padding.
  bool
  isOrderValid() const
  {
    return isOrderValid(0, particles.size());
  }

  //! Return true if no particle in the specified range has moved farther than the allowed padding.
  bool
  isOrderValid(std::size_t begin, std::size_t end) const;

  //! Reorder the particles.
  /*!
    Recalculate the codes. Reorder the particles. Record the new starting
    positions.
  */
  void
  reorder();

  //! Calculate codes from the positions and order the particles.
  void
  order();

  //! Record the starting positions for the ordered particles.
  void
  recordStartingPositions();

  //! Build the lookup table.
  void
  buildLookupTable();

  //! Calculate the adjacent cells for each cell.
  void
  calculateAdjacentCells();

  //! Insert the cells specified with their codes and sizes.
  /*! \return The local range of cells. */
  std::pair<std::size_t, std::size_t>
  insertCells(const std::vector<std::pair<Code, std::size_t> >& codesSizes,
              Code localCodesBegin, Code localCodesEnd);

  //! Erase the shadow particles, given the range of local ones.
  void
  eraseShadow(std::size_t localCellsBegin, std::size_t localCellsEnd);

  //! Normalize the particle positions.
  /*! For plain domains, do nothing. For periodic domains, move to lie within
   the domain. */
  void
  normalizePositions()
  {
    _normalizePositions(std::integral_constant<bool, Periodic>());
  }

  //@}
protected:

  // The _index* functions are protected so that I can more easily test their
  // performance. They are not called in derived classes.

  //! Return the index of the first cell with a code that is not less than the argument.
  /*! Use direct lookup in the case that the shift is zero. */
  std::size_t
  _indexDirect(Code code) const;

  //! Return the index of the first cell with a code that is not less than the argument.
  /*! Use lookup followed by a forward search in the case that the
    shift is positive and small. */
  std::size_t
  _indexForward(Code code) const;

  //! Return the index of the first cell with a code that is not less than the argument.
  /*! Use lookup followed by a binary search in the case that the
    shift is large. */
  std::size_t
  _indexBinary(Code code) const;

private:

  //! Determine the index range for the adjacent cells.
  IndexRange
  _adjacentRange(std::size_t cell, std::false_type /*Periodic*/) const;

  //! Determine the index range for the adjacent cells.
  /*! \note Return a range that is offset by 1 to avoid negative indices. */
  IndexRange
  _adjacentRange(std::size_t cell, std::true_type /*Periodic*/) const;

  //! Find the adjacent cells.
  void
  _findAdjacentCells(std::size_t cell,
                     std::vector<NeighborCell<false> >* adjacent) const;

  //! Find the adjacent cells.
  void
  _findAdjacentCells(std::size_t cell,
                     std::vector<NeighborCell<true> >* adjacent) const;

  //! Do nothing.
  void
  _normalizePositions(std::false_type /*Periodic*/)
  {
  }

  //! Move particles to lie within the domain.
  void
  _normalizePositions(std::true_type /*Periodic*/);
};


} // namespace particle
}

#define __particle_order_tcc__
#include "stlib/particle/order.tcc"
#undef __particle_order_tcc__

#endif
