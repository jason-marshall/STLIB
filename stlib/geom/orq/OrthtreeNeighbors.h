// -*- C++ -*-

/*!
  \file OrthtreeNeighbors.h
  \brief Compute neighbors with an orthtree.
*/

#if !defined(__geom_OrthtreeNeighbors_h__)
#define __geom_OrthtreeNeighbors_h__

#include "stlib/geom/orq/MortonCoordinates.h"

namespace stlib
{
namespace geom
{


// CONTINUE: Maybe I don't need _Float.
//! Compute neighbors with an orthtree.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double. Note that even if you use double-precision numbers for
  positions in the records, single-precision is probably sufficient for
  computing neighbor queries.
  \param _D The space dimension.
  \param _Record The record type, which is most likely a pointer to a class
  or an iterator into a container.
  \param _Location A functor that takes the record type as an argument
  and returns the location for the record.

  This class stores a vector of record locations and their associated Morton
  codes. For neighbor queries, it returns the indices of the neighbors.
*/
template<typename _Float,
         std::size_t _D,
         typename _Record,
         typename _Location = ads::Dereference<_Record> >
class OrthtreeNeighbors
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t D = _D;

  //
  // Types.
  //
public:

  //! The floating-point number type.
  typedef _Float Float;
  //! The record type.
  typedef _Record Record;
  //! A Cartesian point.
  typedef std::array<Float, D> Point;
  //! Bounding box.
  typedef geom::BBox<Float, D> BBox;

  //
  // Nested classes.
  //
private:

  //! Structure that provides \c position and \c code data members.
  struct PositionCode {
    std::array<_Float, _D> position;
    std::size_t code;
  };

  //
  // Data
  //
private:

  //! The functor for computing record locations.
  _Location _location;
  //! The internal vector of records hold positions and code only.
  std::vector<PositionCode> _records;
  //! The functor for computing Morton coordinates.
  MortonCoordinates<Float, D> _mortonCoordinates;

  //
  // Not implemented.
  //
private:

  // Copy constructor not implemented because it just shouldn't be used.
  OrthtreeNeighbors(const OrthtreeNeighbors&);

  // Assignment operator not implemented because it is incompatible with
  // reference data members.
  OrthtreeNeighbors&
  operator=(const OrthtreeNeighbors&);

  //--------------------------------------------------------------------------
  //! \name Constructors.
  // @{
public:

  //! Construct from the location functor.
  OrthtreeNeighbors(const _Location& location = _Location()) :
    _location(location),
    _records(),
    _mortonCoordinates()
  {
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Initialization.
  // @{
public:

  //! Initialize with the given sequence of records.
  template<typename _ForwardIterator>
  void
  initialize(_ForwardIterator begin, _ForwardIterator end)
  {
    // Set the positions for the internal records.
    _records.resize(std::distance(begin, end));
    for (std::size_t i = 0; i != _records.size(); ++i) {
      records[i].position = _location(begin++);
    }
    // Determine an appropriate domain for the tree.
    // Make the functor for computing Morton coordinates.
    _mortonCoordinates = MortonCoordinates<Float, D>(bound());
    // Set the codes for the internal records.
    for (std::size_t i = 0; i != _records.size(); ++i) {
      records[i].code = mortonCode(_mortonCoordinates(records[i].position));
    }
    // CONTINUE: Build the lookup table.
  }

private:

  //! Return an (appropriately expanded) bounding box around the points.
  BBox
  bound() const
  {
    assert(! _records.empty());
    BBox box = {_records[0].position, _records[0].position};
    for (std::size_t i = 1; i != _records.size(); ++i) {
      box.add(_records[i].position);
    }
    // Expand the box to account for round-off errors.
    box.offset(std::sqrt(std::numeric_limits<Float>::epsilon()) *
               max(domain.upper - domain.lower));
    return box;
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Neighbor queries.
  // @{
public:

  //! Find the indices of the records that are in the box.
  boxQuery(const BBox& box, std::vector<std::size_t>* neighbors) const
  {
    neighbors->clear();
    // Determine an appropriate level of refinement for searching.
    // Note that we slightly increase the length so that truncation errors
    // cannot lead to the lower and upper corners differing by more than
    // one box in any direction.
    const std::size_t level =
      _mortonCoordinates.level
      (max(box.upper - box.lower) *
       (1 + std::sqrt(std::numeric_limits<Float>::epsilon())));
    // Calculate Morton coordinates for the corners.
    std::array<std::array<std::size_t, D>, 2> corners = {
      {
        _mortonCoordinates(box.lower, level),
        _mortonCoordinates(box.upper, level)
      }
    };
    // Express the difference between the lower and upper in a bit array.
    // In 3-D, if all coordinates differ then the difference is 111.
    std::size_t difference = 0;
    for (std::size_t i = D; i != 0;) {
      --i;
      difference <<= 1;
      difference |= std::size_t(corners[1][i] != corners[0][i]);
    }
    // CONTINUE: If I shift the bits for the corners here, I could generate
    // codes later with bitwise or's.
    // CONTINUE: Handle difference == 0 as a special case.
    // CONTINUE: I could do two binary searches here to bound the range
    // covered by all boxes. Using this range could result in faster binary
    // searches later.

    // Use the difference bit array to determine which Morton boxes to
    // search.
    std::array<std::size_t, D> coords;
    for (std::size_t i = 0; i != std::size_t(1) << D; ++i) {
      // If this box has no extra bit differences.
      if (i & difference == i) {
        // Select coordinates from the two corners to get the right box.
        for (std::size_t j = 0; j != D; ++j) {
          coords[j] = corner[(i >> j) & std::size_t(1)][j];
        }
        // The first code in this box.
        const std::size_t begin = mortonCode(coords, level);
        // One past the last code in this box.
        const std::size_t end = begin + std::size_t(1) << (Levels - level);
        boxQuery(box, begin, end, neighbors);
      }
    }
  }

private:

  //! Report the records that are in the box and in the specified range of
  // codes.
  boxQuery(const BBox& box, const std::size_t begin, const std::size_t end,
           std::vector<std::size_t>* neighbors) const
  {
    // CONTINUE: Lookup table.
    // Use a binary search to find the beginning of the range.
    // CONTINUE: I need a less than comparison or a functor to make this work.
    std::vector<PositionCode>::const_iterator i =
      std::lower_bound(_records.begin(), _records.end(), begin);
    for (; i != _records.end() && i->code < end; ++i) {
      if (box.isIn(i->position)) {
        neighbors.push_back(std::distance(_records.begin(), i));
      }
    }
  }

  // @}
};

} // namespace geom
}

#define __geom_OrthtreeNeighbors_ipp__
#include "stlib/geom/orq/OrthtreeNeighbors.ipp"
#undef __geom_OrthtreeNeighbors_ipp__

#endif
