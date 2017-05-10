// -*- C++ -*-

#if !defined(__levelSet_count_h__)
#define __levelSet_count_h__

#include "stlib/container/SimpleMultiArray.h"

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetCount Count Known Values
 These functions count the number of known or unknown values in a grid. */
//@{


//! Return true if the grid has any known values.
/*!
  NaN represents an unknown value.
  Return as soon as a known value is found.
*/
template<typename _InputIterator>
inline
bool
hasKnown(_InputIterator begin, _InputIterator end)
{
  typename std::iterator_traits<_InputIterator>::value_type x;
  while (begin != end) {
    x = *begin++;
    if (x == x) {
      return true;
    }
  }
  return false;
}


//! Return true if the grid has any unknown values.
/*!
  NaN represents an unknown value.
  Return as soon as a known value is found.
*/
template<typename _InputIterator>
inline
bool
hasUnknown(_InputIterator begin, _InputIterator end)
{
  typename std::iterator_traits<_InputIterator>::value_type x;
  while (begin != end) {
    x = *begin++;
    if (x != x) {
      return true;
    }
  }
  return false;
}


//! Return true if the grid has any non-positive values.
/*!
  Return as soon as a non-positive value is found.
*/
template<typename _InputIterator>
inline
bool
hasNonPositive(_InputIterator begin, _InputIterator end)
{
  while (begin != end) {
    if (*begin++ <= 0) {
      return true;
    }
  }
  return false;
}


// CONTINUE: Get rid of these as they conflict with the Grid interface.
#if 0
//! Return true if the grid has any known values.
template<typename _T, std::size_t _D>
inline
bool
hasKnown(const container::SimpleMultiArrayConstRef<_T, _D>& grid)
{
  return hasKnown(grid.begin(), grid.end());
}


//! Return true if the grid has any unknown values.
template<typename _T, std::size_t _D>
inline
bool
hasUnknown(const container::SimpleMultiArrayConstRef<_T, _D>& grid)
{
  return hasUnknown(grid.begin(), grid.end());
}
#endif

//! Return true if the grid values are all the same.
template<typename _InputIterator>
inline
bool
allSame(_InputIterator begin, _InputIterator end)
{
  typename std::iterator_traits<_InputIterator>::value_type first = *begin++;
  // If the first grid point is a NaN.
  if (first != first) {
    // Check if all of the grids points are NaN.
    return ! hasKnown(begin, end);
  }
  // Otherwise check if all of the grid points are equal to the first.
  while (begin != end) {
    if (*begin++ != first) {
      return false;
    }
  }
  return true;
}


//! Return true if the grid values are all non-positive.
/*! \note Return false if any grid values are NaN. */
template<typename _InputIterator>
inline
bool
allNonPositive(_InputIterator begin, _InputIterator end)
{
  while (begin != end) {
    if (!(*begin++ <= 0)) {
      return false;
    }
  }
  return true;
}

//! Return true if the grid values have mixed signs.
/*! \note NaN's are ignored. */
template<typename _InputIterator>
inline
bool
mixedSigns(_InputIterator begin, _InputIterator end)
{
  typename std::iterator_traits<_InputIterator>::value_type x;
  bool pos = false;
  bool neg = false;
  while (begin != end) {
    x = *begin++;
    neg |= x < 0;
    pos |= x > 0;
  }
  return pos && neg;
}

//! Return true if the grid point has an unknown neighbor.
/*! Pass the index by value because we will use it for scratch calculations. */
template<typename _T, std::size_t _D>
inline
bool
hasUnknownAdjacentNeighbor
(const container::SimpleMultiArrayConstRef<_T, _D>& grid,
 typename container::SimpleMultiArrayConstRef<_T, _D>::IndexList i)
{
  // Examine each adjacent grid point.
  for (std::size_t n = 0; n != _D; ++n) {
    if (i[n] + 1 != grid.extents()[n]) {
      ++i[n];
      if (grid(i) != grid(i)) {
        return true;
      }
      --i[n];
    }
    if (i[n] != 0) {
      --i[n];
      if (grid(i) != grid(i)) {
        return true;
      }
      ++i[n];
    }
  }
  return false;
}


//! Return true if the grid point has a neighbor with a value below the threshold.
/*! Pass the index by value because we will use it for scratch calculations. */
template<typename _T, std::size_t _D>
inline
bool
hasNeighborBelowThreshold
(const container::SimpleMultiArrayConstRef<_T, _D>& grid,
 typename container::SimpleMultiArrayConstRef<_T, _D>::IndexList i,
 const _T threshold)
{
  // Examine each adjacent grid point.
  for (std::size_t n = 0; n != _D; ++n) {
    if (i[n] + 1 != grid.extents()[n]) {
      ++i[n];
      if (grid(i) < threshold) {
        return true;
      }
      --i[n];
    }
    if (i[n] != 0) {
      --i[n];
      if (grid(i) < threshold) {
        return true;
      }
      ++i[n];
    }
  }
  return false;
}


//! Count the number of known values in the sequence.
/*! NaN represents an unknown value. */
template<typename _ForwardIterator>
inline
std::size_t
countKnown(_ForwardIterator begin, _ForwardIterator end)
{
  std::size_t count = 0;
  while (begin != end) {
    count += *begin == *begin;
    ++begin;
  }
  return count;
}


//! Count the number of unknown values in the sequence.
/*! NaN represents an unknown value. */
template<typename _ForwardIterator>
inline
std::size_t
countUnknown(_ForwardIterator begin, _ForwardIterator end)
{
  return std::distance(begin, end) - countKnown(begin, end);
}


// CONTINUE: Get rid of these as they conflict with the Grid interface.
#if 0
//! Count the number of known points in the grid.
/*! NaN represents an unknown value. */
template<typename _T, std::size_t _D>
inline
std::size_t
countKnown(const container::SimpleMultiArrayConstRef<_T, _D>& grid)
{
  return countKnown(grid.begin(), grid.end());
}


//! Count the number of unknown points in the grid.
/*! NaN represents an unknown value. */
template<typename _T, std::size_t _D>
inline
std::size_t
countUnknown(const container::SimpleMultiArrayConstRef<_T, _D>& grid)
{
  return countUnknown(grid.begin(), grid.end());
}
#endif

//! Print information about the grid.
template<typename _InputIterator>
inline
void
printLevelSetInfo(_InputIterator begin, _InputIterator end, std::ostream& out)
{
  typedef typename std::iterator_traits<_InputIterator>::value_type Value;
  const Value Inf = std::numeric_limits<Value>::infinity();

  std::size_t size = 0;
  std::size_t nonNegative = 0;
  std::size_t negative = 0;
  std::size_t unknown = 0;
  std::size_t positiveFar = 0;
  std::size_t negativeFar = 0;
  Value x;
  while (begin != end) {
    x = *begin++;
    ++size;
    if (x >= 0) {
      ++nonNegative;
    }
    else if (x < 0) {
      ++negative;
    }
    else {
      ++unknown;
    }
    if (x == Inf) {
      ++positiveFar;
    }
    else if (x == -Inf) {
      ++negativeFar;
    }
  }
  out << "Number of grid points = " << size << '\n'
      << "known/unknown = " << size - unknown << " / " << unknown << '\n'
      << "non-negative/negative = " << nonNegative << " / " << negative
      << '\n'
      << "positive far/negative far = " << positiveFar << " / " << negativeFar
      << '\n';
}


// CONTINUE: Get rid of these as they conflict with the Grid interface.
#if 0
//! Print information about the grid.
template<typename _T, std::size_t _D>
inline
void
printLevelSetInfo(const container::SimpleMultiArrayConstRef<_T, _D>& grid,
                  std::ostream& out)
{
  printLevelSetInfo(grid.begin(), grid.end(), out);
}
#endif


//@}


} // namespace levelSet
}

#endif
