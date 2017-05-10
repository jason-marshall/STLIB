// -*- C++ -*-

#if !defined(__levelSet_flood_h__)
#define __levelSet_flood_h__

#include "stlib/levelSet/Grid.h"

#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"

#include <deque>

namespace stlib
{
namespace levelSet
{

/*! \defgroup levelSetFlood Flood Fill
  \par
  The flood filling operation determines the sign of the distance for
  unknown grid points.
*/
//@{


//! Flood fill the grid.
/*!
  Flood fill values outside of the specified range.
*/
template<typename _T, std::size_t _D>
inline
void
floodFillInterval(container::SimpleMultiArrayRef<_T, _D>* f,
                  const _T lowerThreshold, const _T upperThreshold,
                  const _T lowerFill = -std::numeric_limits<_T>::infinity(),
                  const _T upperFill = std::numeric_limits<_T>::infinity())
{
  typedef container::SimpleMultiArrayRef<_T, _D> SimpleMultiArray;
  typedef typename SimpleMultiArray::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Flood fill values where the sign of the distance is known, but the
  // value is outside of the interval.
  for (std::size_t i = 0; i != f->size(); ++i) {
    if ((*f)[i] == (*f)[i]) {
      if ((*f)[i] < lowerThreshold) {
        (*f)[i] = lowerFill;
      }
      else if ((*f)[i] > upperThreshold) {
        (*f)[i] = upperFill;
      }
    }
  }

  // Push the known values that have an unknown neighbor into a deque.
  // We use a first-in-first-out queue in order to minimize the size of
  // the queue. This is essentially a breadth first search.
  std::deque<IndexList> known;
  const IndexList closedUpper = f->extents() - std::size_t(1);
  const Iterator end = Iterator::end(f->extents());
  IndexList j;
  for (Iterator i = Iterator::begin(f->extents()); i != end; ++i) {
    // If this grid point is known.
    if ((*f)(*i) == (*f)(*i)) {
      // Determine if there is an adjacent grid point that is unknown.
      j = *i;
      for (std::size_t n = 0; n != _D; ++n) {
        if (j[n] != closedUpper[n]) {
          ++j[n];
          if ((*f)(j) != (*f)(j)) {
            known.push_back(*i);
            break;
          }
          --j[n];
        }
        // Note that the index is an unsigned integer.
        if (j[n] != 0) {
          --j[n];
          if ((*f)(j) != (*f)(j)) {
            known.push_back(*i);
            break;
          }
          ++j[n];
        }
      }
    }
  }

  // Pop values until the stack is empty.
  IndexList i;
  _T v;
  const _T mean = 0.5 * (lowerThreshold + upperThreshold);
  while (! known.empty()) {
    i = known.front();
    known.pop_front();
    v = (*f)(i) >= mean ? upperFill : lowerFill;
    // Examine each adjacent grid point.
    for (std::size_t n = 0; n != _D; ++n) {
      if (i[n] != closedUpper[n]) {
        ++i[n];
        if ((*f)(i) != (*f)(i)) {
          known.push_back(i);
          (*f)(i) = v;
        }
        --i[n];
      }
      if (i[n] != 0) {
        --i[n];
        if ((*f)(i) != (*f)(i)) {
          known.push_back(i);
          (*f)(i) = v;
        }
        ++i[n];
      }
    }
  }
}


//! Flood fill the grid.
/*!
  Flood fill values outside of the specified range.
*/
template<typename _T, std::size_t _D, std::size_t N>
inline
void
floodFillInterval(Grid<_T, _D, N>* f,
                  const _T lowerThreshold, const _T upperThreshold,
                  const _T lowerFill = -std::numeric_limits<_T>::infinity(),
                  const _T upperFill = std::numeric_limits<_T>::infinity())
{
  typedef container::SimpleMultiArrayRef<_T, _D> MultiArrayRef;
  typedef typename MultiArrayRef::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  // Dispense with the trivial case.
  if (f->empty()) {
    return;
  }

  // Use a multi-array to wrap the patches.
  const IndexList patchExtents = (*f)[0].extents();
  container::SimpleMultiArrayRef<_T, _D> patch(0, patchExtents);

  // Flood fill each of the refined grids.
  for (std::size_t i = 0; i != f->size(); ++i) {
    if ((*f)[i].isRefined()) {
      // Build the parameters.
      patch.rebuild((*f)[i].data(), patchExtents);
      floodFillInterval(&patch, lowerThreshold, upperThreshold,
                        lowerFill, upperFill);
    }
  }

  // Get rid of unnecessary refinement.
  f->coarsen();

  //
  // Flood fill the unrefined patches.
  //
  // Initialize the fill values to NaN.
  for (std::size_t i = 0; i != f->size(); ++i) {
    if (!(*f)[i].isRefined()) {
      (*f)[i].fillValue = std::numeric_limits<_T>::quiet_NaN();
    }
  }
  // Push the refined patches that have an unrefined neighbor into a deque.
  // We use a first-in-first-out queue in order to minimize the size of
  // the queue. This is essentially a breadth first search.
  std::deque<IndexList> known;
  const IndexList closedUpper = f->extents() - std::size_t(1);
  const Iterator end = Iterator::end(f->extents());
  IndexList j;
  for (Iterator i = Iterator::begin(f->extents()); i != end; ++i) {
    // If this patch is refined, the point is known.
    if ((*f)(*i).isRefined()) {
      // Determine if there is an adjacent patch that is unrefined.
      j = *i;
      for (std::size_t n = 0; n != _D; ++n) {
        if (j[n] != closedUpper[n]) {
          ++j[n];
          if (!(*f)(j).isRefined()) {
            known.push_back(*i);
            break;
          }
          --j[n];
        }
        // Note that the index is an unsigned integer.
        if (j[n] != 0) {
          --j[n];
          if (!(*f)(j).isRefined()) {
            known.push_back(*i);
            break;
          }
          ++j[n];
        }
      }
    }
  }
  // When determining the fill value for adjacent neighbors of refined
  // patches, use the value at the center of the face. The value on
  // The adjoining face may not all have the same sign.
  //++++
  //+++-
  //++--
  //----
  // Pop values until the stack is empty.
  IndexList i;
  IndexList center = ext::filled_array<IndexList>(N / 2);
  const _T mean = 0.5 * (lowerThreshold + upperThreshold);
  while (! known.empty()) {
    i = known.front();
    known.pop_front();
    // Examine each adjacent grid point.
    j = i;
    for (std::size_t n = 0; n != _D; ++n) {
      if (i[n] != closedUpper[n]) {
        ++j[n];
        if (!(*f)(j).isRefined() &&
            (*f)(j).fillValue != (*f)(j).fillValue) {
          known.push_back(j);
          if ((*f)(i).isRefined()) {
            center[n] = N - 1;
            (*f)(j).fillValue =
              (*f)(i)(center) >= mean ? upperFill : lowerFill;
            center[n] = N / 2;
          }
          else {
            (*f)(j).fillValue = (*f)(i).fillValue;
          }
        }
        --j[n];
      }
      if (i[n] != 0) {
        --j[n];
        if (!(*f)(j).isRefined() &&
            (*f)(j).fillValue != (*f)(j).fillValue) {
          known.push_back(j);
          if ((*f)(i).isRefined()) {
            center[n] = 0;
            (*f)(j).fillValue =
              (*f)(i)(center) >= mean ? upperFill : lowerFill;
            center[n] = N / 2;
          }
          else {
            (*f)(j).fillValue = (*f)(i).fillValue;
          }
        }
        ++j[n];
      }
    }
  }
}


//! Flood fill the grid.
/*!
  Flood fill with the specified value, which is \f$\infty\f$ by default.
  Unknown values and those greater than the threshold will be set to
  +-<em>fillValue</em> if the sign of the distance can be determined.
*/
template<typename _T, std::size_t _D>
inline
void
floodFill(container::SimpleMultiArrayRef<_T, _D>* f,
          const _T threshold = std::numeric_limits<_T>::max(),
          const _T fillValue = std::numeric_limits<_T>::infinity())
{
  floodFillInterval(f, -threshold, threshold, -fillValue, fillValue);
}


//! Flood fill the grid.
/*!
  Flood fill with the specified value, which is \f$\infty\f$ by default.
  Unknown values and those greater than the threshold will be set to
  +-<em>fillValue</em> if the sign of the distance can be determined.
*/
template<typename _T, std::size_t _D, std::size_t N>
inline
void
floodFill(Grid<_T, _D, N>* f,
          const _T threshold = std::numeric_limits<_T>::max(),
          const _T fillValue = std::numeric_limits<_T>::infinity())
{
  floodFillInterval(f, -threshold, threshold, -fillValue, fillValue);
}


//@}


} // namespace levelSet
}

#endif
