// -*- C++ -*-

/**
  \file
  \brief Contains the selectionSort() functions.
*/

#if !defined(__ads_selectionSort_h__)
#define __ads_selectionSort_h__

#include <utility>

namespace stlib
{
namespace ads
{

//-----------------------------------------------------------------------------
/** \defgroup algorithm_selectionSort Algorithm: Selection Sort */
// @{


/// In-place selection sort.
/**
   \param first The beginning of the input sequence.
   \param last The beginning of the input sequence.
*/
template<typename _RandomAccessIterator>
void
selectionSort(_RandomAccessIterator first, _RandomAccessIterator last);


/// In-place selection sort.
/**
   \param first The beginning of the input sequence.
   \param last The beginning of the input sequence.
   \param compare Comparison functor for sequence values.
*/
template<typename _RandomAccessIterator, typename _Compare>
void
selectionSort(_RandomAccessIterator first, _RandomAccessIterator last,
              _Compare compare);


/// Use a selection sort to generate a sorted sequence. 
/**
   \param first The beginning of the input sequence.
   \param last The beginning of the input sequence.
   \param output An output iterator.

   The output range may not overlap the input range. The input sequence will
   be consumed. After algorithm completion, the input values are not defined. 
*/
template<typename _RandomAccessIterator, typename _OutputIterator>
void
selectionSortSeparateOutput(_RandomAccessIterator first,
                            _RandomAccessIterator last,
                            _OutputIterator output);


/// Use a selection sort to generate a sorted sequence. 
/**
   \param first The beginning of the input sequence.
   \param last The beginning of the input sequence.
   \param output An output iterator.
   \param compare Comparison functor for sequence values.

   The output range may not overlap the input range. The input sequence will
   be consumed. After algorithm completion, the input values are not defined. 
*/
template<typename _RandomAccessIterator, typename _OutputIterator,
         typename _Compare>
void
selectionSortSeparateOutput(_RandomAccessIterator first,
                            _RandomAccessIterator last,
                            _OutputIterator output,
                            _Compare compare);


// @}

} // namespace ads
} // namespace stlib

#define __ads_selectionSort_ipp__
#include "stlib/ads/algorithm/selectionSort.ipp"
#undef __ads_selectionSort_ipp__

#endif
