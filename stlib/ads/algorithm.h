// -*- C++ -*-

/*!
  \file ads/algorithm.h
  \brief Includes the algorithm files.
*/

/*!
  \page ads_algorithm Algorithm Package

  The algorithm package defines
  \ref algorithm_min_max "min and max functions"
  for three, four or
  five arguments.  They have the same kind of interface as std::min(),
  which returns the minimum of two arguments,  For example:
  - ads::min(const T& a, const T& b, const T& c)
  - ads::min(const T& a, const T& b, const T& c, Compare comp)

  It also defines the
  \ref algorithm_sign "ads::sign() function".

  There are functions for \ref algorithm_sort "sorting and ranking".

  There are \ref ads_algorithm_skipElements "functions" for skipping elements
  in a sequence.

  The \ref algorithm_insertion_sort "insertion_sort()" functions are useful
  for sorting nearly sorted sequences.

  The \ref algorithm_selectionSort "selection sort" functions are useful
  for sorting very short sequences.

  The \ref algorithm_statistics "statistics" functions compute the minimum,
  maximum and mean of a range of elements.

  The \ref algorithm_unique "areElementsUnique()" functions determine if
  the unsorted elements in a range are unique.

  The ads::Triplet class is analogous to std::pair.  The ads::OrderedPair
  class is a pair in which the first element precedes the second.

  There are \ref algorithm_extremeElement "functions" to find the extreme
  element (minimum or maximum) in a sequence.
*/

#if !defined(__ads_algorithm_h__)
#define __ads_algorithm_h__

#include "stlib/ads/algorithm/extremeElement.h"
#include "stlib/ads/algorithm/min_max.h"
#include "stlib/ads/algorithm/sign.h"
#include "stlib/ads/algorithm/insertion_sort.h"
#include "stlib/ads/algorithm/OrderedPair.h"
#include "stlib/ads/algorithm/skipElements.h"
#include "stlib/ads/algorithm/sort.h"
#include "stlib/ads/algorithm/statistics.h"
#include "stlib/ads/algorithm/Triplet.h"
#include "stlib/ads/algorithm/unique.h"

#endif
