// -*- C++ -*-

/*!
  \file partition.h
  \brief A fair partition of an integer.
*/

/*!
  \page partition Fair Partition Package.

  This package has the
  numerical::partition(const int x,const int n,const int i)
  function for partitioning an integer.
*/

#if !defined(__numerical_partition_h__)
#define __numerical_partition_h__

#include "stlib/ext/vector.h"
#include "stlib/numerical/integer/compare.h"

#include <algorithm>

#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace stlib
{
namespace numerical
{

//! Return the i_th fair partition of x into n parts.
/*!
  Partition \c x into \c n fair parts.  Return the i_th part.  The parts differ
  by at most one and are in non-increasing order.

  \param x is the number to partition.
  \param n is the number of partitions.
  \param i is the requested partition.

  \pre
  \c x is non-negative.
  \c n is positive.
  \c i is in the range [0..n).
*/
template<typename _Integer>
inline
_Integer
getPartition(const _Integer x, const _Integer n, const _Integer i)
{
#ifdef STLIB_DEBUG
  assert(x >= 0 && n > 0 && 0 <= i && i < n);
#endif
  _Integer p = x / n;
  if (i < x % n) {
    ++p;
  }
  return p;
}

//! Return the i_th fair partition of x into n parts.
/*!
  Partition \c x into \c n fair parts. Return the i_th part. The parts differ
  by at most one and are in non-increasing order.

  \param x is the number to partition.
  \param n is the number of partitions.
  \param i is the requested partition.

  \pre
  \c n is positive.
  \c i is in the range [0..n).

  \note
  This specialization avoids checking that the arguments are non-negative.
  (The general implementation induces (harmless but distracting) compiler
  warnings.)
*/
inline
std::size_t
getPartition(const std::size_t x, const std::size_t n, const std::size_t i)
{
#ifdef STLIB_DEBUG
  assert(n > 0 && i < n);
#endif
  std::size_t p = x / n;
  if (i < x % n) {
    ++p;
  }
  return p;
}

//! Return this thread's fair partition of x.
/*!
  \param x is the number to partition. It must be non-negative.

  \pre The function must be called from within a parallel region.
*/
template<typename _Integer>
inline
_Integer
getPartition(const _Integer x)
{
#ifdef _OPENMP
#ifdef STLIB_DEBUG
  assert(omp_in_parallel());
#endif
  // Partition with the number of threads and this thread's number.
  return getPartition(x, _Integer(omp_get_num_threads()),
                      _Integer(omp_get_thread_num()));
#else
  // Serial behavior.
  return x;
#endif
}

//! Compute the i_th fair partition range of x into n ranges.
/*!
  Partition \c x into \c n fair ranges.  Compute the i_th range.  The lengths
  of the ranges differ by at most one and are in non-increasing order.

  \param x is the number to partition.
  \param n is the number of partitions.
  \param i is the requested range.
  \param a is the begining of the range.
  \param b is the end of the open range, [a..b).

  \pre
  \c x is non-negative.
  \c n is positive.
  \c i is in the range [0..n).
*/
template<typename _Integer>
inline
void
getPartitionRange(const _Integer x, const _Integer n, const _Integer i,
                  _Integer* a, _Integer* b)
{
  const _Integer p = x / n;
  *a = p * i;
  *a += std::min(i, x % n);
  *b = *a + getPartition(x, n, i);
}

//! Compute this thread's fair partition range of x.
/*!
  \param x is the number to partition.
  \param a is the begining of the range.
  \param b is the end of the open range, [a..b).

  \pre The function must be called from within a parallel region.
*/
template<typename _Integer>
inline
void
getPartitionRange(const _Integer x, _Integer* a, _Integer* b)
{
#ifdef _OPENMP
  // Partition with the number of threads and this thread's number.
  getPartitionRange(x, _Integer(omp_get_num_threads()),
                    _Integer(omp_get_thread_num()), a, b);
#else
  // Serial behavior.
  *a = 0;
  *b = x;
#endif
}

//! Compute the fair partition ranges of x into n ranges.
/*!
  Partition \c x into \c n fair ranges.  Compute the delimiters for each range.
  The lengths of the ranges differ by at most one and are in non-increasing
  order.

  \param x is the number to partition.
  \param n is the number of partitions.
  \param delimiters is the output iterator for range delimiters. n + 1 integers
  will be written to this iterator.

  \pre
  \c x is non-negative.
  \c n is positive.
*/
template<typename _Integer, typename _OutputIterator>
inline
void
computePartitions(const _Integer x, const _Integer n,
                  _OutputIterator delimiters)
{
  const _Integer p = x / n;
  _Integer d = 0;
  *delimiters++ = d;
  _Integer i;
  for (i = 0; i < x % n; ++i) {
    *delimiters++ = d += p + 1;
  }
  for (; i != n - 1; ++i) {
    *delimiters++ = d += p;
  }
  *delimiters++ = x;
}

//! Compute the fair partition of the vector of weights.
/*!
  \param weights is the vector of nonnegative weights.
  \param numParts is the number of partitions.
  \param delimiters is the output iterator for range delimiters. numParts + 1
  integers will be written to this iterator.

  \pre
  \c The weights are nonnegative.
  \c numParts is positive.
*/
template<typename _Weight, typename _OutputIterator>
inline
void
computePartitions(const std::vector<_Weight>& weights,
                  const std::size_t numParts,
                  _OutputIterator delimiters)
{
  assert(numParts >= 1);
#ifdef STLIB_DEBUG
  if (! weights.empty()) {
    assert(numerical::isNonNegative(ext::min(weights)));
  }
#endif

  // Set the first delimiter.
  *delimiters++ = 0;
  // Determine the end of each part, which is the beginning of the following
  // part.
  std::size_t d = 0;
  _Weight remainingWeight = ext::sum(weights);
  for (std::size_t i = 0; i != numParts - 1; ++i) {
    // The target size for this part is the remaining weight
    // divided by the number of remaining parts.
    _Weight target = remainingWeight / (numParts - i);
    _Weight partialSum = 0;
    // If there are elements remaining, include at least one for this part.
    if (d != weights.size()) {
      partialSum += weights[d];
      ++d;
    }
    // Loop until we should not include the element in the i_th part.
    while (d != weights.size() &&
           !(partialSum + weights[d] >= target &&
             2 * partialSum + weights[d] > 2 * target)) {
      partialSum += weights[d];
      ++d;
    }
    remainingWeight -= partialSum;
    *delimiters++ = d;
  }
  // Set the last delimiter.
  *delimiters++ = weights.size();
}

} // namespace numerical
}

#endif
