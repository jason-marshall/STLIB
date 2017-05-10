// -*- C++ -*-

/*!
  \file rcb.h
  \brief Partitioning with recursive coordinate bisection.
*/

#if !defined(__concurrent_partition_rcb_h__)
#define __concurrent_partition_rcb_h__

#include "stlib/ads/functor/compose.h"
#include "stlib/ads/functor/Dereference.h"
#include "stlib/ads/functor/index.h"
#include "stlib/ads/iterator/IndirectIterator.h"

#include "stlib/geom/kernel/BBox.h"

#include <vector>

namespace stlib
{
namespace concurrent
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

//! Determine an RCB partitioning of the records.
/*!
  \param numProcessors is the number of processors.  (Input)
  \param identifiers is a vector of length \c numRecords that holds the
  record identifiers.  (Input/Output)
  \param idPartition is a vector of length \c (numProcessors+1).  The
  elements are pointers into the \c identifiers array that determine
  the partition.  (Output)
  \param positions is an array of length \c numRecords that holds the
  Cartesian positions of the records.  (Input)

  rcb() performs recursive coordinate bisection of data.  (The
  data is composed of records which have identifiers and positions.)
  It determines how to divide the records among the processors.  This is
  a serial algorithm.  It can be run on any processor that has
  the identifiers and the positions of all the records.

  Template parameters:  \c N is the dimension, it must be a positive integer.
  \c IDType is the identifier type.  For example, the identifier type
  might be an integer that is the index of records in an array.  Or
  it could be a pointer to a record.

  \c identifiers and \c idPartition determine the partition
  of the records.  \c idPartition is a vector of pointers into the
  \c identifiers array.  The function sets the values to determine
  how many records each processor should have.  Processor \c n will
  hold records with identifiers in the range [ \c idPartition[n] ..
  \c idPartition[n+1]).  The \c identifiers array will be permuted
  so the approprate record identifiers are in that range.
*/
template<std::size_t N, typename IDType>
void
rcb(const std::size_t numProcessors,
    std::vector<IDType>* identifiers, std::vector<IDType*>* idPartition,
    const std::vector<std::array<double, N> >& positions);

} // namespace concurrent
}

#define __partition_rcb_ipp__
#include "stlib/concurrent/partition/rcb.ipp"
#undef __partition_rcb_ipp__

#endif
