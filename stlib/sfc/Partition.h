// -*- C++ -*-

#if !defined(__sfc_Partition_h__)
#define __sfc_Partition_h__

/*!
  \file
  \brief Partition a set of cells with associated codes.
*/

#include "stlib/sfc/NonOverlappingCells.h"

#include "stlib/numerical/partition.h"

namespace stlib
{
namespace sfc
{

//! Partition a set of cells with associated codes.
/*!
  Partition the cells of either a UniformCells or a AdaptiveCells data
  structure. (This functor may not be used with LinearOrthantTrie because 
  its cells overlap.) In partitioning, the number of objects in the cells
  are used as weights.

  We use a functor instead of a function interface so that storing the
  vector of delimiters is convenient. 
*/
template<typename _Traits>
class Partition
{
  // Types.
public:

  //! The unsigned integer type is used for codes.
  typedef typename _Traits::Code Code;

  // Member data.
public:

  //! A vector of p + 1 code delimiters.
  std::vector<Code> delimiters;

  // Member functions.
public:

  //! Initialize using the number of parts.
  /*! Allocate the vector of delimiters and fill with invalid values. */
  Partition(std::size_t numParts) :
    delimiters(numParts + 1, std::size_t(-1))
  {
  }

  //! Return the number of parts.
  std::size_t
  size() const
  {
    return delimiters.size() - 1;
  }

  //! Partition the cells.
  template<typename _Cell, template<typename> class _Order>
  void
  operator()(NonOverlappingCells<_Traits, _Cell, true, _Order> const& cells);
};


} // namespace sfc
} // namespace stlib

#define __sfc_Partition_tcc__
#include "stlib/sfc/Partition.tcc"
#undef __sfc_Partition_tcc__

#endif
