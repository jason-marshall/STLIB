// -*- C++ -*-

#if !defined(__sfc_gatherRelevant_h__)
#define __sfc_gatherRelevant_h__

/*!
  \file
  \brief Gather the relevant objects to each process.
*/

#include "stlib/sfc/AdaptiveCellsMpi.h"
#include "stlib/numerical/integer/bits.h"

#include <memory>

namespace stlib
{
namespace sfc
{


/// Gather the relevant objects using a point-to-point communication pattern.
/**
   \param objects The vector of local objects.
   \param distributedObjectCells A cell data structure for the distributed
   objects. This is used to define the set of permissible cells when 
   building a AdaptiveCells for the local objects.
   \param relevantCells The cell indices in distributedObjectCells that 
   are relevant for this process.
   \param comm The MPI communicator.
   \return The vector of relevant objects.
*/
template<typename _Object, typename _Traits, typename _Cell>
inline
std::vector<_Object>
gatherRelevant(
  std::vector<_Object> const& objects,
  stlib::sfc::AdaptiveCells<_Traits, _Cell, true> const&
  distributedObjectCells,
  std::vector<std::size_t> const& relevantCells,
  MPI_Comm const comm)
{
  // Use the point-to-point communication pattern.
  return gatherRelevantPointToPoint(objects, distributedObjectCells,
                                    relevantCells, comm);
}


} // namespace sfc
} // namespace stlib

#define __sfc_gatherRelevant_tcc__
#include "stlib/sfc/gatherRelevant.tcc"
#undef __sfc_gatherRelevant_tcc__

#endif
