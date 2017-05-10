// -*- C++ -*-

/*!
  \file particle/neighbors.h
  \brief Common functionality for finding neighbors.
*/

#if !defined(__particle_neighbors_h__)
#define __particle_neighbors_h__

#include "stlib/particle/order.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/numerical/constants/Exponentiation.h"
#include "stlib/performance/SimpleTimer.h"

#include <vector>

namespace stlib
{
namespace particle
{


//! Performance for storing particle neighbors.
class NeighborsPerformance
{
  //
  // Member data.
  //
protected:

  //! The number of times the potential neighbors have been found.
  std::size_t _potentialNeighborsCount;
  //! The number of times the neighbors have been found.
  std::size_t _neighborsCount;
  //! A timer for measuring time spent in various functions.
  performance::SimpleTimer _timer;
  //! The time spent finding potential neighbors.
  double _timePotentialNeighbors;
  //! The time spent finding neighbors.
  double _timeNeighbors;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor.
  NeighborsPerformance();

  //@}
  //--------------------------------------------------------------------------
  //! \name File I/O.
  //@{
public:

  //! Print performance information.
  void
  printPerformanceInfo(std::ostream& out) const;

  //@}
};


} // namespace particle
}

#define __particle_neighbors_tcc__
#include "stlib/particle/neighbors.tcc"
#undef __particle_neighbors_tcc__

#endif
