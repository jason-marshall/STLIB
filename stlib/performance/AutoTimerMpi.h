// -*- C++ -*-

#if !defined(__performance_AutoTimerMpi_h__)
#define __performance_AutoTimerMpi_h__

/*!
  \file performance/AutoTimerMpi.h
  \brief Timer that starts in the constructor and stops in the destructor.
*/

#include "stlib/performance/SimpleTimer.h"

#include "stlib/mpi/statistics.h"

namespace stlib
{
namespace performance
{


/// Timer that starts in the constructor and stops in the destructor.
class AutoTimerMpi
{
private:

  char const* const _name;
  MPI_Comm _comm;
  SimpleTimer _timer;

public:

  /// Start timing the event with the indicated name.
  AutoTimerMpi(char const* const name = "Elapsed time",
               MPI_Comm comm = MPI_COMM_WORLD) :
    _name(name),
    _comm(comm)
  {
    _timer.start();
  }

  /// Stop the timer and print the elapsed time.
  ~AutoTimerMpi()
  {
    _timer.stop();
    mpi::printStatistics(std::cout, _name, _timer.elapsed(), _comm);
  }

  /// No copy constructor.
  AutoTimerMpi(AutoTimerMpi const&) = delete;

  /// No assignment operator.
  void
  operator=(AutoTimerMpi const&) = delete;
};


} // namespace performance
} // namespace stlib

#endif
