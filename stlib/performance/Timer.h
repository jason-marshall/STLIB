// -*- C++ -*-

#if !defined(__performance_Timer_h__)
#define __performance_Timer_h__

/*!
  \file performance/Timer.h
  \brief High precision timer.
*/

#include <boost/config.hpp>

#include <chrono>

namespace stlib
{
namespace performance
{


/// High precision timer.
class Timer
{
private:

  typedef std::chrono::high_resolution_clock Clock;
  typedef Clock::duration Duration;

  /** If the timer is running, the time since the last epoch. If stopped, the
      time since starting. */
  Duration _time;
  /// True if the timer is stopped.
  bool _isStopped;

public:

  /// Construct and start the timer.
  Timer() BOOST_NOEXCEPT;

  /// Return true if the timer is stopped.
  bool
  isStopped() const BOOST_NOEXCEPT
  {
    return _isStopped;
  }

  /// Start the timer.
  void
  start() BOOST_NOEXCEPT;

  /// Stop the timer.
  void
  stop() BOOST_NOEXCEPT;

  /// Resume timing.
  void
  resume() BOOST_NOEXCEPT;

  /// Return the elapsed time in seconds.
  double
  elapsed() const BOOST_NOEXCEPT;

  /// Return the elapsed time in nanoseconds.
  std::chrono::nanoseconds::rep
  nanoseconds() const BOOST_NOEXCEPT;
};


} // namespace performance
} // namespace stlib

#define __performance_Timer_tcc__
#include "stlib/performance/Timer.tcc"
#undef __performance_Timer_tcc__

#endif
