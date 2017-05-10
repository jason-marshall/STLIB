// -*- C++ -*-

#if !defined(__performance_SimpleTimer_h__)
#define __performance_SimpleTimer_h__

/*!
  \file performance/SimpleTimer.h
  \brief High precision timer with a simple interface.
*/

#include <boost/config.hpp>

#include <chrono>

namespace stlib
{
namespace performance
{


/// High precision timer with a simple interface.
/**
   Note that constructing this class does not start the timer.
*/
class SimpleTimer
{
private:

  typedef std::chrono::high_resolution_clock Clock;
  typedef Clock::duration Duration;

  /** If the timer is running, the time since the last epoch. If stopped, the
      time since starting. */
  Duration _time;

public:

  /// Start the timer.
  void
  start() BOOST_NOEXCEPT
  {
    _time = Clock::now().time_since_epoch();
  }

  /// Stop the timer.
  void
  stop() BOOST_NOEXCEPT
  {
    _time = Clock::now().time_since_epoch() - _time;
  }

  /// Return the elapsed time in seconds.
  /** \pre The timer must be stopped. */
  double
  elapsed() const BOOST_NOEXCEPT
  {
    return nanoseconds() * 1e-9;
  }

  /// Return the elapsed time in nanoseconds.
  /** \pre The timer must be stopped. */
  std::chrono::nanoseconds::rep
  nanoseconds() const BOOST_NOEXCEPT
  {
    return std::chrono::nanoseconds(_time).count();
  }
};


} // namespace performance
} // namespace stlib

#endif
