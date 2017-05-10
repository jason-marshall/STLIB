// -*- C++ -*-

/*!
  \file ads/timer/Timer.h
  \brief Implements a class for a timer.
*/

#if !defined(__ads_Timer_h__)
#define __ads_Timer_h__

#include <cassert>
#include <limits>

#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <ctime>
#endif

namespace stlib
{
namespace ads
{

//! A timer that measures elapsed time in seconds.
/*!
  \deprecated
  Use the timers in the performance package instead.

  Note that this class has specialization for darwin and linux. To use
  the timer on linux, you will need to link with the rt library.

  The member function tic() and toc() provide a simple way to time an event.
  \code
  Timer timer;
  timer.tic();
  // Code to time.
  ...
  double elapsedTime = timer.toc();
  std::cout << "Elapsed time = " << elapsedTime << " seconds.\n";
  \endcode

  If you need to stop and restart the timer use start(), stop(), and reset().
  \code
  Timer timer;
  for (std::size_t i = 0; i != 100; ++i) {
  timer.start();
  // Code to time.
  ...
  timer.stop();
  }
  std::cout << "Elapsed time = " << timer << " seconds.\n";

  // Time something else with the same timer.
  timer.reset();
  timer.start();
  // Code to time.
  ...
  timer.stop();
  std::cout << "Elapsed time = " << timer << " seconds.\n";
  \endcode

  If the mach kernel is available, this timer uses the high resolution
  mach timer. Otherwise is uses the low resolution timer available
  through \c std::clock().

  \todo Check the availability of the mach kernel on Linux.
*/
class Timer
{
private:

#ifdef __APPLE__
  uint64_t _start;
  uint64_t _elapsed;
  mach_timebase_info_data_t _timeBaseInfo;
#elif defined __linux__
  timespec _start;
  timespec _elapsed;
#else
  clock_t _start;
  clock_t _elapsed;
#endif

public:

  //! Default constructor.
  Timer();

  // Use the default copy constructor, assignment operator, and destructor.

  //! Start/reset the timer.
  void
  tic()
  {
    reset();
    start();
  }

  //! Return the time in seconds since the last tic.
  double
  toc()
  {
    stop();
    return *this;
  }

  //! Start/restart the timer.
  void
  start();

  //! Stop the timer.
  void
  stop();

  //! Reset the timer.
  void
  reset();

  //! Return the elapsed time in seconds.
  operator double() const;
};

} // namespace ads
}

#define __ads_timer_Timer_ipp__
#include "stlib/ads/timer/Timer.ipp"
#undef __ads_timer_Timer_ipp__

#endif
