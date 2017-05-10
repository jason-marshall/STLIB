// -*- C++ -*-

#if !defined(__performance_AutoTimer_h__)
#define __performance_AutoTimer_h__

/*!
  \file performance/AutoTimer.h
  \brief Timer that starts in the constructor and stops in the destructor.
*/

#include "stlib/performance/SimpleTimer.h"

#include <iostream>

namespace stlib
{
namespace performance
{


/// Timer that starts in the constructor and stops in the destructor.
class AutoTimer
{
private:

  char const* const _name;
  SimpleTimer _timer;

public:

  /// Start timing the event with the indicated name.
  AutoTimer(char const* const name = "Elapsed time") :
    _name(name)
  {
    _timer.start();
  }

  /// Stop the timer and print the elapsed time.
  ~AutoTimer()
  {
    _timer.stop();
    std::cout << _name << " = " << _timer.elapsed() << " seconds.\n";
  }

  /// No copy constructor.
  AutoTimer(AutoTimer const&) = delete;

  /// No assignment operator.
  void
  operator=(AutoTimer const&) = delete;
};


} // namespace performance
} // namespace stlib

#endif
