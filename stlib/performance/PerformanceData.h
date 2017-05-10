// -*- C++ -*-

#if !defined(__performance_PerformanceData_h__)
#define __performance_PerformanceData_h__

/*!
  \file performance/PerformanceData.h
  \brief Performance data within a scope.
*/

#include "stlib/performance/SimpleTimer.h"
#include "stlib/performance/Timer.h"

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace stlib
{
namespace performance
{


/// Performance data within a scope.
class PerformanceData {
public:

  /// Numeric data.
  std::map<std::string, double> numerics;
  /// Ordered keys for the numeric data.
  std::vector<std::string> numericKeys;
  /// Record times.
  std::map<std::string, double> times;
  /// Ordered keys for the times.
  std::vector<std::string> timeKeys;
  /// Use this to measure the total time spent in this scope.
  Timer total;

private:

  /// The current event that is being timed.
  double* _current;
  /// Note that, within a scope, we can only measure one event at a time.
  SimpleTimer _timer;

public:

  /// Return true if there is no recorded data.
  bool
  empty() const BOOST_NOEXCEPT;
    
  /// Record the numeric data. Set if new, otherwise increment.
  void
  record(std::string const& key, double value);

  /// Start timing the indicated event.
  void
  start(std::string const& key);

  /// Record the elapsed time for the current event.
  void
  stop();
};


} // namespace performance
} // namespace stlib

#define __performance_PerformanceData_tcc__
#include "stlib/performance/PerformanceData.tcc"
#undef __performance_PerformanceData_tcc__

#endif
