// -*- C++ -*-

#if !defined(__ads_timer_Timer_ipp__)
#error This file is an implementation detail of Timer.
#endif

namespace stlib
{
namespace ads
{

#ifdef __APPLE__

// Use the mach absolute time units. Information is available at
// http://developer.apple.com/mac/library/qa/qa2004/qa1398.html

inline
Timer::
Timer() :
  // Initialize with an invalid start time.
  _start(std::numeric_limits<uint64_t>::max()),
  _elapsed(0)
{
  mach_timebase_info(&_timeBaseInfo);
}

inline
void
Timer::
start()
{
  _start = mach_absolute_time();
}

inline
void
Timer::
stop()
{
#ifdef STLIB_DEBUG
  // Check that the clock was started.
  assert(_start != std::numeric_limits<uint64_t>::max());
#endif
  _elapsed += mach_absolute_time() - _start;
}

inline
void
Timer::
reset()
{
  _start = std::numeric_limits<uint64_t>::max();
  _elapsed = 0;
}

inline
Timer::
operator double() const
{
  return 1e-9 * (_elapsed * _timeBaseInfo.numer / _timeBaseInfo.denom);
}

#elif defined __linux__
// stolen from
// http://www.guyrutenberg.com/2007/09/22/profiling-code-using-clock_gettime/
inline
timespec
diff(timespec start, timespec end)
{
  //printf("diff called on \nend   = (%3ld, %10ld)\nstart = (%3ld, %10ld)\n",
  //end.tv_sec, end.tv_nsec, start.tv_sec, start.tv_nsec);
  timespec temp;
  if ((end.tv_nsec - start.tv_nsec) < 0) {
    temp.tv_sec = end.tv_sec - start.tv_sec - 1;
    temp.tv_nsec = 1000000000 + end.tv_nsec - start.tv_nsec;
  }
  else {
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  }
  return temp;
}

inline
Timer::
Timer() :
  // Initialize with an invalid start time.
  _start(),
  _elapsed()
{
  _start.tv_sec = std::numeric_limits<time_t>::max();
  _start.tv_nsec = std::numeric_limits<long>::max();
  _elapsed.tv_sec = 0;
  _elapsed.tv_nsec = 0;
}

inline
void
Timer::
start()
{
  clock_gettime(CLOCK_REALTIME, &_start);
}

inline
void
Timer::
stop()
{
#ifdef STLIB_DEBUG
  // Check that the clock was started.
  assert(_start.tv_sec != std::numeric_limits<time_t>::max() &&
         _start.tv_nsec != std::numeric_limits<long>::max());
#endif
  timespec temp;
  clock_gettime(CLOCK_REALTIME, &temp);
  temp = diff(_start, temp);
  _elapsed.tv_sec += temp.tv_sec;
  _elapsed.tv_nsec += temp.tv_nsec;
  if (_elapsed.tv_nsec > 1000000000) {
    _elapsed.tv_sec++;
    _elapsed.tv_nsec -= 1000000000;
  }
}

inline
void
Timer::
reset()
{
  _start.tv_sec = std::numeric_limits<time_t>::max();
  _start.tv_nsec = std::numeric_limits<long>::max();
  _elapsed.tv_sec = 0;
  _elapsed.tv_nsec = 0;
}

inline
Timer::
operator double() const
{
  return double(_elapsed.tv_sec + _elapsed.tv_nsec / 1e9);
}


#else
//----------------------------------------------------------------------------

inline
Timer::
Timer() :
  // Initialize with an invalid start time.
  _start(std::numeric_limits<clock_t>::max()),
  _elapsed(0)
{
}

inline
void
Timer::
start()
{
  _start = std::clock();
}

inline
void
Timer::
stop()
{
#ifdef STLIB_DEBUG
  // Check that the clock was started.
  assert(_start != std::numeric_limits<clock_t>::max());
#endif
  _elapsed += std::clock() - _start;
}

inline
void
Timer::
reset()
{
  _start = std::numeric_limits<clock_t>::max();
  _elapsed = 0;
}

inline
Timer::
operator double() const
{
  return double(_elapsed) / CLOCKS_PER_SEC;
}

#endif

} // namespace ads
}
