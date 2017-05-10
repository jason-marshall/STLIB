// -*- C++ -*-

#if !defined(__performance_Timer_tcc__)
#error This file is an implementation detail of Timer.
#endif


namespace stlib
{
namespace performance
{


inline
Timer::
Timer() BOOST_NOEXCEPT
{
  start();
}

inline
void
Timer::
start() BOOST_NOEXCEPT
{
  _isStopped = false;
  _time = Clock::now().time_since_epoch();
}

inline
void
Timer::
stop() BOOST_NOEXCEPT
{
  if (_isStopped) {
    return;
  }
  _isStopped = true;
  _time = Clock::now().time_since_epoch() - _time;
}

inline
void
Timer::
resume() BOOST_NOEXCEPT
{
  if (_isStopped) {
    Duration t = _time;
    start();
    _time -= t;
  }
}

inline
double
Timer::
elapsed() const BOOST_NOEXCEPT
{
  return nanoseconds() * 1e-9;
}

inline
std::chrono::nanoseconds::rep
Timer::
nanoseconds() const BOOST_NOEXCEPT
{
  Duration d;
  if (_isStopped) {
    d = _time;
  }
  else {
    d = Clock::now().time_since_epoch() - _time;
  }
  return std::chrono::nanoseconds(d).count();
}


} // namespace performance
} // namespace stlib
