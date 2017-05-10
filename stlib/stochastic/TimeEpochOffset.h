// -*- C++ -*-

/*!
  \file stochastic/TimeEpochOffset.h
  \brief A reaction.
*/

#if !defined(__stochastic_TimeEpochOffset_h__)
#define __stochastic_TimeEpochOffset_h__

#include <limits>

#include <cassert>

namespace stlib
{
namespace stochastic
{

//! Measure time using an epoch and offset.
/*!
  Measure time when the increments are 32-bit exponential deviates. Note that
  \c double has 53 bits of precision in the mantissa. Thus one can increment
  a number without losing random bits if the ratio of the number to the
  mean of the exponential deviate is no greater than
  \f$2^{53-32} = 2^{21} = 2097152\f$. This class uses two numbers to represent
  the time: an epoch and an offset. The sum of these is the time. When
  incrementing the time, we check if the mean is large enough to avoid
  losing precision. If so, we simply increment the offset. If not, we add
  the offset to the epoch and set the offset to the increment.

  Using a single number to represent the time, one will start losing random
  bits when the time exceeds the mean of the exponential deviate by a factor
  of \f$2^{21} \approx 2 \times 10^{6}\f$. By using an epoch and an offset to
  represent the time, this factor becomes
  \f$2^{42} \approx 4 \times 10^{12}\f$.
*/
class TimeEpochOffset
{
protected:

  //
  // Member data.
  //

  //! The time is the sum of the epoch and the offset.
  double _epoch;
  //! The time is the sum of the epoch and the offset.
  double _offset;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Time is zero.
  TimeEpochOffset() :
    _epoch(0),
    _offset(0) {}

  // Use the default copy constructor, assignment operator and destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the time.
  operator double() const
  {
    return _epoch + _offset;
  }

  //! Get the epoch.
  double
  getEpoch() const
  {
    return _epoch;
  }

  //! Get the offset.
  double
  getOffset() const
  {
    return _offset;
  }

  //! Return true if we should start a new epoch.
  bool
  shouldStartNewEpoch(const double mean) const
  {
    // The maximum allowed ratio of the time offset to the mean time step
    // is 2^(53-32).
    const double inverseMaxRatio = 1. / 2097152.;
    // If we will start losing random bits by adding to the time offset.
    return mean < _offset * inverseMaxRatio;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Set the time offset.
  void
  setOffset(const double offset)
  {
    _offset = offset;
  }

  //! Set the time.
  TimeEpochOffset&
  operator=(const double time)
  {
    _epoch = time;
    _offset = 0;
    return *this;
  }

  //! Increment the time.
  TimeEpochOffset&
  operator+=(const double tau)
  {
#ifdef STLIB_DEBUG
    assert(tau != std::numeric_limits<double>::max());
#endif
    _offset += tau;
    return *this;
  }

  //! Start a new epoch.
  void
  startNewEpoch()
  {
    _epoch += _offset;
    _offset = 0;
  }

  //! Update the epoch if necessary.
  void
  updateEpoch(const double mean)
  {
    if (shouldStartNewEpoch(mean)) {
      startNewEpoch();
    }
  }

  // CONTINUE
#if 0
  //! Increment the time using the mean and an exponential deviate.
  void
  increment(const double mean, const double exponentialDeviate)
  {
    // Handle the special case that the rate is zero (mean is infinite).
    if (mean == std::numeric_limits<double>::max()) {
      _epoch = std::numeric_limits<double>::max();
      _offset = 0;
      return;
    }

    // The maximum allowed ratio of the time offset to the mean time step
    // is 2^(53-32).
    const double maxRatio = 2097152;
    // If we will start losing random bits by adding to the time offset.
    if (mean * maxRatio < _offset) {
      // Start a new time epoch.
      _epoch += _offset;
      _offset = mean * exponentialDeviate;
    }
    else {
      _offset += mean * exponentialDeviate;
    }
  }
#endif

  //@}
};

} // namespace stochastic
}

#endif
