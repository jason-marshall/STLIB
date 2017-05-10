// -*- C++ -*-

/*!
  \file stochastic/HistogramReference.h
  \brief A weighted histogram that borrows memory.
*/

#if !defined(__stochastic_HistogramReference_h__)
#define __stochastic_HistogramReference_h__

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

#include <cassert>
#include <cmath>

namespace stlib
{
namespace stochastic
{

//! A histogram that references memory for the bins.
/*!
  This histogram class manages both population statistics and a histogram that
  records events in bins. This class is used by HistogramsPacked, which
  allocates the memory for all of the histograms.

  This class may be used for for either
  <a href="http://en.wikipedia.org/wiki/Weighted_mean">weighted</a> or
  unweighted events. Consider studying the transient behavior of a system
  by record the state at specified points in time (frames). One would collect
  collect statistics and histograms for each species at each frame.
  If one uses \ref stochastic_gillespie1977 "Gillespie's Direct method"
  to generate trajectories, the data collected are \e unweighted events.
  In generating a trajectory one records the species populations whenever
  a frame is crossed, that is when a time step crosses a recording time.
  Since each trajectory is equally likely, the events (species populations)
  are unweighted; they each contribute equally to the moments and histograms.

  Next consider studying the steady state behavior of a system using the
  Direct method. One records statistics and histograms for each species. In
  a single step of the simulation a time increment is calculated by generating
  an exponential deviate and a reaction channel is chosen by generating
  a discrete deviate. The state of the system (species populations) are
  recorded with a weight equal to the time increment. The empirical
  probability of a species population having a certain value is the
  ratio of the time spent in that state and the total simulation time (summed
  over all trajectories). The mean and variance of the species populations
  are weighted statistics.

  The accumulate() function is used for recording both weighted and
  unweighted events. For the latter case one specifies a unit weight.

  Consider a sample of a population {\e x<sub>i</sub>} with associated
  weights {\e w<sub>i</sub>}. The weighted mean is
  \f[
  \mu = \frac{\sum_{i=1}^n w_i x_i}{\sum_{i=1}^n w_i}.
  \f]
  The unbiased estimate of the population variance is
  \f[
  s^2 = \frac{\sum_{i=1}^n w_i (x_i - \mu)^2}{\frac{n-1}{n} \sum_{i=1}^n w_i}.
  \f]
  These are not actually the formulas we use for computing the mean and
  variance. We use \ref stochastic_west1979 "West's" algorithm to
  dynamically update internal variables from which the mean and variance
  may be easily computed. Specifically we store the following:
  - \c _cardinality: The cardinality of the set of measurements.
  - \c _sumOfWeights: The sum of the weights.
  - \c _mean: The mean.
  - \c _summedSecondCenteredMoment: The summed second centered moment
  \f$\sum_i(w_i(x_i-\mu)^2)\f$
  .
  The variance is given by \code _summedSecondCenteredMoment *  _cardinality /
  ((_cardinality - 1) * _sumOfWeights) \endcode.

  This class holds a histogram with a fixed number of bins. As events are
  recorded the lower bound and the bin width is adjusted to cover
  all of the events that have been recorded so far. These quantities are
  modified in such a way that one does not resort to approximations
  when rebuilding the histogram. The following rules summarize the
  procedure.
  - The bin width is a power of 2.
  - The lower bound is a multiple of the bin width.
  - If an event lies outside of the range of the current histogram, the
  bin width is doubled (and the lower bound adjusted) until all events lie
  within the histogram's range.

  Note that this class references the memory for the histogram instead of
  allocating it. The HistogramsPacked and HistogramsAveragePacked classes
  each store a vector of HistogramReference%s and have a contiguous array
  for all of the histogram bins. While this approach complicates the code
  a bit, it yields better cache utilization.
*/
class HistogramReference
{
  //
  // Friends.
  //

  friend std::ostream& operator<<(std::ostream&, const HistogramReference&);

  //
  // Public types.
  //

public:

  //! A const iterator on the weighted probabilities.
  typedef const double* const_iterator;
  //! An iterator on the weighted probabilities.
  typedef double* iterator;

  //
  // Member data.
  //

private:

  //! The cardinality of the set of measurements.
  double _cardinality;
  //! The sum of the weights.
  double _sumOfWeights;
  //! The mean.
  double _mean;
  //! The summed second centered moment sum_i(w_i(x_i-mu)**2)
  double _summedSecondCenteredMoment;

  //! The closed lower bound is a multiple of the width.
  double _lowerBound;
  //! The width of a bin is a power of 2.
  double _width;
  //! The inverse of the bin width.
  double _inverseWidth;
  //! The open upper bound.
  /*! _upperBound == _lowerBound + _width * _bins.size() */
  double _upperBound;
  //! The number of bins.
  std::size_t _size;
  //! The bins contain the weighted probabilities.
  double* _bins;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Default constructor. Invalid histogram.
  HistogramReference() :
    // Statistics.
    _cardinality(0),
    _sumOfWeights(0),
    _mean(0),
    _summedSecondCenteredMoment(0),
    // Histogram.
    _lowerBound(0),
    _width(0),
    _inverseWidth(0),
    _upperBound(0),
    _size(0),
    _bins(0) {}

  // Use the default copy constructor. Reference the same memory.
  // Use the default assignment operator. Reference the same memory.

  //! Construct from the number of bins and a pointer to the bin data.
  HistogramReference(const std::size_t size, double* bins) :
    // Statistics.
    _cardinality(0),
    _sumOfWeights(0),
    _mean(0),
    _summedSecondCenteredMoment(0),
    // Histogram.
    _lowerBound(0),
    _width(0),
    _inverseWidth(0),
    _upperBound(0),
    _size(0),
    _bins(0)
  {
    initialize(size, bins);
  }

  // Use the default destructor.

  //! Initialize with the number of bins and a pointer to the bin data.
  /*!
    Empty the bins.
  */
  void
  initialize(const std::size_t size, double* bins)
  {
    // Statistics.
    _cardinality = 0;
    _sumOfWeights = 0;
    _mean = 0;
    _summedSecondCenteredMoment = 0;
    // Histogram.
    _lowerBound = 0;
    _width = 1;
    _inverseWidth = 1;
    _upperBound = size;
    _size = size;
    _bins = bins;
    std::fill(begin(), end(), 0);
  }

  //! Copy the bin data to a new memory location. Update the bins pointer.
  void
  copyBins(double* bins)
  {
    std::copy(_bins, _bins + _size, bins);
    _bins = bins;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the number of bins.
  std::size_t
  size() const
  {
    return _size;
  }

  //! Return a const iterator to the beginning of the weighted probabilities.
  const_iterator
  begin() const
  {
    return _bins;
  }

  //! Return a const iterator to the end of the weighted probabilities.
  const_iterator
  end() const
  {
    return _bins + _size;
  }

  //! Compute the sum of the weighted probabilities.
  double
  computeSum() const
  {
    return std::accumulate(begin(), end(), 0.);
  }

  //! Return the closed lower bound.
  double
  getLowerBound() const
  {
    return _lowerBound;
  }

  //! Return the bin width.
  double
  getWidth() const
  {
    return _width;
  }

  //! Return the open upper bound.
  double
  getUpperBound() const
  {
    return _upperBound;
  }

  //! Return the specified weighted probability.
  double
  operator[](const std::size_t n) const
  {
#ifdef STLIB_DEBUG
    assert(n < _size);
#endif
    return _bins[n];
  }

  //! Return a closed lower bound on the non-zero probabilities.
  /*!
    \pre There must be at least one non-zero probability.
  */
  double
  computeMinimumNonzero() const
  {
    for (std::size_t i = 0; i != size(); ++i) {
      if (_bins[i] != 0) {
        return _lowerBound + i * _width;
      }
    }
    assert(false);
    return 0.;
  }

  //! Return an open upper bound on the non-zero probabilities.
  /*!
    \pre There must be at least one non-zero probability.
  */
  double
  computeMaximumNonzero() const
  {
    for (std::size_t i = size(); i != 0; --i) {
      if (_bins[i - 1] != 0) {
        return _lowerBound + i * _width;
      }
    }
    assert(false);
    return 0.;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return an iterator to the beginning of the weighted probabilities.
  iterator
  begin()
  {
    return _bins;
  }

  //! Return an iterator to the end of the weighted probabilities.
  iterator
  end()
  {
    return _bins + _size;
  }

  //! Clear the histogram. Empty the bins but do not alter the lower bound or width.
  void
  clear()
  {
    // Statistics.
    _cardinality = 0;
    _sumOfWeights = 0;
    _mean = 0;
    _summedSecondCenteredMoment = 0;
    // Histogram.
    std::fill(begin(), end(), 0.);
  }

  //! Reset the histogram. Empty the bins. Reset the lower bound and width.
  void
  reset()
  {
    _lowerBound = 0;
    _width = 1;
    _inverseWidth = 1;
    _upperBound = size();
    clear();
  }

  //! Accumulate the probability for the event.
  /*! This function uses \ref stochastic_west1979 "West's" algorithm for
    updating the weighted mean and variance.
   \param event is the event value.
   \param weight is the weighted probability of the event occurring. */
  void
  accumulate(const double event, const double weight)
  {
    // Update the statistics.
#if 0
    // CONTINUE: This is less efficient.
    if (_cardinality == 0) {
      _cardinality = 1;
      _sumOfWeights = weight;
      _mean = event;
      _summedSecondCenteredMoment = 0;
    }
    else {
      ++_cardinality;
      const double newSum = _sumOfWeights + weight;
      const double t = (event - _mean) * weight / newSum;
      _summedSecondCenteredMoment += _sumOfWeights * (event - _mean) * t;
      _mean += t;
      _sumOfWeights = newSum;
    }
#endif
    ++_cardinality;
    const double newSum = _sumOfWeights + weight;
    const double t = (event - _mean) * weight / newSum;
    _summedSecondCenteredMoment += _sumOfWeights * (event - _mean) * t;
    _mean += t;
    _sumOfWeights = newSum;
    // Update the histogram.
    rebuild(event);
    _bins[std::size_t((event - _lowerBound) * _inverseWidth)] += weight;
  }

  //! If necessary, rebuild the histogram so it can contain the specified event.
  void
  rebuild(const double event)
  {
    // Do nothing if the event will be placed in the current histogram.
    if (_lowerBound <= event && event < _upperBound) {
      return;
    }
    // Determine the new lower bound.
    double lower = event;
    if (_lowerBound < lower) {
      std::size_t i;
      for (i = 0; i != size(); ++i) {
        if (_bins[i] != 0) {
          break;
        }
      }
      if (i != size() && _lowerBound + i * _width < lower) {
        lower = _lowerBound + i * _width;
      }
    }
    // Determine the new open upper bound.
    // Add one half to get an open upper bound.
    double upper = event + 0.5;
    if (_upperBound > upper) {
      std::ptrdiff_t i;
      for (i = size() - 1; i >= 0; --i) {
        if (_bins[i] != 0) {
          break;
        }
      }
      if (i != -1 && _lowerBound + (i + 1) * _width > upper) {
        upper = _lowerBound + (i + 1) * _width;
      }
    }
    // Rebuild with the new lower and upper bounds.
    rebuild(lower, upper);
  }

  //! Rebuild the histogram so it covers the specified range.
  void
  rebuild(const double low, const double high)
  {
    rebuild(low, high, _width);
  }

  //! Rebuild the histogram so it covers the specified range and has at least the specified minimum width.
  void
  rebuild(const double low, const double high, double newWidth)
  {
#ifdef STLIB_DEBUG
    assert(low >= 0 && low < high);
#endif
    // Determine the new bounds and a bin width.
    // Note that the width is only allowed to grow.
    double newLowerBound = std::floor(low / newWidth) * newWidth;
    double newUpperBound = newLowerBound + size() * newWidth;
    while (high > newUpperBound) {
      newWidth *= 2;
      newLowerBound = std::floor(low / newWidth) * newWidth;
      newUpperBound = newLowerBound + size() * newWidth;
    }
    // Rebuild the histogram.
    setLowerBoundAndWidth(newLowerBound, newWidth);
  }

  //! Rebuild the histogram so it has the specified lower bound and width.
  void
  setLowerBoundAndWidth(const double newLowerBound, const double newWidth)
  {
    // Copy the probabilities.
    const double newInverseWidth = 1. / newWidth;
    const double newUpperBound = newLowerBound + size() * newWidth;
    std::vector<double> newBins(size(), 0.);
    for (std::size_t i = 0; i != size(); ++i) {
      if (_bins[i] != 0) {
        double event = _lowerBound + i * _width;
#ifdef STLIB_DEBUG
        assert(newLowerBound <= event && event < newUpperBound);
#endif
        newBins[std::size_t((event - newLowerBound) * newInverseWidth)] +=
          _bins[i];
      }
    }
    std::copy(newBins.begin(), newBins.end(), begin());
    // New bounds and width.
    _lowerBound = newLowerBound;
    _width = newWidth;
    _inverseWidth = newInverseWidth;
    _upperBound = newUpperBound;
  }

  //@}
};

//! Write the histogram in ascii format.
/*!
  \relates HistogramReference

  \verbatim
  cardinality
  sumOfWeights
  mean
  summedSecondCenteredMoment
  lower bound
  width
  list of weighted probabilities \endverbatim
*/
inline
std::ostream&
operator<<(std::ostream& out, const HistogramReference& x)
{
  // Statistics.
  out << x._cardinality << '\n'
      << x._sumOfWeights << '\n'
      << x._mean << '\n'
      << x._summedSecondCenteredMoment << '\n'
      // Histogram.
      << x.getLowerBound() << '\n'
      << x.getWidth() << '\n';
  std::copy(x.begin(), x.end(), std::ostream_iterator<double>(out, " "));
  out << '\n';
  return out;
}


//! Return true if the two histograms are synchronized.
/*!
  \relates HistogramReference
  \note The two histograms must have the same number of bins.

  The two histograms are synchronized if they have the same lower bounds
  and widths.
*/
bool
areSynchronized(const HistogramReference& x, const HistogramReference& y)
{
  assert(x.size() == y.size());
  return x.getLowerBound() == y.getLowerBound() &&
         x.getWidth() == y.getWidth();
}


//! Synchronize the two histograms so that they have the same lower bounds and widths.
/*!
  \relates HistogramReference

  \note The two histograms must have the same number of bins.
*/
void
synchronize(HistogramReference* x, HistogramReference* y)
{
  double lower, upper;
  if (x->computeSum() == 0) {
    lower = y->getLowerBound();
    upper = y->getUpperBound();
  }
  else if (y->computeSum() == 0) {
    lower = x->getLowerBound();
    upper = x->getUpperBound();
  }
  else {
    lower = std::min(x->computeMinimumNonzero(), y->computeMinimumNonzero());
    upper = std::max(x->computeMaximumNonzero(), y->computeMaximumNonzero());
  }
  double width = std::max(x->getWidth(), y->getWidth());
  x->rebuild(lower, upper, width);
  y->rebuild(lower, upper, width);
#ifdef STLIB_DEBUG
  assert(areSynchronized(*x, *y));
#endif
}


} // namespace stochastic
}

#endif
