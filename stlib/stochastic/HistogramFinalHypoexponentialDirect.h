// -*- C++ -*-

/*!
  \file stochastic/HistogramFinalHypoexponentialDirect.h
  \brief The direct method for SSA.
*/

#if !defined(__stochastic_HistogramFinalHypoexponentialDirect_h__)
#define __stochastic_HistogramFinalHypoexponentialDirect_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/HistogramsPackedArray.h"
#include "stlib/stochastic/TimeEpochOffset.h"

#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate the state in a histogram at the end time.
/*!
  \param _DiscreteGenerator Random deviate generator for the discrete,
  finite distribution with reaction propensities as scaled probabilities.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _DiscreteGenerator,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HistogramFinalHypoexponentialDirect : public Solver
{
private:

  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;
  //! The discrete, finite generator.
  typedef _DiscreteGenerator DiscreteGenerator;
  //! The discrete, uniform generator.
  typedef typename DiscreteGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  //
  // Member data.
  //
protected:

  //! The time is tracked with an epoch and offset.
  TimeEpochOffset _time;
  //! The propensities functor.
  PropensitiesFunctor _propensitiesFunctor;
  //! The reaction influence graph.
  container::StaticArrayOfArrays<std::size_t> _reactionInfluence;
  //! The discrete uniform random number generator.
  DiscreteUniformGenerator _discreteUniformGenerator;
  //! The discrete random number generator.
  DiscreteGenerator _discreteGenerator;
  //! The recording times.
  std::vector<double> _frames;
  //! The species to record.
  std::vector<std::size_t> _recordedSpecies;
  //! Histograms for the recorded species.
  HistogramsPackedArray _histograms;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramFinalHypoexponentialDirect();
  //! Copy constructor not implemented.
  HistogramFinalHypoexponentialDirect
  (const HistogramFinalHypoexponentialDirect&);
  //! Assignment operator not implemented.
  HistogramFinalHypoexponentialDirect&
  operator=(const HistogramFinalHypoexponentialDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  /*!
    There may be only one frame. The argument is received as a vector for
    compatibility with the other solvers.
  */
  HistogramFinalHypoexponentialDirect
  (const State& state, const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins, const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, maxSteps),
    _time(),
    // Copy.
    _propensitiesFunctor(propensitiesFunctor),
    _reactionInfluence(reactionInfluence),
    // Construct.
    _discreteUniformGenerator(),
    _discreteGenerator(&_discreteUniformGenerator),
    _frames(frames),
    _recordedSpecies(recordedSpecies),
    _histograms(frames.size(), recordedSpecies.size(), numberOfBins,
                multiplicity)
  {
    assert(_frames.size() == 1);
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations.
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    // Initialize the state.
    Base::initialize(populations);
    _time = time;
    // Compute the propensities and initialize the discrete, finite generator.
    computePropensities();
    // Prepare for recording a trajectory.
    _histograms.initialize();
  }

  //! Generate a trajectory and record the state in the histogram.
  void
  simulate()
  {
    // The mean of the time.
    double mean = 0;
    // The variance of the time.
    double variance = 0;
    // The standard deviation of the time.
    double stdDev = 0;

    // Step until the expected time is far past the end time.
    while (mean - 4 * stdDev < _frames[0]) {
      // Check that we have not exceed the allowed number of steps.
      if (! incrementStepCount()) {
        setStepCountError(mean);
        break;
      }

      // Record the probability for the current state.
      const double p = probability(_frames[0], mean, variance,
                                   stdDev, _discreteGenerator.sum());

      if (p != 0) {
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(0, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), p);
        }
      }

      // Compute the hypoexponential moments for the next reaction.
      if (_discreteGenerator.isValid()) {
        const double r = 1. / _discreteGenerator.sum();
        mean += r;
        variance += r * r;
        stdDev = std::sqrt(variance);
      }
      else {
        mean = std::numeric_limits<double>::max();
        variance = 0;
        stdDev = 0;
      }

      // Determine the reaction to fire.
      const std::size_t reactionIndex = _discreteGenerator();
#ifdef STLIB_DEBUG
      assert(_discreteGenerator[reactionIndex] > 0);
#endif
      // Fire the reaction.
      _state.fireReaction(reactionIndex);
      // Recompute the propensities and update the discrete, finite generator.
      updatePropensities(reactionIndex);
    }
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    _histograms.synchronize();
  }

protected:

  //! Record a step count error message. Record the mean time.
  void
  setStepCountError(const double mean)
  {
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps
        << " was reached. Mean time = " << mean << ".";
    _error += out.str();
  }

  //! The probability that the current reaction happens before the end time and the next reaction happens after the end time.
  double
  probability(const double t, const double mean, const double variance,
              const double stdDev, const double lambda) const
  {
    const double sqrt2Pi = std::sqrt(2. * 3.1415926535897931);

    if (mean + 4 * stdDev < t || t < mean - 4 * stdDev) {
      return 0;
    }
    return stdDev * exp(-0.5 * (t - mean) * (t - mean) / variance) /
           (sqrt2Pi * (mean + lambda * variance - t));
#if 0
    return 0.5 * std::exp(- lambda * (t - mean - 0.5 * lambda * variance)) *
           (1. + erf((t - mean - lambda * variance) / (sqrt2 * stdDev)));
#endif
  }


  //! The probability that the current reaction happens before the end time.
  double
  cdf(const double t, const double mean, const double stdDev) const
  {
    const double sqrt2 = std::sqrt(2.);

    if (t < mean - 4 * stdDev) {
      return 0;
    }
    if (t > mean + 4 * stdDev) {
      return 1;
    }
    return 0.5 * (1. + erf((t - mean) / (sqrt2 * stdDev)));
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return _discreteUniformGenerator;
  }

  //! Return the vector of recorded species.
  const std::vector<std::size_t>&
  getRecordedSpecies() const
  {
    return _recordedSpecies;
  }

  //! Return the set of histograms.
  const HistogramsPackedArray&
  getHistograms() const
  {
    return _histograms;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a reference to the discrete, finite generator.
  DiscreteGenerator&
  getDiscreteGenerator()
  {
    return _discreteGenerator;
  }

  //! Return a reference to the discrete, uniform generator.
  DiscreteUniformGenerator&
  getDiscreteUniformGenerator()
  {
    return _discreteUniformGenerator;
  }

protected:

  void
  computePropensities()
  {
    // Allocate a propensity array.
    std::vector<double> propensities(_state.getNumberOfReactions());
    // Compute each propensity.
    for (std::size_t i = 0; i != propensities.size(); ++i) {
      propensities[i] = _propensitiesFunctor(i, _state.getPopulations());
    }
    _discreteGenerator.initialize(propensities.begin(), propensities.end());
  }

  void
  updatePropensities(const std::size_t reactionIndex)
  {
    for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
         i = _reactionInfluence.begin(reactionIndex);
         i != _reactionInfluence.end(reactionIndex); ++i) {
      _discreteGenerator.set
      (*i, _propensitiesFunctor(*i, _state.getPopulations()));
    }
    updateSum(std::integral_constant<bool,
              DiscreteGenerator::AutomaticUpdate>());
  }

  //! Do nothing because the discrete generator automatically update the PMF sum.
  void
  updateSum(std::true_type /*Automatic update*/)
  {
  }

  //! Tell the discrete generator to update the PMF sum.
  void
  updateSum(std::false_type /*Automatic update*/)
  {
    _discreteGenerator.updateSum();
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
