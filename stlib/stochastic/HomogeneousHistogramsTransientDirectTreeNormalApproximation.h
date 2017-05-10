// -*- C++ -*-

/*!
  \file stochastic/HomogeneousHistogramsTransientDirectTreeNormalApproximation.h
  \brief The direct method for SSA. Use the trajectory tree method with a normal approximation of the hypoexponential distribution.
*/

#if !defined(__stochastic_HomogeneousHistogramsTransientDirectTreeNormalApproximation_h__)
#define __stochastic_HomogeneousHistogramsTransientDirectTreeNormalApproximation_h__

#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/TimeEpochOffset.h"

#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionNormalApproximation.h"

namespace stlib
{
namespace stochastic
{

//! Trajectory tree method with a normal approximation of the hypoexponential distribution.
/*!
  \param _DiscreteGenerator Random deviate generator for the discrete,
  finite distribution with reaction propensities as scaled probabilities.
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.

  The method implemented here is a logical extension of the method in
  HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit .
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HomogeneousHistogramsTransientDirectTreeNormalApproximation :
  public HistogramsBase < _DiscreteGenerator, _ExponentialGenerator,
  _PropensitiesFunctor >
{
  //
  // Private types.
  //
private:

  typedef HistogramsBase < _DiscreteGenerator, _ExponentialGenerator,
          _PropensitiesFunctor > Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef typename Base::PropensitiesFunctor PropensitiesFunctor;
  //! The exponential generator.
  typedef typename Base::ExponentialGenerator ExponentialGenerator;
  //! The discrete, finite generator.
  typedef typename Base::DiscreteGenerator DiscreteGenerator;
  //! The discrete, uniform generator.
  typedef typename Base::DiscreteUniformGenerator DiscreteUniformGenerator;

  //
  // Member data.
  //
protected:

  using Base::_state;
  using Base::_time;
  using Base::_error;
  using Base::_discreteUniformGenerator;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;
  using Base::_frames;
  using Base::_recordedSpecies;
  using Base::_histograms;

  //! The continuous, uniform random number generator.
  numerical::ContinuousUniformGeneratorOpen<DiscreteUniformGenerator, double>
  _continuousUniform;
  //! The normal approximation to the hypoexponential distribution.
  numerical::HypoexponentialDistributionNormalApproximation _hypoexponential;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeNormalApproximation();
  //! Copy constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeNormalApproximation(
    const HomogeneousHistogramsTransientDirectTreeNormalApproximation&);
  //! Assignment operator not implemented.
  HomogeneousHistogramsTransientDirectTreeNormalApproximation&
  operator=(const HomogeneousHistogramsTransientDirectTreeNormalApproximation&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HomogeneousHistogramsTransientDirectTreeNormalApproximation
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps,
   const double allowedError) :
    Base(state, propensitiesFunctor, reactionInfluence, frames,
         recordedSpecies, numberOfBins, multiplicity, maxSteps),
    _continuousUniform(&_discreteUniformGenerator),
    _hypoexponential(allowedError)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    Base::initialize(populations, time);
    _hypoexponential.clear();
  }

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate()
  {
    const double startTime = _time;
    // The active frames.
    std::size_t activeBegin = 0, activeEnd = 0, approximateBegin = 0;
    // Complementary CDF for the exponential distribution.
    const std::size_t framesSize = _frames.size();
    std::vector<double> ccdf(framesSize, 0);

    //
    // Step until we have recorded the state at each of the frames.
    //
    double lambda = 0, mean = 0;
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        return;
      }

      lambda = _discreteGenerator.sum();
      // The special case that no more reactions can fire.
      if (lambda == 0) {
        mean = std::numeric_limits<double>::max();
        _hypoexponential.setMeanToInfinity();
      }
      else {
        mean = 1. / lambda;
        _hypoexponential.insertInverse(mean);
      }
      // Only compute exponential deviates while the exact frames
      // are being recorded.
      if (activeBegin == activeEnd || activeBegin != approximateBegin) {
        // Compute the time step and increment the time.
        _time.updateEpoch(mean);
        _time += mean * _exponentialGenerator();
      }

      // Add frames to the active list.
      while (activeEnd < framesSize &&
             _hypoexponential.isCcdfNonzero(_frames[activeEnd] - startTime)) {
        // If we can't use the normal approximation.
        if (! _hypoexponential.isValid()) {
          // Add the frame to the exact group.
          ++approximateBegin;
        }
        // Add the frame to the active group.
        ++activeEnd;
      }

      // For each active frame in the exact group that we have passed in this
      // step.
      while (activeBegin != approximateBegin && _time > _frames[activeBegin]) {
        // Record the current state.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(activeBegin, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), 1.);
        }
        // Remove the frame from the active list.
        ++activeBegin;
      }

      // For each active frame in the approximate group.
      for (std::size_t frameIndex = approximateBegin; frameIndex != activeEnd;
           ++frameIndex) {
        // The new value of the complementary CDF.
        const double cc = _hypoexponential.ccdf(_frames[frameIndex]
                                                - startTime);
        // Use max to avoid small negative probabilities that result from
        // truncation errors.
        double stateProbability = std::max(0., cc - ccdf[frameIndex]);
        ccdf[frameIndex] = cc;
        // Record the probabilities for the current state.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(frameIndex, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), stateProbability);
        }
      }

      // Remove approximate frames that we have passed.
      while (activeBegin == approximateBegin && activeBegin < activeEnd &&
             ccdf[activeBegin] == 1) {
        ++activeBegin;
        ++approximateBegin;
      }
      // If we have finished recording at all frames.
      if (activeBegin == framesSize) {
        return;
      }
      // Fire a reaction.
      fireReaction();
      // If there are no more exact frames.
      if (activeBegin == approximateBegin && _hypoexponential.isValid()) {
        // Go to the simplified loop below.
        break;
      }
    }

    // Simplified loop for recording only approximate frames.
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        return;
      }

      lambda = _discreteGenerator.sum();
      // The special case that no more reactions can fire.
      if (lambda == 0) {
        mean = std::numeric_limits<double>::max();
        _hypoexponential.setMeanToInfinity();
      }
      else {
        mean = 1. / lambda;
        _hypoexponential.insertInverse(mean);
      }

      // Add frames to the active list.
      while (activeEnd < framesSize &&
             _hypoexponential.isCcdfNonzero(_frames[activeEnd] - startTime)) {
        // Add the frame to the active group.
        ++activeEnd;
      }

      // For each active frame.
      for (std::size_t frameIndex = activeBegin; frameIndex != activeEnd;
           ++frameIndex) {
        // The new value of the complementary CDF.
        const double cc = _hypoexponential.ccdf(_frames[frameIndex]
                                                - startTime);
        // Use max to avoid small negative probabilities that result from
        // truncation errors.
        double stateProbability = std::max(0., cc - ccdf[frameIndex]);
        ccdf[frameIndex] = cc;
        // Record the probabilities for the current state.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(frameIndex, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), stateProbability);
        }
      }

      // Remove frames that we have passed.
      while (activeBegin < activeEnd && ccdf[activeBegin] == 1) {
        ++activeBegin;
      }
      // If we have finished recording at all frames.
      if (activeBegin == framesSize) {
        return;
      }
      // Fire a reaction.
      fireReaction();
    }
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  using Base::synchronize;

  //! Fire a reaction. Update the discrete generator.
  void
  fireReaction()
  {
    // Determine the reaction to fire.
    const std::size_t reactionIndex = _discreteGenerator();
#ifdef STLIB_DEBUG
    assert(_discreteGenerator[reactionIndex] > 0);
#endif
    // Fire the reaction.
    _state.fireReaction(reactionIndex);
    // Recompute the propensities and update the discrete, finite generator.
    Base::updatePropensities(reactionIndex);
  }

protected:

  //! Record a step count error message. Record the current time.
  using Base::setStepCountError;

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  using Base::getStepCount;

  //! Return a const reference to the state.
  using Base::getState;

  //! Return a const reference to the discrete, uniform generator.
  using Base::getDiscreteUniformGenerator;

  //! Return the vector of recorded species.
  using Base::getRecordedSpecies;

  //! Return the set of histograms.
  using Base::getHistograms;

  //@}
};

//@}

} // namespace stochastic
}

#endif
