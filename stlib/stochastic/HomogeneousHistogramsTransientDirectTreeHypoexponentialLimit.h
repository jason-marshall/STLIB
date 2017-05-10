// -*- C++ -*-

/*!
  \file stochastic/HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit.h
  \brief The direct method for SSA. Use the trajectory tree method with an exponential distribution limit.
*/

#if !defined(__stochastic_HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit_h__)
#define __stochastic_HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit_h__

#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/TimeEpochOffset.h"

#include "stlib/numerical/random/hypoexponential/HypoexponentialDistributionDistinctDynamicMinimumParameters.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate histograms at specified frames using Gillespie's direct method.
/*!
  \param _DiscreteGenerator Random deviate generator for the discrete,
  finite distribution with reaction propensities as scaled probabilities.
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.

  Like HomogeneousHistogramsTransientDirectTreeExponentialLimit,
  this class uses a time-branching, limit version of the trajectory tree method.
  Before proceeding, review that method. Now instead of having a limit tree
  that spans a single simulation step, the tree may span an arbitrary number
  of steps. This means that time now has a hypoexponential distribution.
  The simulation steps in the tree need not be consecutive; we choose the steps
  with the smallest summed reaction propensities. As with the single-step tree,
  we are constrained in choosing the steps. They may not be placed to close
  to a recording frame.

  When we are in the process of recording the state at a frame, we can add
  steps to the limit tree, but we cannot replace steps.

  Note that for arbitrary parameters the hypoexponential distribution has
  a complicated CDF. However if the parameters are distinct the formulas are
  simple. Thus we add the constraint that the summed propensity values
  are distinct. For this case the complementary CDF is:
  \f[
  \mathrm{ccdf}(t) = \sum_{i = 1}^{n} c_i \exp(- \alpha_i t)
  \f]
  \f[
  c_i = \prod_{j = 1, j \neq i}^{n} \frac{\alpha_j}{\alpha_j - \alpha_i}
  \f]
  where the \f$\alpha_i\f$ are the parameters.
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit :
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
  //! The hypoexponential distribution.
  numerical::HypoexponentialDistributionDistinctDynamicMinimumParameters
  _hypoexponential;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit();
  //! Copy constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit(
    const HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit&);
  //! Assignment operator not implemented.
  HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit&
  operator=(const HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HomogeneousHistogramsTransientDirectTreeHypoexponentialLimit
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps,
   const std::size_t numberOfParameters) :
    Base(state, propensitiesFunctor, reactionInfluence, frames,
         recordedSpecies, numberOfBins, multiplicity, maxSteps),
    _continuousUniform(&_discreteUniformGenerator),
    _hypoexponential(numberOfParameters)
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
    // The active frames.
    std::size_t activeBegin = 0, activeEnd = 0;
    // Complementary CDF for the exponential distribution.
    std::vector<double> ccdf(_frames.size(), 0);

    //
    // Step until we have recorded the state at each of the frames.
    //
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        break;
      }

      double lambda = _discreteGenerator.sum();
      // If we are not recording.
      if (activeBegin == activeEnd) {
        // We may insert or replace a parameter.
        lambda = _hypoexponential.insertOrReplace(lambda);
      }
      else {
        // We may only insert a parameter.
        lambda = _hypoexponential.insert(lambda);
      }
      // If we will add a regular step to the trajectory (not to the limit
      // tree).
      if (lambda != std::numeric_limits<double>::max()) {
        // Compute the mean of the exponential deviate.
        const double mean = (lambda != 0 ? 1. / lambda :
                             std::numeric_limits<double>::max());
        // Compute the time step and increment the time.
        _time.updateEpoch(mean);
        _time += mean * _exponentialGenerator();
      }

      // Add frames to the active list.
      while (activeEnd < _frames.size() &&
             _hypoexponential.isCcdfNonzero(_frames[activeEnd] - _time)) {
        ++activeEnd;
      }

      // For each active frame.
      for (std::size_t frameIndex = activeBegin; frameIndex != activeEnd;
           ++frameIndex) {
        // The new value of the complementary CDF.
        const double cc = _hypoexponential.ccdf(_frames[frameIndex] - _time);
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
      while (activeBegin < activeEnd && _time > _frames[activeBegin]) {
        ++activeBegin;
      }
      // If we have finished recording at all frames.
      if (activeBegin == _frames.size()) {
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
