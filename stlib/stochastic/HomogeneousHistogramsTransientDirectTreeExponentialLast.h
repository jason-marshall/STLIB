// -*- C++ -*-

/*!
  \file stochastic/HomogeneousHistogramsTransientDirectTreeExponentialLast.h
  \brief The direct method for SSA with hypoexponential sampling.
*/

#if !defined(__stochastic_HomogeneousHistogramsTransientDirectTreeExponentialLast_h__)
#define __stochastic_HomogeneousHistogramsTransientDirectTreeExponentialLast_h__

#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/TimeEpochOffset.h"

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

  This class uses a time-branching version of the trajectory tree method.
  There is a simple trajectory, generated with the Direct method, that forms
  the trunk. For this trajectory the state and time are recorded in the
  member variables \c _state and \c _time . Consider a single frame at which
  we will record the distribution of states. At a certain time, which we
  will define below the state of the trunk is cloned and a trajectory tree
  is formed. The trajectory tree shares state with the trunk, but stores
  its own times. As the trunk is generated, so too is the trajectory tree.
  Below we describe the latter.

  Consider the exponential distribution with rate parameter \f$\lambda\f$.
  \f[ \mathrm{pdf}(x) = \lambda \mathrm{e}^{-\lambda x} \f]
  \f[ \mathrm{cdf}(x) = 1 - \mathrm{e}^{-\lambda x} \f]

  Let \e t be the current time and \e T be the time at which to record
  the state. Let \e X be the random variable for the next time step.
  \e X has an exponential distribution where \f$\lambda\f$ is the sum
  of the propensities. At each step we consider two possibilities:
  -# There are no more reactions before time \e T.
  -# There are one or more reactions before time \e T.
  .
  The probability of each is determined by the exponential distribution.
  \f[ P(X + t > T) = P(X > T - t) = \mathrm{e}^{-\lambda (T-t)}\f]
  \f[ P(X + t \leq T) = P(X \leq T - t) = 1 - \mathrm{e}^{-\lambda (T-t)}\f]
  At each step we split into two states by multiplying the current
  state's probability by these two. We don't need to generate a deviate
  for the first case. We simply record the state with the appropriate
  probability. For the second case we need to generate an exponential
  deviate with the constraint \f$X \leq T - t\f$. Let \e s denote the
  difference \f$T - t\f$. We have calculated the probability that
  an exponential deviate is less that \e s. Let
  \f$p = 1 - \mathrm{e}^{-\lambda s}\f$ denote this probability.
  We generate the constrained exponential deviate with the method of
  inversion. Let \e U be a unit, uniform deviate.
  \f[ 1 - \mathrm{e}^{- \lambda X} = p U \f]
  \f[ \mathrm{e}^{- \lambda X} = 1 - p U \f]
  \f[ X = - \ln(1 - p U) / \lambda \f]

  We will only record a state if its probability is greater than the machine
  precision. This gives us the following constraint on when to begin
  recording states.
  \f[ \mathrm{e}^{-\lambda (T-t)} > \epsilon \f]
  \f[ \lambda (T-t) < -\ln(\epsilon) \f]
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HomogeneousHistogramsTransientDirectTreeExponentialLast :
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

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeExponentialLast();
  //! Copy constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeExponentialLast
  (const HomogeneousHistogramsTransientDirectTreeExponentialLast&);
  //! Assignment operator not implemented.
  HomogeneousHistogramsTransientDirectTreeExponentialLast&
  operator=(const HomogeneousHistogramsTransientDirectTreeExponentialLast&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HomogeneousHistogramsTransientDirectTreeExponentialLast
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, frames,
         recordedSpecies, numberOfBins, multiplicity, maxSteps),
    _continuousUniform(&_discreteUniformGenerator)
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
  }

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate()
  {
    //
    // Initialization.
    //
    const double threshold = - std::log(std::numeric_limits<double>::epsilon());
    std::vector<double> probabilities(_frames.size(), 1.);
    std::vector<double> treeTimes(_frames.size());
    // The active frames.
    std::size_t activeBegin = 0, activeEnd = 0;
    //
    // Step until we have recorded the state at each of the frames.
    //
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        break;
      }

      // Add frames to the active list.
      const double lambda = _discreteGenerator.sum();
      while (activeEnd < _frames.size() &&
             lambda * (_frames[activeEnd] - _time) < threshold) {
        treeTimes[activeEnd] = _time;
        ++activeEnd;
      }

      // Compute the mean of the exponential deviate.
      const double mean = (lambda != 0 ? 1. / lambda :
                           std::numeric_limits<double>::max());
      // For each active frame.
      for (std::size_t frameIndex = activeBegin; frameIndex != activeEnd;
           ++frameIndex) {
        // The long step branch.
        const double probabilityLong =
          std::exp(- lambda * (_frames[frameIndex] - treeTimes[frameIndex]));
        const double stateProbability = probabilities[frameIndex] *
                                        probabilityLong;
        // Record the probabilities for the current state.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(frameIndex, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), stateProbability);
        }
        // The short step branch.
        const double probabilityShort = 1. - probabilityLong;
        probabilities[frameIndex] *= probabilityShort;
        treeTimes[frameIndex] += - mean * std::log(1. - probabilityShort *
                                 _continuousUniform());
      }

      // Compute the time of the next reaction along the trunk.
      _time.updateEpoch(mean);
      _time += mean * _exponentialGenerator();

      // Remove active frames whose probabilities are too low.
      while (activeBegin < activeEnd && probabilities[activeBegin] <
             std::numeric_limits<double>::epsilon()) {
        ++activeBegin;
      }
      // If we have finished recording at all frames.
      if (activeBegin == _frames.size()) {
        return;
      }

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
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  using Base::synchronize;

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
