// -*- C++ -*-

/*!
  \file stochastic/HomogeneousHistogramsTransientDirectTreeExponentialLimit.h
  \brief The direct method for SSA. Use the trajectory tree method with an exponential distribution limit.
*/

#if !defined(__stochastic_HomogeneousHistogramsTransientDirectTreeExponentialLimit_h__)
#define __stochastic_HomogeneousHistogramsTransientDirectTreeExponentialLimit_h__

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

  Like HomogeneousHistogramsTransientDirectTreeExponentialLast,
  this class uses a time-branching version of the trajectory tree method.
  Before proceeding, review that method. Now instead of generating a
  trajectory tree for each recorded frame, there is a single tree.
  Consider placing the tree at the first step in the simulation. We take a
  single step and generate a number of branches each with a different time
  step. Then we merge the branches into a trunk that has a single state, but
  has different times according to the different branch time steps.
  We can then generate the trunk according to the Direct method. At each step
  we fire a reaction and increment each of the times by a single time step.
  If there are \e B branches in the tree then there is a weighted probablity
  of 1 / \e B associated with each time.

  Now let the number of branches tend to infinity. The
  distribution of time steps in the tree will approach an exponential
  distribution. Thus instead of keeping track of \e B different times, we
  let the time be a probability distribution. When recording the state at
  frames, we weight it by the probability of that state occuring
  at that point in time. This weight is the probability that the last reaction
  occurs before the frame and the next reaction occurs after the frame.
  We can use the CDF of the exponential distribution to calculate these
  probabilities. We call this approach of letting the number of branches
  tend to infinity a <i>limit tree</i>.

  Note that placing the limit tree at the first step in the simulation
  is not necessarily the best place to put it. However, we are constrained
  in locations. We cannot put the limit tree to close to a frame where we
  are recording the state. We need to put it far enough before a frame
  so that the prabability of crossing the frame in a single simulation
  steps is effectively zero (less than the machine epsilon). Subject to
  these constraints, we may put the limit tree in the most advantageous
  position. That position is the time step at which the sum of the propensities
  is minimum. At this point, the mean time step will be largest. This choice
  maximizes the variance in the time distribution so that we more effectively
  sample the state.
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HomogeneousHistogramsTransientDirectTreeExponentialLimit :
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
  HomogeneousHistogramsTransientDirectTreeExponentialLimit();
  //! Copy constructor not implemented.
  HomogeneousHistogramsTransientDirectTreeExponentialLimit(
    const HomogeneousHistogramsTransientDirectTreeExponentialLimit&);
  //! Assignment operator not implemented.
  HomogeneousHistogramsTransientDirectTreeExponentialLimit&
  operator=(const HomogeneousHistogramsTransientDirectTreeExponentialLimit&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HomogeneousHistogramsTransientDirectTreeExponentialLimit
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
    //std::vector<double> probabilities(_frames.size(), 1.);
    //std::vector<double> treeTimes(_frames.size());
    // The active frames.
    std::size_t activeBegin = 0, activeEnd = 0;
    //
    // Consider the possibility that the first step could take us past the
    // next frame time.
    //
    // Complementary CDF for the exponential distribution.
    std::vector<double> ccdf(_frames.size(), 0);
    // Add frames to the active list.
    double minLambda = _discreteGenerator.sum();
    while (activeEnd < _frames.size() &&
           minLambda * (_frames[activeEnd] - _time) < threshold) {
      ccdf[activeEnd] = std::exp(- minLambda * (_frames[activeEnd] - _time));
      // Record the probabilities for the initial state.
      for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
        _histograms(activeEnd, i).accumulate
        (_state.getPopulation(_recordedSpecies[i]), ccdf[activeEnd]);
      }
      ++activeEnd;
    }
    // If no reactions can fire.
    if (minLambda == 0) {
      return;
    }
    // Fire the reaction for the distribution step.
    fireReaction();

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
      // If we are not recording and the current rate is the slowest so far.
      if (activeBegin == activeEnd && lambda < minLambda) {
        // Use the current step for the distribution step.
        std::swap(lambda, minLambda);
      }
      // Compute the mean of the exponential deviate.
      const double mean = (lambda != 0 ? 1. / lambda :
                           std::numeric_limits<double>::max());
      // Compute the time step and increment the time.
      _time.updateEpoch(mean);
      _time += mean * _exponentialGenerator();

      // Add frames to the active list.
      while (activeEnd < _frames.size() &&
             minLambda * (_frames[activeEnd] - _time) < threshold) {
        ++activeEnd;
      }

      // For each active frame.
      for (std::size_t frameIndex = activeBegin; frameIndex != activeEnd;
           ++frameIndex) {
        // time = t0 + tau
        // P(X + time > T) - P(X + t0 > T)
        // The new value of the complementary CDF.
        const double cc =
          std::min(1., std::exp(- minLambda * (_frames[frameIndex] - _time)));
        const double stateProbability = cc - ccdf[frameIndex];
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
