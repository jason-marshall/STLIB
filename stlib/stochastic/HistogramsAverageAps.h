// -*- C++ -*-

/*!
  \file stochastic/HistogramsAverageAps.h
  \brief Collect the average values of species populations. Use the all possible states method.
*/

#if !defined(__stochastic_HistogramsAverageAps_h__)
#define __stochastic_HistogramsAverageAps_h__

#include "stlib/stochastic/HistogramsAverageBase.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate a histogram of equilibrium state using Gillespie's direct method.
/*!
  \param _DiscreteGenerator Random deviate generator for the discrete,
  finite distribution with reaction propensities as scaled probabilities.
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _DiscreteGenerator,
           class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class HistogramsAverageAps :
  public HistogramsAverageBase < _DiscreteGenerator, _ExponentialGenerator,
  _PropensitiesFunctor >
{
  //
  // Private types.
  //
private:

  typedef HistogramsAverageBase < _DiscreteGenerator, _ExponentialGenerator,
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
  using Base::_tau;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;
  using Base::_recordedSpecies;
  using Base::_histograms;
  using Base::_reactionInfluence;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAverageAps();
  //! Copy constructor not implemented.
  HistogramsAverageAps(const HistogramsAverageAps&);
  //! Assignment operator not implemented.
  HistogramsAverageAps&
  operator=(const HistogramsAverageAps&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsAverageAps
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, recordedSpecies,
         numberOfBins, multiplicity, maxSteps)
  {
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  using Base::initialize;

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate(const double equilibrationTime, const double recordingTime)
  {
    // Let the process equilibrate.
    Base::equilibrate(equilibrationTime);
    // Step until we have reached the end time.
    const double endTime = _time + recordingTime;
    while (true) {
      // Check that we have not exceeded the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        break;
      }

      // Compute an exponential deviate and the time step. Note that we don't
      // use computeTau() because we need the exponential deviate and not just
      // the time step.
      const double exponentialDeviate = _exponentialGenerator();
      if (_discreteGenerator.isValid()) {
        const double mean = 1.0 / _discreteGenerator.sum();
        _time.updateEpoch(mean);
        _tau = mean * exponentialDeviate;
      }
      else {
        _tau = std::numeric_limits<double>::max();
      }

      // If we have not passed the end time.
      if (_time + _tau < endTime) {
        // Increment the time.
        _time += _tau;
      }
      else {
        // Reduce the time step and indicate that this is the last step.
        _tau = endTime - _time;
        _time = endTime;
      }

      // For each reaction.
      for (std::size_t reactionIndex = 0;
           reactionIndex != _state.getNumberOfReactions(); ++reactionIndex) {
        // Skip reactions that have zero probability.
        if (_discreteGenerator[reactionIndex] == 0) {
          continue;
        }
        // This will hold the sum of the propensities after firing the reaction.
        double sum = _discreteGenerator.sum();
        // Fire the reaction.
        _state.fireReaction(reactionIndex);
        // Compute the sum of the propensities after firing the reaction.
        for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
             i = _reactionInfluence.begin(reactionIndex);
             i != _reactionInfluence.end(reactionIndex); ++i) {
          sum += this->_propensitiesFunctor(*i, _state.getPopulations()) -
                 _discreteGenerator[*i];
        }
        // Record the probabilities for this state.
        // P(reaction) == _discreteGenerator[reactionIndex] /
        //                _discreteGenerator.sum()
        // Time to hold state == exponentialDeviate / sum
        // The weight is the product of these two.
        const double weight = _discreteGenerator[reactionIndex] *
                              exponentialDeviate / (_discreteGenerator.sum() * sum);
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                    weight);
        }
        // Un-fire the reaction.
        _state.unFireReaction(reactionIndex);
      }

      // If we have reached the end time.
      if (_time >= endTime) {
        // End the simulation.
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
