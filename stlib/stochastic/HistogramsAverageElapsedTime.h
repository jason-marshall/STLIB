// -*- C++ -*-

/*!
  \file stochastic/HistogramsAverageElapsedTime.h
  \brief Collect the average values of species populations.
*/

#if !defined(__stochastic_HistogramsAverageElapsedTime_h__)
#define __stochastic_HistogramsAverageElapsedTime_h__

#include "stlib/stochastic/HistogramsAverageBase.h"
#include "stlib/stochastic/modifiedRecordedSpecies.h"

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
class HistogramsAverageElapsedTime :
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

  //! The times at which each species was last modified.
  std::vector<double> _lastModified;
  //! The recorded species that are modified by each reaction.
  container::StaticArrayOfArrays<std::size_t> _modifiedRecordedSpecies;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAverageElapsedTime();
  //! Copy constructor not implemented.
  HistogramsAverageElapsedTime(const HistogramsAverageElapsedTime&);
  //! Assignment operator not implemented.
  HistogramsAverageElapsedTime&
  operator=(const HistogramsAverageElapsedTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsAverageElapsedTime
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, recordedSpecies,
         numberOfBins, multiplicity, maxSteps),
    _lastModified(recordedSpecies.size()),
    _modifiedRecordedSpecies()
  {
    // Calculate the recorded species that are modified by each reaction.
    calculateModifiedRecordedSpecies(state, recordedSpecies,
                                     &_modifiedRecordedSpecies);
  }

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
    typedef container::StaticArrayOfArrays<std::size_t>::const_iterator
    const_iterator;

    // Let the process equilibrate.
    Base::equilibrate(equilibrationTime);
    // Step until we have reached the end time.
    const double endTime = _time + recordingTime;
    std::fill(_lastModified.begin(), _lastModified.end(), _time);
    std::size_t reactionIndex;
    const_iterator modified;
    while (true) {
      // Check that we have not exceeded the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        break;
      }

      _tau = Base::computeTau();
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

      // If we have not reached the end time.
      if (_time < endTime) {
        // Determine the reaction to fire.
        reactionIndex = _discreteGenerator();
#ifdef STLIB_DEBUG
        assert(_discreteGenerator[reactionIndex] > 0);
#endif
        // Record the probabilities for each recorded species that is affected
        // by this reaction.
        modified = _modifiedRecordedSpecies(reactionIndex);
        const std::size_t size = _modifiedRecordedSpecies.size(reactionIndex);
        for (std::size_t i = 0; i != size; ++i) {
          const std::size_t recordedIndex = modified[i];
          _histograms(recordedIndex).accumulate
          (_state.getPopulation(_recordedSpecies[recordedIndex]),
           _time - _lastModified[recordedIndex]);
          _lastModified[recordedIndex] = _time;
        }
      }
      else {
        // Record the probabilities for each recorded species.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                    _time - _lastModified[i]);
        }
        // End the simulation.
        return;
      }

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
