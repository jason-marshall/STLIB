// -*- C++ -*-

/*!
  \file stochastic/HistogramsAverageElapsedMultiTime.h
  \brief Collect the average values of species populations.
*/

#if !defined(__stochastic_HistogramsAverageElapsedMultiTime_h__)
#define __stochastic_HistogramsAverageElapsedMultiTime_h__

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
class HistogramsAverageElapsedMultiTime :
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
  using Base::_error;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;
  using Base::_recordedSpecies;
  using Base::_histograms;

  //! The simulation end time.
  double _endTime;
  //! The time multiplicity.
  std::size_t _multiplicity;
  //! The times for each branch.
  std::vector<double> _times;
  //! The sum of the times at which each species was last modified.
  std::vector<double> _lastModifiedSums;
  //! The recorded species that are modified by each reaction.
  container::StaticArrayOfArrays<std::size_t> _modifiedRecordedSpecies;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAverageElapsedMultiTime();
  //! Copy constructor not implemented.
  HistogramsAverageElapsedMultiTime(const HistogramsAverageElapsedMultiTime&);
  //! Assignment operator not implemented.
  HistogramsAverageElapsedMultiTime&
  operator=(const HistogramsAverageElapsedMultiTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsAverageElapsedMultiTime
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t histogramMultiplicity,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, recordedSpecies,
         numberOfBins, histogramMultiplicity, maxSteps),
    _multiplicity(multiplicity),
    _times(multiplicity),
    _lastModifiedSums(recordedSpecies.size()),
    _modifiedRecordedSpecies()
  {
    // Calculate the recorded species that are modified by each reaction.
    calculateModifiedRecordedSpecies(state, recordedSpecies,
                                     &_modifiedRecordedSpecies);
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
    typedef container::StaticArrayOfArrays<std::size_t>::const_iterator
    const_iterator;

    // Let the process equilibrate.
    Base::equilibrate(equilibrationTime);

    // Step until we have reached the end time for each branch.
    std::size_t reactionIndex;
    const_iterator modified;
    // Set the times to zero.
    _times.resize(_multiplicity);
    std::fill(_times.begin(), _times.end(), _time);
    std::fill(_lastModifiedSums.begin(), _lastModifiedSums.end(), 0);
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! incrementStepCount()) {
        setStepCountError();
        break;
      }

      // The time step.
      if (_discreteGenerator.isValid()) {
        const double mean = 1.0 / _discreteGenerator.sum();
        _time.updateEpoch(mean);
        for (std::size_t i = 0; i != _times.size(); ++i) {
          _tau = _exponentialGenerator() * mean;
          // If we have not passed the end time.
          if (_times[i] + _tau < _endTime) {
            // Increment the time.
            _times[i] += _tau;
          }
          else {
            // Indicate that this is the last step.
            _times[i] = _endTime;
          }
        }
      }
      else {
        // Indicate that this is the last step.
        std::fill(_times.begin(), _times.end(), _endTime);
      }

      const double sumOfTimes =
        std::accumulate(_times.begin(), _times.end(), 0.);
      // If not all of the dependent branches have reached the end time.
      if (std::size_t(std::count(_times.begin(), _times.end(), _endTime)) !=
          _times.size()) {
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
           sumOfTimes - _lastModifiedSums[recordedIndex]);
          _lastModifiedSums[recordedIndex] = sumOfTimes;
        }
      }
      else {
        // Record the probabilities for each recorded species.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                    sumOfTimes - _lastModifiedSums[i]);
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

  //! Record a step count error message. Append the current times to the error message.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached. "
        << " Times = ";
    for (std::size_t i = 0; i != _times.size(); ++i) {
      out << _times[i] << ' ';
    }
    out << " Time = " << _time << ".";
    _error += out.str();
  }

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
