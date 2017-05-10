// -*- C++ -*-

/*!
  \file stochastic/HistogramsAverageMultiTime.h
  \brief Collect the average values of species populations.
*/

#if !defined(__stochastic_HistogramsAverageMultiTime_h__)
#define __stochastic_HistogramsAverageMultiTime_h__

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
class HistogramsAverageMultiTime :
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
  //! The weight per time trajectory is the inverse of the number of simultaneous trajectories.
  double _weight;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsAverageMultiTime();
  //! Copy constructor not implemented.
  HistogramsAverageMultiTime(const HistogramsAverageMultiTime&);
  //! Assignment operator not implemented.
  HistogramsAverageMultiTime&
  operator=(const HistogramsAverageMultiTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsAverageMultiTime
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const double endTime,
   const double equilibrationTime,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t histogramMultiplicity,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, equilibrationTime,
         recordedSpecies, numberOfBins, histogramMultiplicity, maxSteps),
    _endTime(endTime),
    _multiplicity(multiplicity),
    _times(multiplicity),
    _weight(1. / multiplicity)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations)
  {
    Base::initialize(populations);
  }

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate()
  {
    // Let the process equilibrate.
    Base::equilibrate();
    // Step until we have reached the end time for each branch.
    double tau, tauSum;
    // Set the times to zero.
    _times.resize(_multiplicity);
    std::fill(_times.begin(), _times.end(), 0);
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! incrementStepCount()) {
        setStepCountError();
        break;
      }

      tauSum = 0;
      // The time step.
      if (_discreteGenerator.isValid()) {
        const double inverseSum = 1. / _discreteGenerator.sum();
        for (std::size_t i = 0; i != _times.size(); ++i) {
          tau = _exponentialGenerator() * inverseSum;
          // If we have not passed the end time.
          if (_times[i] + tau < _endTime) {
            // Increment the time.
            _times[i] += tau;
          }
          else {
            // Indicate that this is the last step.
            tau = _endTime - _times[i];
            // Erase this time trajectory.
            std::swap(_times[i], _times.back());
            _times.pop_back();
            --i;
          }
          tauSum += tau;
        }
      }
      else {
        for (std::size_t i = 0; i != _times.size(); ++i) {
          // Indicate that this is the last step.
          tauSum += _endTime - _times[i];
        }
        _times.clear();
      }

      // Record the probabilities for the current state.
      for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
        _histograms(i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                  _weight * tauSum);
      }

      // If all of the dependent branches have reached the end time.
      if (_times.empty()) {
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
