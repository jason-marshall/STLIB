// -*- C++ -*-

/*!
  \file stochastic/HistogramsMultiTimeDirect.h
  \brief The direct method for SSA.
*/

#if !defined(__stochastic_HistogramsMultiTimeDirect_h__)
#define __stochastic_HistogramsMultiTimeDirect_h__

#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/TimeEpochOffset.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate histograms at specified frames using Gillespie's direct method with multiple time trajectories.
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
class HistogramsMultiTimeDirect :
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
  // Nested classes.
  //
protected:

  //! The time and the current frame for a time trajectory.
  struct TimeAndFrame {
    //! The time.
    TimeEpochOffset time;
    //! The first frame after the current time.
    std::size_t frameIndex;
  };

  //
  // Member data.
  //
protected:

  using Base::_state;
  using Base::_error;
  using Base::_exponentialGenerator;
  using Base::_discreteGenerator;
  using Base::_frames;
  using Base::_recordedSpecies;
  using Base::_histograms;

  //! The time multiplicity.
  std::size_t _multiplicity;
  //! The times.
  std::vector<TimeAndFrame> _times;
  //! The weight per time trajectory is the inverse of the number of simultaneous trajectories.
  double _weight;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsMultiTimeDirect();
  //! Copy constructor not implemented.
  HistogramsMultiTimeDirect(const HistogramsMultiTimeDirect&);
  //! Assignment operator not implemented.
  HistogramsMultiTimeDirect&
  operator=(const HistogramsMultiTimeDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsMultiTimeDirect
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t histogramMultiplicity,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, frames,
         recordedSpecies, numberOfBins, histogramMultiplicity, maxSteps),
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
  initialize(const std::vector<double>& populations, const double time)
  {
    Base::initialize(populations, time);

    // Set the starting time.
    for (std::size_t i = 0; i != _times.size(); ++i) {
      _times[i].time = time;
      _times[i].frameIndex = 0;
    }
  }

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate()
  {
    // Step until we have recorded each of the states at each of the frames.
    while (true) {
      // Check that we have not exceed the allowed number of steps.
      if (! Base::incrementStepCount()) {
        setStepCountError();
        break;
      }

      // Compute the times of the next reaction.
      if (_discreteGenerator.isValid()) {
        const double mean = 1.0 / _discreteGenerator.sum();
        for (std::size_t i = 0; i != _times.size(); ++i) {
          _times[i].time.updateEpoch(mean);
          _times[i].time += mean * _exponentialGenerator();
        }
      }
      else {
        for (std::size_t i = 0; i != _times.size(); ++i) {
          _times[i].time = std::numeric_limits<double>::max();
        }
      }

      for (std::size_t i = 0; i != _times.size(); ++i) {
        TimeAndFrame& t = _times[i];
        // For each frame that we will cross with this reaction.
        while (t.frameIndex != _frames.size() &&
               t.time >= _frames[t.frameIndex]) {
          // Record the probabilities for the current state.
          for (std::size_t j = 0; j != _recordedSpecies.size(); ++j) {
            _histograms(t.frameIndex, j).accumulate
            (_state.getPopulation(_recordedSpecies[j]), _weight);
          }
          // Move to the next frame.
          ++t.frameIndex;
        }
        // If we have recorded the state at all of the frames.
        if (t.frameIndex == _frames.size()) {
          // Erase this time trajectory.
          std::swap(t, _times.back());
          _times.pop_back();
          --i;
        }
      }

      // If we have recorded the last frame for all trajectories.
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
    Base::setStepCountError();
    std::ostringstream out;
    out << "The maximum step count " << Base::_maxSteps << " was reached. "
        << " Times = ";
    for (std::size_t i = 0; i != _times.size(); ++i) {
      out << _times[i].time << ' ';
    }
    out << ".";
    _error += out.str();
  }

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

  //! Get the multiplicity of the time trajectories.
  std::size_t
  getMultiplicity() const
  {
    return _multiplicity;
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
