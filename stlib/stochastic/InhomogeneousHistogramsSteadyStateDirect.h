// -*- C++ -*-

/*!
  \file stochastic/InhomogeneousHistogramsSteadyStateDirect.h
  \brief Accumulate histograms at specified frames using Gillespie's direct method.
*/

#if !defined(__stochastic_InhomogeneousHistogramsSteadyStateDirect_h__)
#define __stochastic_InhomogeneousHistogramsSteadyStateDirect_h__

#include "stlib/stochastic/InhomogeneousDirect.h"
#include "stlib/stochastic/HistogramsAveragePackedArray.h"
#include "stlib/stochastic/TimeEpochOffset.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate histograms at specified frames using Gillespie's direct method.
class InhomogeneousHistogramsSteadyStateDirect :
  public InhomogeneousDirect
{
  //
  // Private types.
  //
private:

  typedef InhomogeneousDirect Base;

  //
  // Member data.
  //
private:

  //! The species to record.
  std::vector<std::size_t> _recordedSpecies;
  //! Histograms for the recorded species.
  HistogramsAveragePackedArray _histograms;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  InhomogeneousHistogramsSteadyStateDirect();
  //! Copy constructor not implemented.
  InhomogeneousHistogramsSteadyStateDirect
  (const InhomogeneousHistogramsSteadyStateDirect&);
  //! Assignment operator not implemented.
  InhomogeneousHistogramsSteadyStateDirect&
  operator=(const InhomogeneousHistogramsSteadyStateDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  InhomogeneousHistogramsSteadyStateDirect
  (const State& state,
   const ReactionSet& reactionSet,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, reactionSet, maxSteps),
    _recordedSpecies(recordedSpecies),
    _histograms(recordedSpecies.size(), numberOfBins, multiplicity)
  {
  }

  // Default destructor is fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  /*!
    Override the Base::initialize().
  */
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    // Initialize the state.
    Base::initialize(populations, time);
    // Prepare for recording a trajectory.
    _histograms.initialize();
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    _histograms.synchronize();
  }

  //! Generate a trajectory and record the state in the histograms.
  void
  simulate(const double equilibrationTime, const double recordingTime)
  {
    // Let the process equilibrate.
    double endTime = _time + equilibrationTime;
    Base::simulate(endTime);
    // The next (unfired) reaction will carry the simulation past the end time.
    _time = endTime;

    // Step until we have reached the end time.
    endTime = _time + recordingTime;
    while (true) {
      // Check that we have not exceeded the allowed number of steps.
      if (! incrementStepCount()) {
        setStepCountError();
        break;
      }

      // Recompute the propensities for the current populations and time.
      computePropensities();
      _tau = computeTau();
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

      // Record the probabilities for the current state.
      for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
        _histograms(i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                  _tau);
      }

      // If we have reached the end time.
      if (_time >= endTime) {
        // End the simulation.
        return;
      }

      // Recompute the propensities for the new time.
      computePropensities();
      // Fire a reaction if possible.
      if (_propensitiesFunctor.sum() > 0) {
        _state.fireReaction(generateDiscreteDeviate());
      }
    }
  }

protected:

  //! Record a step count error message. Record the current time.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    Base::setStepCountError();
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached. "
        << " Time = " << _time << ".";
    _error += out.str();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the vector of recorded species.
  const std::vector<std::size_t>&
  getRecordedSpecies() const
  {
    return _recordedSpecies;
  }

  //! Return the set of histograms.
  const HistogramsAveragePackedArray&
  getHistograms() const
  {
    return _histograms;
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
