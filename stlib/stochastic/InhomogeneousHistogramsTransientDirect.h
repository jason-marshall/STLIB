// -*- C++ -*-

/*!
  \file stochastic/InhomogeneousHistogramsTransientDirect.h
  \brief Accumulate histograms at specified frames using Gillespie's direct method.
*/

#if !defined(__stochastic_InhomogeneousHistogramsTransientDirect_h__)
#define __stochastic_InhomogeneousHistogramsTransientDirect_h__

#include "stlib/stochastic/InhomogeneousDirect.h"
#include "stlib/stochastic/HistogramsPackedArray.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate histograms at specified frames using Gillespie's direct method.
class InhomogeneousHistogramsTransientDirect :
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

  //! The times at which to accumulate histograms of the state.
  std::vector<double> _frames;
  //! The species to record.
  std::vector<std::size_t> _recordedSpecies;
  //! Histograms for the recorded species.
  HistogramsPackedArray _histograms;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  InhomogeneousHistogramsTransientDirect();
  //! Copy constructor not implemented.
  InhomogeneousHistogramsTransientDirect
  (const InhomogeneousHistogramsTransientDirect&);
  //! Assignment operator not implemented.
  InhomogeneousHistogramsTransientDirect&
  operator=(const InhomogeneousHistogramsTransientDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  InhomogeneousHistogramsTransientDirect
  (const State& state,
   const ReactionSet& reactionSet,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins, const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, reactionSet, maxSteps),
    _frames(frames),
    _recordedSpecies(recordedSpecies),
    _histograms(frames.size(), recordedSpecies.size(), numberOfBins,
                multiplicity)
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
  simulate()
  {
    std::size_t frameIndex = 0;
    // Step until we have recorded the state at each of the frames.
    while (true) {
      // Check that we have not exceeded the allowed number of steps.
      if (! incrementStepCount()) {
        setStepCountError();
        break;
      }

      // Recompute the propensities for the current populations and time.
      computePropensities();
      // Compute the time of the next reaction.
      if (_propensitiesFunctor.sum() <= 0) {
        _time = std::numeric_limits<double>::max();
      }
      else {
        const double mean = 1.0 / _propensitiesFunctor.sum();
        _time.updateEpoch(mean);
        _time += mean * _exponentialGenerator();
      }

      // For each frame that we will cross with this reaction.
      while (_time >= _frames[frameIndex]) {
        // Record the probabilities for the current state.
        for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
          _histograms(frameIndex, i).accumulate
          (_state.getPopulation(_recordedSpecies[i]), 1.);
        }
        // Move to the next frame.
        ++frameIndex;
        // If we have recorded the state at all of the frames.
        if (frameIndex == _frames.size()) {
          // End the simulation.
          return;
        }
      }

      // Recompute the propensities for the new time. The discrete deviate
      // depends on the propensities at the end of the time step, not at
      // the beginnng.
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
  const HistogramsPackedArray&
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
