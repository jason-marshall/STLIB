// -*- C++ -*-

/*!
  \file stochastic/HistogramsDirectTree.h
  \brief The direct method for SSA.
*/

#if !defined(__stochastic_HistogramsDirectTree_h__)
#define __stochastic_HistogramsDirectTree_h__

#include "stlib/stochastic/HistogramsBase.h"
#include "stlib/stochastic/TimeEpochOffset.h"

namespace stlib
{
namespace stochastic
{

//! Accumulate histograms at specified frames using Gillespie's direct method with multi-time trajectory trees.
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
class HistogramsDirectTree :
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

  //! The time, weight, and the index of the next splitting time for a time trajectory.
  struct TrajectoryTime {
    //! The time.
    TimeEpochOffset time;
    //! The weight.
    double weight;
    //! The index of the next splitting time.
    std::size_t index;
    //! The rank is used to identify which trajectories are most independent.
    std::size_t rank;
    //! The rank offset is used to calculate new ranks when trajectories are split.
    /*! When a trajectory splits, the rank of the new trajectory is
      rank + rankOffset. Then both rankOffset's are doubled. */
    std::size_t rankOffset;
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

  //! The trajectory times.
  std::vector<TrajectoryTime> _trajectories;
  //! The initial time multiplicity.
  std::size_t _initialMultiplicity;
  //! The total reaction count for all recorded samples.
  /*! Use a floating-point type because this could be very large.*/
  double _numberOfReactions;
  //! The number of recorded samples.
  /*! Use a floating-point type because we use this in a formula with _numberOfReactions.*/
  double _numberOfSamples;
  //! The times at which to split the trajectories.
  std::vector<double> _splittingTimes;
  //! Used for splitting trajectories.
  std::vector<TrajectoryTime> _splitTrajectories;
  //! Used for recording new trajectories before they are added to _trajectories.
  std::vector<TrajectoryTime> _newTrajectories;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HistogramsDirectTree();
  //! Copy constructor not implemented.
  HistogramsDirectTree(const HistogramsDirectTree&);
  //! Assignment operator not implemented.
  HistogramsDirectTree&
  operator=(const HistogramsDirectTree&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HistogramsDirectTree
  (const State& state,
   const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins,
   const std::size_t histogramMultiplicity,
   const std::size_t initialMultiplicity,
   const double maxSteps) :
    Base(state, propensitiesFunctor, reactionInfluence, frames,
         recordedSpecies, numberOfBins, histogramMultiplicity, maxSteps),
    _trajectories(),
    _initialMultiplicity(initialMultiplicity),
    _numberOfReactions(0),
    _numberOfSamples(0),
    _splittingTimes(),
    _splitTrajectories(),
    _newTrajectories()
  {
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations, const double startTime)
  {
    Base::initialize(populations, startTime);

    // CONTINUE
    assert(_frames.size() == 1);

    // Set the trajectory times.
    const double weight = 1. / _initialMultiplicity;
    _trajectories.resize(_initialMultiplicity);
    for (std::size_t i = 0; i != _trajectories.size(); ++i) {
      _trajectories[i].time = startTime;
      _trajectories[i].weight = weight;
      _trajectories[i].index = 0;
      _trajectories[i].rank = i;
      _trajectories[i].rankOffset = _trajectories.size();
    }

    //
    // Set the splitting times.
    //
    // If this is the first trajectory generated and we do not have an estimate
    // for the number of reactions generate about 1,000,000 trajectories.
    double expectedReactions = 1e6;
    if (_numberOfSamples != 0) {
      expectedReactions = _numberOfReactions / _numberOfSamples;
    }
    // The target number of trajectories is the expected number of reactions.
    double numberOfTrajectories = expectedReactions;
    //double numberOfTrajectories = 2 * std::sqrt(expectedReactions);
    std::vector<double> times;
    double n = _initialMultiplicity;
    const double simulationTime = _frames[0] - startTime;
    // Start at four steps away from the frame.
    double difference = 4. * _frames[0] / expectedReactions;
    while (n < numberOfTrajectories && difference < simulationTime) {
      times.push_back(_frames[0] - difference);
      // Conidered in reverse, using a factor of 4 enables one to
      // approximately double the distance between samples at each step.
      difference *= 4;
      n *= 2;
    }
    _splittingTimes.resize(times.size());
    std::copy(times.rbegin(), times.rend(), _splittingTimes.begin());
  }
  /*
    end time = 10,000

    difference *= 8
    1,100
    1.05 seconds
    0.111 from 4 samples

    difference *= 4
    1,000
    1.03 seconds
    0.0716 from 4 samples

    multiplicity = 2
    670
    0.0742 from 10 samples

    multiplicity = 10
    175
    0.0625 from 8 samples

    difference *= 2
    220
    1.04 seconds
    0.0849 from 8 samples

    500
    2.34 seconds
    0.055

    1,000
    4.64 seconds
    0.0298
   */

  //! Generate a trajectory tree and record the state in the histograms.
  void
  simulate()
  {
    std::size_t numberOfSplits;
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
        for (std::size_t i = 0; i != _trajectories.size(); ++i) {
          _trajectories[i].time.updateEpoch(mean);
          _trajectories[i].time += mean * _exponentialGenerator();
        }
      }
      else {
        for (std::size_t i = 0; i != _trajectories.size(); ++i) {
          _trajectories[i].time = std::numeric_limits<double>::max();
        }
      }

      for (std::size_t i = 0; i != _trajectories.size(); ++i) {
        TrajectoryTime& t = _trajectories[i];
        // If we cross the frame.
        if (t.time >= _frames[0]) {
          // Record the probabilities for the current state and erase the
          // trajectory.
          record(&t);
          // Go to the next trajectory.
          --i;
          continue;
        }
        // Count the number of splits and update the weight and index.
        numberOfSplits = countSplits(&t);
        // If the trajectory should be split.
        if (numberOfSplits != 0) {
          split(&t, numberOfSplits);
        }
      }
      // Add the new trajectories obtained through splitting.
      for (std::size_t i = 0; i != _newTrajectories.size(); ++i) {
        _trajectories.push_back(_newTrajectories[i]);
      }
      _newTrajectories.clear();
      // If we have recorded all of the trajectories.
      if (_trajectories.empty()) {
        // End the simulation.
        return;
      }
      // Fire a reaction.
      fire();
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
    for (std::size_t i = 0; i != _trajectories.size(); ++i) {
      out << _trajectories[i].time << ' ';
    }
    out << ".";
    _error += out.str();
  }

  //! Record the state.
  void
  record(TrajectoryTime* t)
  {
    for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
      _histograms(0, i).accumulate
      (_state.getPopulation(_recordedSpecies[i]), t->weight);
    }
    _numberOfReactions += _state.getReactionCount();
    ++_numberOfSamples;
    // Erase this time trajectory.
    *t = _trajectories.back();
    _trajectories.pop_back();
  }

  //! Count the number of splits and update the weight and index.
  std::size_t
  countSplits(TrajectoryTime* t)
  {
    std::size_t numberOfSplits = 0;
    while (t->index != _splittingTimes.size() &&
           t->time >= _splittingTimes[t->index]) {
      ++numberOfSplits;
      t->weight *= 0.5;
      ++t->index;
    }
    return numberOfSplits;
  }

  //! Split the trajectory.
  void
  split(TrajectoryTime* t, std::size_t numberOfSplits)
  {
    // CONTINUE: See if handling numberOfSplits == 1 as a special case improves
    // performance.
    if (numberOfSplits == 1) {
      _newTrajectories.push_back(*t);
      _newTrajectories.back().rank += t->rankOffset;
      t->rankOffset *= 2;
      _newTrajectories.back().rankOffset = t->rankOffset;
      return;
    }
    // Copy the original trajectory.
    _splitTrajectories.push_back(*t);
    // For each time that we will split the trajectories.
    for (; numberOfSplits != 0; --numberOfSplits) {
      // Split each trajectory.
      std::size_t size = _splitTrajectories.size();
      for (std::size_t i = 0; i != size; ++i) {
        // Copy the trajectory.
        _splitTrajectories.push_back(_splitTrajectories[i]);
        // Compute the new rank for the second half.
        _splitTrajectories.back().rank += t->rankOffset;
        // The rank offsets are doubled.
        _splitTrajectories[i].rankOffset *= 2;
        _splitTrajectories.back().rankOffset *= 2;
      }
      t->rankOffset *= 2;
    }
    // Record the split trajectories. (The original is still in the list.)
    for (std::size_t i = 1; i != _splitTrajectories.size(); ++i) {
      _newTrajectories.push_back(_splitTrajectories[i]);
    }
    _splitTrajectories.clear();
  }

  //! Fire a reaction.
  void
  fire()
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
