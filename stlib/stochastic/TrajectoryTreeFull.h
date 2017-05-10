// -*- C++ -*-

/*!
  \file stochastic/TrajectoryTreeFull.h
  \brief The direct method for SSA.
*/

#if !defined(__stochastic_TrajectoryTreeFull_h__)
#define __stochastic_TrajectoryTreeFull_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/HistogramsPackedArray.h"

#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using a full trajectory tree.
/*!
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class TrajectoryTreeFull : public Solver
{
protected:

  //! The base class.
  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;
  //! The exponential generator.
  typedef _ExponentialGenerator ExponentialGenerator;
  //! The discrete, uniform generator.
  typedef typename ExponentialGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  //
  // Nested classes.
  //
protected:

  //! Holds state in the trajectory tree.
  struct Node {
    //! The reaction to fire next.
    std::size_t reaction;
    //! The time of the next reaction to fire.
    double time;
    //! The probability of the state.
    double probability;
    //! The sum of the propensities.
    double sum;
    //! The error in the sum of the propensities.
    double sumError;
    //! The inverse of the sum of the propensities.
    double inverseSum;
  };

  //
  // Member data.
  //
protected:

  //! A scratch node used to avoid constructor/destructor calls.
  Node _node;
  //! The stack of states.
  std::vector<Node> _stack;
  //! The stack of influenced propensities.
  std::vector<double> _influencedPropensities;
  //! The array of reaction propensities.
  std::vector<double> _propensities;
  //! The propensities functor.
  PropensitiesFunctor _propensitiesFunctor;
  //! The reaction influence graph.
  container::StaticArrayOfArrays<std::size_t> _reactionInfluence;
  //! The discrete uniform random number generator.
  DiscreteUniformGenerator _discreteUniformGenerator;
  //! The exponential random number generator.
  ExponentialGenerator _exponentialGenerator;
  //! The recording times.
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
  TrajectoryTreeFull();
  //! Copy constructor not implemented.
  TrajectoryTreeFull(const TrajectoryTreeFull&);
  //! Assignment operator not implemented.
  TrajectoryTreeFull&
  operator=(const TrajectoryTreeFull&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  TrajectoryTreeFull
  (const State& state, const PropensitiesFunctor& propensitiesFunctor,
   const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
   const std::vector<double>& frames,
   const std::vector<std::size_t>& recordedSpecies,
   const std::size_t numberOfBins, const std::size_t multiplicity,
   const double maxSteps) :
    Base(state, maxSteps),
    // Copy.
    _node(),
    _stack(),
    _influencedPropensities(),
    _propensities(state.getNumberOfReactions(), 0.),
    _propensitiesFunctor(propensitiesFunctor),
    _reactionInfluence(reactionInfluence),
    // Construct.
    _discreteUniformGenerator(),
    _exponentialGenerator(&_discreteUniformGenerator),
    _frames(frames),
    _recordedSpecies(recordedSpecies),
    _histograms(frames.size(), recordedSpecies.size(), numberOfBins,
                multiplicity)
  {
    assert(_frames.size() == 1);
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
    // Initialize the state.
    Base::initialize(populations);
    // Compute the propensities.
    computePropensities();

    // Clear the stacks that hold the state and the influenced propensities.
    _stack.clear();
    _influencedPropensities.clear();
    // Add the first step to the stack.
    _node.reaction = 0;
    _node.time = time;
    _node.probability = 1;
    // Set the sum and error so they will be recomputed.
    _node.sum = 0;
    _node.sumError = 1;
    checkError();
    // Initialize the stack.
    _stack.push_back(_node);
    //! The inverse of the sum of the propensities and the reaction time.
    pickTime();

    // Prepare for recording a trajectory.
    _histograms.initialize();
  }

  //! Generate a full trajectory tree. Record the state in the histogram.
  void
  simulate()
  {
    // Loop until the tree has been generated.
    while (! _stack.empty()) {
      // Check that we have not exceeded the allowed number of steps.
      if (! incrementStepCount()) {
        break;
      }

      // If we have fired all of the reactions.
      if (_stack.back().reaction == _state.getNumberOfReactions()) {
        pop();
        continue;
      }
      // If we reach the end time before the reaction fires.
      if (_stack.back().time >= _frames[0]) {
        // Record the probability for the current state.
        record();
        // Back up to the previous step.
        pop();
        continue;
      }
      // If the current reaction has nonzero probability.
      if (_propensities[_stack.back().reaction] != 0) {
        // Fire the current reaction.
        push();
      }
      else {
        // Move to the next reaction in this step.
        ++_stack.back().reaction;
      }
    }
  }

  //! Synchronize the two sets of histograms so that corresponding historams have the same lower bounds and widths.
  void
  synchronize()
  {
    _histograms.synchronize();
  }

protected:

  //! Record a step count error message. Record the current time.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached. Times = ";
    for (std::size_t i = 0; i != _stack.size(); ++i) {
      out << _stack[i].time << ' ';
    }
    _error += out.str();
  }

  //! Fire a reaction.
  void
  push()
  {
    // Fire the reaction.
    _state.fireReaction(_stack.back().reaction);

    //
    // Make the new node.
    //
    // Start with the first reaction.
    _node.reaction = 0;
    // The probability for the state.
    _node.probability = _stack.back().probability *
                        _propensities[_stack.back().reaction] * _stack.back().inverseSum;
    // Compute the new sum of the propensities and store the influenced
    // reaction propensities.
    _node.sum = _stack.back().sum;
    _node.sumError = _stack.back().sumError;
    for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
         i = _reactionInfluence.begin(_stack.back().reaction);
         i != _reactionInfluence.end(_stack.back().reaction); ++i) {
      // The new reaction propensity.
      const double newPropensity =
        _propensitiesFunctor(*i, _state.getPopulations());
      // Update the sum of the propensities and its error.
      _node.sumError += (_node.sum + newPropensity + _propensities[*i]) *
                        std::numeric_limits<double>::epsilon();
      _node.sum += newPropensity - _propensities[*i];
      // Record the old propensity and set the new one.
      _influencedPropensities.push_back(_propensities[*i]);
      _propensities[*i] = newPropensity;
    }
    // Check the error in the sum of the propensities. Recompute if necessary.
    checkError();
    // Compute the inverse of the sum of the propensities and the time of
    // the next reaction.
    pickTime();

    // Add the new node.
    _stack.push_back(_node);
  }

  //! Un-fire a reaction.
  void
  pop()
  {
    // Back up to the previous step.
    _stack.pop_back();

    // Check that the stack is not empty.
    if (_stack.empty()) {
      return;
    }

    // Un-fire the reaction.
    _state.unFireReaction(_stack.back().reaction);
    // Restore the influenced reaction propensities.
    for (typename container::StaticArrayOfArrays<std::size_t>::
         const_reverse_iterator
         i = _reactionInfluence.rbegin(_stack.back().reaction);
         i != _reactionInfluence.rend(_stack.back().reaction); ++i) {
      _propensities[*i] = _influencedPropensities.back();
      _influencedPropensities.pop_back();
    }

    // Move to the next reaction in the step.
    ++_stack.back().reaction;
  }

  //! Record the current state in the histogram.
  void
  record()
  {
    for (std::size_t i = 0; i != _recordedSpecies.size(); ++i) {
      _histograms(0, i).accumulate(_state.getPopulation(_recordedSpecies[i]),
                                   _stack.back().probability);
    }
  }

  //! Check the error in the sum of the propensities. Recompute if necessary.
  void
  checkError()
  {
    // The allowed relative error is 2^-32.
    const double allowedRelativeError = 2.3283064365386963e-10;
    if (_node.sumError > allowedRelativeError * _node.sum) {
      _node.sum = std::accumulate(_propensities.begin(), _propensities.end(),
                                  0.);
      _node.sumError = _propensities.size() * _node.sum *
                       std::numeric_limits<double>::epsilon();
    }
  }

  //! Pick the time for the next reaction. Set _node.time and _node.inverseSum.
  /*! Offset from the time in the last node in the stack. */
  void
  pickTime()
  {
#ifdef STLIB_DEBUG
    assert(! _stack.empty());
#endif
    if (_node.sum == 0) {
      _node.inverseSum = std::numeric_limits<double>::max();
      _node.time = std::numeric_limits<double>::max();
    }
    else {
      _node.inverseSum = 1. / _node.sum;
      _node.time = _stack.back().time + _exponentialGenerator() *
                   _node.inverseSum;
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return _discreteUniformGenerator;
  }

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
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a reference to the discrete, uniform generator.
  DiscreteUniformGenerator&
  getDiscreteUniformGenerator()
  {
    return _discreteUniformGenerator;
  }

protected:

  void
  computePropensities()
  {
    // Compute each propensity.
    for (std::size_t i = 0; i != _propensities.size(); ++i) {
      _propensities[i] = _propensitiesFunctor(i, _state.getPopulations());
    }
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
