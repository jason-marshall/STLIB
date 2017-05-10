// -*- C++ -*-

/*!
  \file stochastic/NextReaction.h
  \brief The next reaction method for SSA.
*/

#if !defined(__stochastic_NextReaction_h__)
#define __stochastic_NextReaction_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/TimeEpochOffset.h"

#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using Gibson and Bruck's next reaction method.
/*!
  \param _ReactionPriorityQueue The priority queue for the reactions.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _ReactionPriorityQueue,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class NextReaction : public Solver
{
protected:

  //! The base class.
  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The reaction priority queue.
  typedef _ReactionPriorityQueue ReactionPriorityQueue;
  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;

  //! The discrete uniform generator.
  typedef typename ReactionPriorityQueue::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  //
  // Enumerations.
  //
private:

  enum {ComputeIndividualPropensities =
        std::is_same<typename PropensitiesFunctor::result_type,
        double>::value
       };

  //
  // Member data.
  //
private:

  TimeEpochOffset _time;
  PropensitiesFunctor _propensitiesFunctor;
  // Store a pointer here because there are many options for this class.
  // They require different constructor calls.
  ReactionPriorityQueue* _reactionPriorityQueue;
  container::StaticArrayOfArrays<std::size_t> _reactionInfluence;
  std::vector<double> _propensities;
  std::vector<double> _oldPropensities;
  std::size_t _reactionIndex;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  NextReaction();
  //! Copy constructor not implemented.
  NextReaction(const NextReaction&);
  //! Assignment operator not implemented.
  NextReaction&
  operator=(const NextReaction&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  NextReaction(const State& state,
               const PropensitiesFunctor& propensitiesFunctor,
               ReactionPriorityQueue* reactionPriorityQueue,
               const container::StaticArrayOfArrays<std::size_t>& reactionInfluence,
               const double maxSteps) :
    Base(state, maxSteps),
    // Copy.
    _time(),
    _propensitiesFunctor(propensitiesFunctor),
    _reactionPriorityQueue(reactionPriorityQueue),
    _reactionInfluence(reactionInfluence),
    // Allocate.
    _propensities(state.getNumberOfReactions()),
    _oldPropensities(ComputeIndividualPropensities ? 0 : _propensities.size()),
    // Invalid index.
    _reactionIndex(state.getNumberOfReactions())
  {
    // Set the pointer to the propensities array if necessary.
    setPropensities(std::integral_constant<bool,
                    ReactionPriorityQueue::UsesPropensities>());
  }

private:

  // The indexed priority queue does not use the propensities.
  void
  setPropensities(std::false_type /*UsesPropensities*/)
  {
  }

  // The indexed priority queue uses the propensities.
  void
  setPropensities(std::true_type /*UsesPropensities*/)
  {
    _reactionPriorityQueue->setPropensities(&_propensities);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and reset the time.
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    // Initialize the state.
    Base::initialize(populations);
    _time = time;

    _reactionIndex = _state.getNumberOfReactions();
    // Compute the initial propensities.
    computePropensities();

    // Initialize the reaction priority queue.
    _reactionPriorityQueue->clear();
    for (std::size_t i = 0; i != _propensities.size(); ++i) {
      if (_propensities[i] != 0) {
        _reactionPriorityQueue->push(i, _time.getOffset(), _propensities[i]);
      }
    }
  }

  //! Simulate until the end time is reached.
  void
  simulate(const double endTime)
  {
    // Step until no more reactions can fire or we reach the end time.
    while (step(endTime)) {
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  bool
  step(const double endTime);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the current time.
  double
  getTime() const
  {
    // Convert the time epoch and offset to a single time.
    return _time;
  }

  //! Return the array of propensities.
  const std::vector<double>&
  getPropensities() const
  {
    return _propensities;
  }

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return _reactionPriorityQueue->getDiscreteUniformGenerator();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
private:

  void
  computePropensities()
  {
    _computePropensities(std::integral_constant<bool,
                         ComputeIndividualPropensities>());
  }

  void
  _computePropensities(std::false_type /*dummy*/)
  {
    _propensitiesFunctor(_state.getPopulations(), _propensities.begin());
  }

  void
  _computePropensities(std::true_type /*dummy*/)
  {
    // Compute each propensity.
    for (std::size_t i = 0; i != _propensities.size(); ++i) {
      _propensities[i] = _propensitiesFunctor(i, _state.getPopulations());
    }
  }

  void
  updatePropensitiesAndReactionTimes(const std::size_t reactionIndex)
  {
    _updatePropensitiesAndReactionTimes
    (std::integral_constant<bool, ComputeIndividualPropensities>(),
     reactionIndex);
  }

  void
  _updatePropensitiesAndReactionTimes
  (std::false_type /*ComputeIndividualPropensities*/,
   const std::size_t reactionIndex);

  void
  _updatePropensitiesAndReactionTimes
  (std::true_type /*ComputeIndividualPropensities*/,
   const std::size_t reactionIndex);

  //@}
};

//@}

} // namespace stochastic
}

#define __stochastic_NextReaction_ipp__
#include "stlib/stochastic/NextReaction.ipp"
#undef __stochastic_NextReaction_ipp__

#endif
