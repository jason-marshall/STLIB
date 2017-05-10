// -*- C++ -*-

#if !defined(__stochastic_NextReaction_ipp__)
#error This file is an implementation detail of NextReaction.
#endif

namespace stlib
{
namespace stochastic
{

//----------------------------------------------------------------------------
// Implementation details.
//----------------------------------------------------------------------------

// Try to take a step with the next reaction SSA method.
// Return true if a step is taken.
template < class _ReactionPriorityQueue,
           class _PropensitiesFunctor >
inline
bool
NextReaction<_ReactionPriorityQueue, _PropensitiesFunctor>::
step(const double endTime)
{
  // The index of the reaction that will fire next.
  if (_reactionIndex == _state.getNumberOfReactions()) {
    _reactionIndex = _reactionPriorityQueue->top();
  }
  // The time offset in the current epoch at which the next reaction occurs.
  const double reactionTime = _reactionPriorityQueue->get(_reactionIndex);

  // If we have reached the end time.
  if (_time.getEpoch() + reactionTime >= endTime) {
    return false;
  }

  // Check that we have not exceeded the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Fire the reaction.
  _time.setOffset(reactionTime);
  _state.fireReaction(_reactionIndex);

  updatePropensitiesAndReactionTimes(_reactionIndex);
  _reactionIndex = _state.getNumberOfReactions();

  return true;
}

template < class _ReactionPriorityQueue,
           class _PropensitiesFunctor >
inline
void
NextReaction<_ReactionPriorityQueue, _PropensitiesFunctor>::
_updatePropensitiesAndReactionTimes
(std::false_type /*ComputeIndividualPropensities*/,
 const std::size_t reactionIndex)
{
  // Recompute the propensities.
  _propensities.swap(_oldPropensities);
  computePropensities();

  if (_propensities[reactionIndex] != 0) {
    const double mean = 1. / _propensities[reactionIndex];
    // If we will start losing random bits by adding to the time offset.
    if (_time.shouldStartNewEpoch(mean)) {
      // Start a new time epoch.
      _reactionPriorityQueue->startNewEpoch(_time.getOffset());
      _time.startNewEpoch();
    }
    // Compute a new reaction time for the fired reaction.
    _reactionPriorityQueue->pushTopInverse(_time.getOffset(), mean);
  }
  else {
    // Remove the reaction from the queue.
    _reactionPriorityQueue->popTop();
  }

  // Update the reaction times for the reactions whose propensities
  // were influenced.
  for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
       i = _reactionInfluence.begin(reactionIndex);
       i != _reactionInfluence.end(reactionIndex); ++i) {
    // Update the reaction time.
    _reactionPriorityQueue->update(*i, _time.getOffset(),
                                   _oldPropensities[*i], _propensities[*i]);
  }
}


template < class _ReactionPriorityQueue,
           class _PropensitiesFunctor >
inline
void
NextReaction<_ReactionPriorityQueue, _PropensitiesFunctor>::
_updatePropensitiesAndReactionTimes
(std::true_type /*ComputeIndividualPropensities*/,
 const std::size_t reactionIndex)
{
  // Compute the new propensity for the fired reaction.
  _propensities[reactionIndex] =
    _propensitiesFunctor(reactionIndex, _state.getPopulations());

  if (_propensities[reactionIndex] != 0) {
    const double mean = 1. / _propensities[reactionIndex];
    // If we will start losing random bits by adding to the time offset.
    if (_time.shouldStartNewEpoch(mean)) {
      // Start a new time epoch.
      _reactionPriorityQueue->startNewEpoch(_time.getOffset());
      _time.startNewEpoch();
    }
    // Compute a new reaction time for the fired reaction.
    _reactionPriorityQueue->pushTopInverse(_time.getOffset(), mean);
  }
  else {
    // Remove the reaction from the queue.
    _reactionPriorityQueue->popTop();
  }

  // Update the reaction times for the reactions whose propensities
  // were influenced.
  for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
       i = _reactionInfluence.begin(reactionIndex);
       i != _reactionInfluence.end(reactionIndex); ++i) {
    // Compute the new propensity for the influenced reaction.
    const double newPropensity =
      _propensitiesFunctor(*i, _state.getPopulations());
    // Update the reaction time.
    _reactionPriorityQueue->update(*i, _time.getOffset(),
                                   _propensities[*i], newPropensity);
    // Update the propensity.
    _propensities[*i] = newPropensity;
  }
}

} // namespace stochastic
}
