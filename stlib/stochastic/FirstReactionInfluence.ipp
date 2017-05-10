// -*- C++ -*-

#if !defined(__stochastic_FirstReactionInfluence_ipp__)
#error This file is an implementation detail of FirstReactionInfluence.
#endif

namespace stlib
{
namespace stochastic
{

//----------------------------------------------------------------------------
// Implementation details.
//----------------------------------------------------------------------------

// Try to take a step with the first reaction method.
// Return true if a step is taken.
template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
bool
FirstReactionInfluence<_ExponentialGenerator, _PropensitiesFunctor>::
step(const double endTime)
{
  // If we have reached the end time.
  if (_time + _timeToFirstReaction >= endTime) {
    return false;
  }

  // Check that we have not exceed the allowed number of steps.
  if (! Base::incrementStepCount()) {
    Base::setStepCountError();
    return false;
  }

  // Fire the reaction.
  _time += _timeToFirstReaction;
  _state.fireReaction(_indexOfFirstReaction);
  // Update the propensities and compute the time to the next reaction.
  updatePropensities(_indexOfFirstReaction);
  computeTimeToFirstReaction();
  return true;
}


template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
void
FirstReactionInfluence<_ExponentialGenerator, _PropensitiesFunctor>::
updatePropensities(const std::size_t reactionIndex)
{
  // Update the reaction times for the reactions whose propensities
  // were influenced.
  for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
       i = _reactionInfluence.begin(reactionIndex);
       i != _reactionInfluence.end(reactionIndex); ++i) {
    // Compute the new propensity for the influenced reactions.
    _inversePropensities[*i] =
      safelyInvert(_propensitiesFunctor(*i, _state.getPopulations()));
    _time.updateEpoch(_inversePropensities[*i]);
  }

  // Compute the new propensity for the fired reaction.
  _inversePropensities[reactionIndex] =
    safelyInvert(_propensitiesFunctor(reactionIndex, _state.getPopulations()));
  _time.updateEpoch(_inversePropensities[reactionIndex]);
}

} // namespace stochastic
}
