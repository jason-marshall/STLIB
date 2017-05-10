// -*- C++ -*-

#if !defined(__stochastic_FirstReactionAbsoluteTime_ipp__)
#error This file is an implementation detail of FirstReactionAbsoluteTime.
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
FirstReactionAbsoluteTime<_ExponentialGenerator, _PropensitiesFunctor>::
step(const double endTime)
{
  // If we have reached the end time.
  if (_time.getEpoch() + _timeOfFirstReaction >= endTime) {
    return false;
  }

  // Check that we have not exceed the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Fire the reaction.
  _time.setOffset(_timeOfFirstReaction);
  _state.fireReaction(_indexOfFirstReaction);
  // Update the propensities and compute the time to the next reaction.
  updateReactionTimes(_indexOfFirstReaction);
  computeTimeOfFirstReaction();
  return true;
}

template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
void
FirstReactionAbsoluteTime<_ExponentialGenerator, _PropensitiesFunctor>::
updateReactionTimes(const std::size_t reactionIndex)
{
  // Update the reaction times for the reactions whose propensities
  // were influenced.
  for (typename container::StaticArrayOfArrays<std::size_t>::const_iterator
       i = _reactionInfluence.begin(reactionIndex);
       i != _reactionInfluence.end(reactionIndex); ++i) {
    const double p = _propensitiesFunctor(*i, _state.getPopulations());
    if (p != 0) {
      _reactionTimes[*i] = _time.getOffset() + _exponentialGenerator() / p;
    }
    else {
      _reactionTimes[*i] = std::numeric_limits<double>::max();
    }
  }

  // Compute the new reaction time for the fired reaction.
  const double p = _propensitiesFunctor(reactionIndex, _state.getPopulations());
  if (p != 0) {
    const double mean = 1. / p;
    if (_time.shouldStartNewEpoch(mean)) {
      // Start a new time epoch.
      _reactionTimes -= _time.getOffset();
      _time.startNewEpoch();
    }
    _reactionTimes[reactionIndex] = _time.getOffset() +
                                    mean * _exponentialGenerator();
  }
  else {
    _reactionTimes[reactionIndex] = std::numeric_limits<double>::max();
  }
}

} // namespace stochastic
}
