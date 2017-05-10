// -*- C++ -*-

#if !defined(__stochastic_Direct_ipp__)
#error This file is an implementation detail of Direct.
#endif

namespace stlib
{
namespace stochastic
{

// Try to take a step with the direct method.
// Return true if a step is taken.
template < class _DiscreteGenerator,
           class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
bool
Direct < _DiscreteGenerator,
       _ExponentialGenerator,
       _PropensitiesFunctor >::
       step(const double endTime)
{
  // If we have reached the end time.
  if (_time + _tau >= endTime) {
    return false;
  }

  // Check that we have not exceeded the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Determine the reaction to fire.
  const std::size_t reactionIndex = _discreteGenerator();
#ifdef STLIB_DEBUG
  assert(_discreteGenerator[reactionIndex] > 0);
#endif
  // Fire the reaction.
  _time += _tau;
  _state.fireReaction(reactionIndex);
  // Recompute the propensities and update the discrete, finite generator.
  updatePropensities(reactionIndex);
  // Compute the next time step.
  _tau = computeTau();
  return true;
}

// Try to take a step with the direct method.
// Return true if a step is taken.
template < class _DiscreteGenerator,
           class _ExponentialGenerator,
           class _PropensitiesFunctor >
template<typename _IntOutputIter, typename NumberOutputIter>
inline
bool
Direct < _DiscreteGenerator,
       _ExponentialGenerator,
       _PropensitiesFunctor >::
       step(const double endTime, _IntOutputIter indices, NumberOutputIter times)
{
  // If we have reached the end time.
  if (_time + _tau >= endTime) {
    return false;
  }

  // Check that we have not exceed the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Determine the reaction to fire.
  const std::size_t reactionIndex = _discreteGenerator();
#ifdef STLIB_DEBUG
  assert(_discreteGenerator[reactionIndex] > 0);
#endif
  // Fire the reaction.
  _time += _tau;
  _state.fireReaction(reactionIndex);
  // Record the reaction index and time.
  *indices++ = reactionIndex;
  *times++ = _time;
  // Recompute the propensities and update the discrete, finite generator.
  updatePropensities(reactionIndex);
  // Compute the next time step.
  _tau = computeTau();
  return true;
}

} // namespace stochastic
}
