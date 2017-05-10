// -*- C++ -*-

#if !defined(__stochastic_FirstReaction_ipp__)
#error This file is an implementation detail of FirstReaction.
#endif

namespace stlib
{
namespace stochastic
{

//----------------------------------------------------------------------------
// Implementation details.
//----------------------------------------------------------------------------

template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>::
FirstReaction(const State& state,
              const PropensitiesFunctor& propensitiesFunctor,
              const double maxSteps) :
  Base(state, maxSteps),
  // Copy.
  _time(),
  _propensitiesFunctor(propensitiesFunctor),
  // Construct.
  _discreteUniformGenerator(),
  _exponentialGenerator(&_discreteUniformGenerator),
  // Invalid value.
  _timeToFirstReaction(-1),
  _indexOfFirstReaction(-1)
{
}

template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
void
FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>::
initialize(const std::vector<double>& populations, const double time)
{
  // Initialize the state.
  Base::initialize(populations);
  _time = time;

  // Compute the initial propensities and the initial time to the first
  // reaction.
  computeTimeToFirstReaction();
}

// Try to take a step with the first reaction method.
// Return true if a step is taken.
template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
bool
FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>::
step(const double endTime)
{
  // If we have reached the end time.
  if (_time + _timeToFirstReaction >= endTime) {
    return false;
  }

  // Check that we have not exceeded the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Fire the reaction.
  _time += _timeToFirstReaction;
  _state.fireReaction(_indexOfFirstReaction);
  // Compute the time to the next reaction.
  computeTimeToFirstReaction();
  return true;
}


template < class _ExponentialGenerator,
           class _PropensitiesFunctor >
inline
void
FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>::
computeTimeToFirstReaction()
{
  // Start with infinity.
  _timeToFirstReaction = std::numeric_limits<double>::max();
  _indexOfFirstReaction = 0;
  double t, propensity;
  for (std::size_t i = 0; i != _state.getNumberOfReactions(); ++i) {
    propensity = _propensitiesFunctor(i, _state.getPopulations());
    if (propensity != 0) {
      const double mean = 1.0 / propensity;
      _time.updateEpoch(mean);
      t = mean * _exponentialGenerator();
      if (t < _timeToFirstReaction) {
        _timeToFirstReaction = t;
        _indexOfFirstReaction = i;
      }
    }
  }
}

} // namespace stochastic
}
