// -*- C++ -*-

#if !defined(__stochastic_TauLeapingSal_ipp__)
#error This file is an implementation detail of TauLeapingSal.
#endif

namespace stlib
{
namespace stochastic
{

// Constructor.
inline
TauLeapingSal::
TauLeapingSal(const State& state,
              const PropensitiesFunctor& propensitiesFunctor,
              const double maxSteps) :
  Base(state, maxSteps),
  _propensitiesFunctor(propensitiesFunctor),
  _propensityTimeDerivatives(state.getNumberOfSpecies()),
  // Invalid value.
  _time(std::numeric_limits<double>::max()),
  _propensities(state.getNumberOfReactions()),
  _dpdt(state.getNumberOfReactions()),
  _rateConstants(state.getNumberOfReactions()),
  _reactionFirings(state.getNumberOfReactions()),
  // Construct.
  _discreteUniformGenerator(),
  _normalGenerator(&_discreteUniformGenerator),
  // CONTINUE: normal threshhold.
  _poissonGenerator(&_normalGenerator, 1000)
{
  // Store the rate constants for each reaction. This is used in computing
  // the time step.
  for (std::size_t i = 0; i != _rateConstants.size(); ++i) {
    _rateConstants[i] = _propensitiesFunctor.getReaction(i).
                        computeRateConstant();
  }
}


inline
void
TauLeapingSal::
initialize(const std::vector<double>& populations, const double time)
{
  // Initialize the state.
  Base::initialize(populations);
  _time = time;
}


// Try to take a step.  Return true if a step is taken.
inline
bool
TauLeapingSal::
step(MemberFunctionPointer method, const double epsilon, const double endTime)
{
  // If we have reached the end time.
  if (_time >= endTime) {
    return false;
  }

  // Check that we have not exceeded the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Compute the propensities. We will use these in computing tau and in
  // firing reactions.
  computePropensities();

  // Compute the time leap.
  double tau = computeTau(epsilon);
  // If the time leap will take us past the end time.
  if (_time + tau > endTime) {
    tau = endTime - _time;
    _time = endTime;
  }
  else {
    _time += tau;
  }

  // Fire the reactions.
  (this->*method)(tau);

  // If there are negative populations.
  if (! _state.isValid()) {
    // Correct negative populations and reduce the time step if necessary.
    fixNegativePopulations(tau);
  }

  // No errors if we reached here.
  return true;
}


// Try to take a step.  Return true if a step is taken.
inline
bool
TauLeapingSal::
stepFixed(MemberFunctionPointer method, double tau, const double endTime)
{
  // If we have reached the end time.
  if (_time >= endTime) {
    return false;
  }

  // Check that we have not exceed the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Compute the propensities.
  computePropensities();

  // If the time leap will take us past the end time.
  if (_time + tau > endTime) {
    tau = endTime - _time;
    _time = endTime;
  }
  else {
    _time += tau;
  }

  // Fire the reactions.
  (this->*method)(tau);

  // If there are negative populations.
  if (! _state.isValid()) {
    // Correct negative populations and reduce the time step if necessary.
    fixNegativePopulations(tau);
  }

  // No errors if we reached here.
  return true;
}


inline
void
TauLeapingSal::
fixNegativePopulations(const double tau)
{
#ifdef STLIB_DEBUG
  assert(! _state.isValid());
#endif

  // CONTINUE: Write a solution that will work when there are fast
  // reactions.
  // If there were many reaction events during the step, we simply
  // set the negative populations to zero.
  if (ext::max(_reactionFirings) > 1e3) {
    _state.fixNegativePopulations();
    return;
  }

  // First undo the current step.
  _time -= tau;
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _state.fireReaction(m, -_reactionFirings[m]);
  }

  // Convert the reaction firing counts to std::size_t.
  std::vector<std::size_t> reactionCounts(_reactionFirings.size());
  for (std::size_t i = 0; i != reactionCounts.size(); ++i) {
    reactionCounts[i] = std::size_t(_reactionFirings[i]);
  }

  const std::size_t numFirings = ext::sum(reactionCounts);

  // Loop until a population becomes negative.
  std::size_t i = 0;
  for (; i != numFirings; ++i) {
    // Pick a reaction channel j.
    std::size_t count =
      std::size_t
      ((numFirings - i) *
       numerical::transformDiscreteDeviateToContinuousDeviateOpen<double>
       (_discreteUniformGenerator()));
    std::size_t j = 0;
    for (; j != reactionCounts.size(); ++j) {
      if (reactionCounts[j] > count) {
        break;
      }
      count -= reactionCounts[j];
    }
    assert(j != reactionCounts.size());

    // Fire the reaction.
    _state.fireReaction(j);

    // If a population became negative.
    if (! _state.isValid()) {
      // Unfire the reaction and break out of the loop.
      _state.unFireReaction(j);
      break;
    }
  }

  // Advance the appropriate fraction of the time step.
  _time += tau * double(i) / double(numFirings);
}


inline
void
TauLeapingSal::
stepLinear(const double tau)
{
  const double ht2 = 0.5 * tau * tau;
  // Fire the reactions.
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _reactionFirings[m] = _poissonGenerator(_propensities[m] * tau +
                                            _dpdt[m] * ht2);
    _state.fireReaction(m, _reactionFirings[m]);
  }
}


inline
void
TauLeapingSal::
computePropensities(const std::vector<double>& populations)
{
  // The propensities.
  for (std::size_t m = 0; m < _propensities.size(); ++m) {
    _propensities[m] = _propensitiesFunctor(m, populations);
  }
  // The time derivatives of the propensities.
  _propensityTimeDerivatives(_state, _propensitiesFunctor, _propensities,
                             &_dpdt);
}


inline
double
TauLeapingSal::
computeTau(const double epsilon)
{
#ifdef STLIB_DEBUG
  assert(epsilon > 0);
#endif

  // Take the minimum over the reactions.
  double tau = std::numeric_limits<double>::max();
  double t;
  for (std::size_t i = 0; i != _propensities.size(); ++i) {
    t = epsilon * std::max(_propensities[i], _rateConstants[i]) /
        std::abs(_dpdt[i]);
    if (t < tau) {
      tau = t;
    }
  }
  return tau;
}


} // namespace stochastic
}
