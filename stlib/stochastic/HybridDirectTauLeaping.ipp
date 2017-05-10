// -*- C++ -*-

#if !defined(__stochastic_HybridDirectTauLeaping_ipp__)
#error This file is an implementation detail of HybridDirectTauLeaping.
#endif

namespace stlib
{
namespace stochastic
{

// Initialize the state with the initial populations and reset the time.
template<class _PropensitiesFunctor>
inline
void
HybridDirectTauLeaping<_PropensitiesFunctor>::
initialize(const std::vector<double>& populations, const double time)
{
  // Initialize the state.
  Base::initialize(populations);
  _time = time;
  _directStepCount = 0;
  _tauLeapingStepCount = 0;

  // Initially put all of the reactions in the slow group.
  _discreteGenerator.clear();
  _tauLeaping.clear();
  for (std::size_t i = 0; i != _propensities.size(); ++i) {
    _discreteGenerator.insert(i);
  }

  // Compute the propensities.
  computePropensities();
  // Compute the initial exponential deviate.
  _exponentialDeviate = _exponentialGenerator();
}

// CONTINUE: Can I avoid computing certain propensities?
// Try to take a step with the direct/tau-leaping method.
// Return true if a step is taken.
template<class _PropensitiesFunctor>
inline
bool
HybridDirectTauLeaping<_PropensitiesFunctor>::
step(MemberFunctionPointer method, const double endTime)
{
  const double minimumReactions = 0.1;

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
  const double originalTau =
    _tauLeaping.computeStep(_state.getStateChangeVectors(), _propensities,
                            _state.getPopulations());
  double tau = originalTau;

  // Move the volatile reactions to the direct group.
  moveVolatileAndSlowToDirect(minimumReactions / tau);

  const double directPmfSum = computeDirectPmfSum();

  bool shouldTakeDirectStep = false;
  // If the tau-leaping step is as large as the direct step. (The direct step
  // is _exponentialDeviate / directPmfSum.)
  if (directPmfSum != 0 && (tau == std::numeric_limits<double>::max() ||
                            directPmfSum * tau >= _exponentialDeviate)) {
    shouldTakeDirectStep = true;
    // The tau-leaping step is the direct step.
    tau = _exponentialDeviate / directPmfSum;
  }

  // Determine the new time.
  bool areFinished = false;
  // If the time leap will take us past the end time.
  if (_time + tau > endTime) {
    tau = endTime - _time;
    areFinished = true;
    shouldTakeDirectStep = false;
  }

  // CONTINUE
#if 0
  print(std::cout);
  std::cout << "Original tau = " << originalTau << '\n'
            << "Direct PMF sum = " << directPmfSum << '\n'
            << "Should take direct step = " << shouldTakeDirectStep << '\n';
#endif

  // Advance the state by firing the tau-leaping reactions.
  if (!(this->*method)(tau)) {
    // CONTINUE
    //print(std::cout);
    return false;
  }

  // Add the propensities contribution for this time step to the discrete
  // generator.
  _discreteGenerator.addPmf(_propensities, tau);

  // If a slow reaction fires at the end of the step.
  if (shouldTakeDirectStep) {
    // Determine the reaction to fire.
    const std::size_t reactionIndex = _discreteGenerator();
    // Fire the reaction.
    _state.fireReaction(reactionIndex);
    // If the fired reaction is not volatile and is fast, move it to the
    // tau-leaping group.
    // CONTINUE
    if (! isVolatile(reactionIndex)) {
      // If there are no reactions in the tau-leaping group and this is the
      // fastest reaction.
      // If the reaction would fire a sufficient number of times per step
      // in the tau-leaping group.
      if ((_tauLeaping.empty() &&
           _discreteGenerator.isMaximum(reactionIndex)) ||
          (_propensities[reactionIndex] / tau) * originalTau >
          minimumReactions) {
        _discreteGenerator.erase(reactionIndex);
        _tauLeaping.insert(reactionIndex);
      }
    }
    // Compute a new exponential deviate.
    _exponentialDeviate = _exponentialGenerator();
    ++_directStepCount;
  }
  else {
    _exponentialDeviate -= tau * directPmfSum;
  }

  // If the state is not valid, do not advance the time and return false.
  if (! _state.isValid()) {
    // CONTINUE
    //print(std::cout);
    return false;
  }
  else {
    if (areFinished) {
      // Advance the time to the ending time.
      _time = endTime;
    }
    else {
      _time += tau;
    }
  }

  return true;
}


template<class _PropensitiesFunctor>
inline
bool
HybridDirectTauLeaping<_PropensitiesFunctor>::
stepForward(const double tau)
{
  const std::vector<std::size_t>& tauLeapingReactions =
    _tauLeaping.getActiveReactions();
  if (! tauLeapingReactions.empty()) {
    for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
      const std::size_t m = tauLeapingReactions[i];
      _state.fireReaction(m, _tauLeaping.generatePoisson
                          (_propensities[m] * tau));
    }
    ++_tauLeapingStepCount;
  }
  return _state.isValid();
}


template<class _PropensitiesFunctor>
inline
bool
HybridDirectTauLeaping<_PropensitiesFunctor>::
stepMidpoint(const double tau)
{
  const std::vector<std::size_t>& tauLeapingReactions =
    _tauLeaping.getActiveReactions();
  if (tauLeapingReactions.empty()) {
    return true;
  }

  ++_tauLeapingStepCount;
  // Now the propensities have been calculated at the beginning of the
  // time interval.
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
#endif

  // Determine the midpoint populations.
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  const double half = 0.5 * tau;
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    // Note: The reaction counts are not integer.
    _state.fireReaction(&_p, m, _propensities[m] * half);
  }
  // Check the populations.
  for (std::size_t i = 0; i != _p.size(); ++i) {
    if (_p[i] < 0) {
      return false;
    }
  }

  // Determine the midpoint propensities.
  computePropensities(_p);

  // Take a step with the midpoint propensities.
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    _state.fireReaction(m, _tauLeaping.generatePoisson(_propensities[m] * tau));
  }
  // Return true if the state is valid.
  return _state.isValid();
}


template<class _PropensitiesFunctor>
inline
bool
HybridDirectTauLeaping<_PropensitiesFunctor>::
stepRungeKutta4(const double tau)
{
  const double halfTau = 0.5 * tau;
  const std::vector<std::size_t>& tauLeapingReactions =
    _tauLeaping.getActiveReactions();
  if (tauLeapingReactions.empty()) {
    return true;
  }

  ++_tauLeapingStepCount;
  // Now the propensities have been calculated at the beginning of the
  // time interval.
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
  assert(_propensities.size() == _k1.size());
#endif

  // k1
  std::copy(_propensities.begin(), _propensities.end(), _k1.begin());

  // k2
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    _state.fireReaction(&_p, m, halfTau * _k1[m]);
  }
  computePropensities(_p);
  std::copy(_propensities.begin(), _propensities.end(), _k2.begin());

  // k3
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    _state.fireReaction(&_p, m, halfTau * _k2[m]);
  }
  computePropensities(_p);
  std::copy(_propensities.begin(), _propensities.end(), _k3.begin());

  // k4
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    _state.fireReaction(&_p, m, tau * _k3[m]);
  }
  computePropensities(_p);
  std::copy(_propensities.begin(), _propensities.end(), _k4.begin());

  // Average propensities.
  for (std::size_t i = 0; i != _propensities.size(); ++i) {
    _propensities[i] = (1. / 6.) * (_k1[i] + 2 * (_k2[i] + _k3[i]) + _k4[i]);
  }

  // Take a step with the average propensities.
  for (std::size_t i = 0; i != tauLeapingReactions.size(); ++i) {
    const std::size_t m = tauLeapingReactions[i];
    _state.fireReaction(m, _tauLeaping.generatePoisson(tau * _propensities[m]));
  }

  // Return true if the state is valid.
  return _state.isValid();
}


// Move the volatile reactions to the direct group.
template<class _PropensitiesFunctor>
inline
void
HybridDirectTauLeaping<_PropensitiesFunctor>::
moveVolatileAndSlowToDirect(const double minimumPropensity)
{
  std::size_t reaction;
  const std::vector<std::size_t>& tauLeapingReactions =
    _tauLeaping.getActiveReactions();
  std::size_t i = 0;
  while (i != tauLeapingReactions.size()) {
    reaction = tauLeapingReactions[i];
    if (isVolatile(reaction) || _propensities[reaction] < minimumPropensity) {
      _tauLeaping.erase(reaction);
      _discreteGenerator.insert(reaction);
    }
    else {
      ++i;
    }
  }
}

template<class _PropensitiesFunctor>
inline
bool
HybridDirectTauLeaping<_PropensitiesFunctor>::
isVolatile(const std::size_t index)
{
  typedef typename container::SparseVector<std::size_t>::const_iterator
  const_iterator;

  // For each reactant.
  const container::SparseVector<std::size_t>& reactants =
    _propensitiesFunctor.getReaction(index).getReactants();
  for (const_iterator i = reactants.begin(); i != reactants.end(); ++i) {
    // If the stoichiometry times the population of the species is less than
    // the threshhold the reaction is volatile.
    if (i->second * _state.getPopulations()[i->first] < _volatileLimit) {
      return true;
    }
  }
  // For each product.
  const container::SparseVector<std::size_t>& products =
    _propensitiesFunctor.getReaction(index).getProducts();
  for (const_iterator i = products.begin(); i != products.end(); ++i) {
    // If the stoichiometry times the population of the species is less than
    // the threshhold the reaction is volatile.
    if (i->second * _state.getPopulations()[i->first] < _volatileLimit) {
      return true;
    }
  }
  // CONTINUE: When I implement modifiers:
  // The modifiers don't affect volatility.
  return false;
}

} // namespace stochastic
}
