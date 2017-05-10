// -*- C++ -*-

#if !defined(__stochastic_TauLeaping_ipp__)
#error This file is an implementation detail of TauLeaping.
#endif

namespace stlib
{
namespace stochastic
{

// Constructor.
template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
TauLeaping(const State& state,
           const PropensitiesFunctor& propensitiesFunctor,
           const double maxSteps) :
  Base(state, maxSteps),
  _propensitiesFunctor(propensitiesFunctor),
  // Invalid value.
  _time(std::numeric_limits<double>::max()),
  _propensities(state.getNumberOfReactions()),
  _reactionFirings(state.getNumberOfReactions()),
  // Construct.
  _discreteUniformGenerator(),
  _normalGenerator(&_discreteUniformGenerator),
  // CONTINUE: normal threshhold.
  _poissonGenerator(&_normalGenerator, 1000),
  _mu(state.getNumberOfSpecies()),
  _sigmaSquared(state.getNumberOfSpecies()),
  _highestOrder(state.getNumberOfSpecies()),
  _highestIndividualOrder(state.getNumberOfSpecies())
{
}

template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
initialize(const std::vector<double>& populations, const double time)
{
  typedef typename PropensitiesFunctor::ReactionType::SparseVectorSizeType
  SparseVectorSizeType;
  typedef typename SparseVectorSizeType::const_iterator const_iterator;

  // Initialize the state.
  Base::initialize(populations);
  _time = time;

  // CONTINUE: Can I move this to the constructor?
  //
  // Compute the orders for the species.
  //
  // Initialize the arrays.
  std::fill(_highestOrder.begin(), _highestOrder.end(), 0);
  std::fill(_highestIndividualOrder.begin(), _highestIndividualOrder.end(), 0);

  std::size_t sum;
  // Loop over the reactions.
  for (std::size_t n = 0; n != _state.getNumberOfReactions(); ++n) {
    const SparseVectorSizeType& reactants =
      _propensitiesFunctor.getReaction(n).getReactants();
    // The sum of the reactant coefficients.
    sum = container::sum(reactants);
    // Loop over the reactant species.
    for (const_iterator i = reactants.begin(); i != reactants.end(); ++i) {
      if (sum > _highestOrder[i->first]) {
        _highestOrder[i->first] = sum;
      }
      if (i->second > _highestIndividualOrder[i->first]) {
        _highestIndividualOrder[i->first] = i->second;
      }
    }
  }
}

// Try to take a step.  Return true if a step is taken.
template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
bool
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
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
    if (_CorrectNegativePopulations) {
      // Correct negative populations and reduce the time step if necessary.
      fixNegativePopulations(tau);
    }
    else {
      // If the state is not valid, do not advance the time and return false.
      std::ostringstream out;
      out << "There are negative species populations at time "
          << _time << ".";
      _error += out.str();
      return false;
    }
  }

  // No errors if we reached here.
  return true;
}

// Try to take a step.  Return true if a step is taken.
template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
bool
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
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
    if (_CorrectNegativePopulations) {
      // Correct negative populations and reduce the time step if necessary.
      fixNegativePopulations(tau);
    }
    else {
      // If the state is not valid, do not advance the time and return false.
      std::ostringstream out;
      out << "There are negative species populations at time "
          << _time << ".";
      _error += out.str();
      return false;
    }
  }

  // No errors if we reached here.
  return true;
}


// CONTINUE: This algorithm does not work.
#if 0
template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
fixNegativePopulations(const double tau)
{
  typedef typename State::StateChangeVectors::const_iterator const_iterator;
#ifdef STLIB_DEBUG
  assert(! _state.isValid());
#endif

  // First undo the time step.
  _time -= tau;

  // Determine the critical species, those species whose population did
  // or could become negative during the current step.
  std::set<std::size_t> criticalSpecies;
  for (std::size_t i = 0; i != _state.getNumberOfSpecies(); ++i) {
    if (_state.getPopulation(i) < 10) {
      criticalSpecies.insert(i);
    }
  }

  // Determine the critical reactions, those reactions that modify critical
  // species.
  std::vector<std::size_t> criticalReactions, nonCriticalReactions;
  // Loop over the reactions.
  for (std::size_t i = 0; i != _state.getNumberOfReactions(); ++i) {
    // No need to check reactions that did not fire.
    if (_reactionFirings[i] == 0) {
      continue;
    }
    bool isCritical = false;
    // Loop over the species that this reaction modifies.
    for (const_iterator j = _state.getStateChangeVectors().begin(i);
         j != _state.getStateChangeVectors().end(i); ++j) {
      if (criticalSpecies.count(j->first)) {
        isCritical = true;
        criticalReactions.push_back(i);
        break;
      }
    }
    if (! isCritical) {
      nonCriticalReactions.push_back(i);
    }
  }
  assert(! criticalReactions.empty());

  // Undo the current step.
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _state.fireReaction(m, -_reactionFirings[m]);
  }

  // Determine when the time intervals between critical reactions.
  std::vector<double> dt(criticalReactions.size());
  for (std::size_t i = 0; i != criticalReactions.size(); ++i) {
    std::size_t index = criticalReactions[i];
    dt[i] = tau / _reactionFirings[index];
  }

  // Set up an indexed priority queue to hold the critical reaction times.
  ads::IndexedPriorityQueueBinaryHeap<double>
  reactionTimes(criticalReactions.size());
  for (std::size_t i = 0; i != dt.size(); ++i) {
    // Choose the initial time offset to be half of dt.
    reactionTimes.push(i, _time + 0.5 * dt[i]);
  }

  double newTime = _time;
  // Fire critical reactions until a species population becomes negative.
  bool isNegative = false;
  while (! isNegative) {
    // Get the first reaction.
    const std::size_t i = reactionTimes.top();
    const std::size_t index = criticalReactions[i];
    // Fire the reaction.
    _state.fireReaction(index);
    // Check to see if any of the affected species populations became
    // negative.
    // Loop over the species that this reaction modifies.
    for (const_iterator j = _state.getStateChangeVectors().begin(index);
         j != _state.getStateChangeVectors().end(index); ++j) {
      if (_state.getPopulation(j->first) < 0) {
        isNegative = true;
        break;
      }
    }
    // If a species population became negative.
    if (isNegative) {
      // Undo the reaction.
      _state.fireReaction(index, -1);
    }
    else {
      // Record the time.
      newTime = reactionTimes.get(i);
    }
    // CONTINUE: I would need to pop and the push a new reaction time here
    // to correct this method.
  }

  // Fire the non-critical reactions.
  for (std::size_t i = 0; i != nonCriticalReactions.size(); ++i) {
    const std::size_t index = nonCriticalReactions[i];
    // Determine how many times the reaction fires according to the fraction
    // of the time step tau that we will take.  Round to the nearest integer.
    _state.fireReaction(index, std::floor((newTime - _time) /
                                          tau * _reactionFirings[index] + 0.5));
  }

  // Record the time of the last valid reaction firing.
  _time = newTime;
}
#endif


template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
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


template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
stepForward(const double tau)
{
  // Fire the reactions.
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _reactionFirings[m] = _poissonGenerator(_propensities[m] * tau);
    _state.fireReaction(m, _reactionFirings[m]);
  }
}


template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
stepMidpoint(const double tau)
{
  // Now the propensities have been calculated at the beginning of the
  // time interval.
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
#endif

  // Determine the midpoint populations.
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  const double half = 0.5 * tau;
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    // Note: The reaction counts are not integer.
    _state.fireReaction(&_p, m, _propensities[m] * half);
  }
  // Determine the midpoint propensities.
  computePropensities(_p);

  // Fire the reactions.
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _reactionFirings[m] = _poissonGenerator(_propensities[m] * tau);
    _state.fireReaction(m, _reactionFirings[m]);
  }
}

template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
stepRungeKutta4(const double tau)
{
  // Now the propensities have been calculated at the beginning of the
  // time interval.
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
  assert(_propensities.size() == _k1.size());
#endif

  // k1
  for (std::size_t i = 0; i != _k1.size(); ++i) {
    _k1[i] = tau * _propensities[i];
  }

  // k2
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, 0.5 * _k1[m]);
  }
  computePropensities(_p);
  for (std::size_t i = 0; i != _k2.size(); ++i) {
    _k2[i] = tau * _propensities[i];
  }

  // k3
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, 0.5 * _k2[m]);
  }
  computePropensities(_p);
  for (std::size_t i = 0; i != _k3.size(); ++i) {
    _k3[i] = tau * _propensities[i];
  }

  // k4
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, _k3[m]);
  }
  computePropensities(_p);
  for (std::size_t i = 0; i != _k4.size(); ++i) {
    _k4[i] = tau * _propensities[i];
  }

  // Average propensities times tau.
  for (std::size_t i = 0; i != _propensities.size(); ++i) {
    _propensities[i] = (1. / 6.) * (_k1[i] + 2 * (_k2[i] + _k3[i]) + _k4[i]);
  }

  // Fire the reactions.
  for (std::size_t m = 0; m != _reactionFirings.size(); ++m) {
    _reactionFirings[m] = _poissonGenerator(_propensities[m]);
    _state.fireReaction(m, _reactionFirings[m]);
  }
}

template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
computePropensities(const std::vector<double>& populations)
{
  for (std::size_t m = 0; m < _propensities.size(); ++m) {
    _propensities[m] = _propensitiesFunctor(m, populations);
  }
}

template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
double
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
computeTau(const double epsilon)
{
#ifdef STLIB_DEBUG
  assert(epsilon > 0);
#endif

  // Compute mu and sigmaSquared.
  computeMuAndSigmaSquared();

  // Initialize tau to infinity.
  double tau = std::numeric_limits<double>::max();

  // Take the minimum over the species.
  double numerator, temp, a, b;
  for (std::size_t n = 0; n != _mu.size(); ++n) {
    // If the n_th species is not a reactant in any reaction.
    if (_highestOrder[n] == 0) {
      // This species does not affect tau.
      continue;
    }
    numerator = std::max(epsilon * _state.getPopulation(n) / computeG(n), 1.0);

    if (_mu[n] != 0) {
      a = numerator / std::abs(_mu[n]);
    }
    else {
      a = std::numeric_limits<double>::max();
    }

    if (_sigmaSquared[n] != 0) {
      b = numerator * numerator / _sigmaSquared[n];
    }
    else {
      b = std::numeric_limits<double>::max();
    }
    temp = std::min(a, b);
    if (temp < tau) {
      tau = temp;
    }
  }

  return tau;
}


// Compute the g described in "Efficient step size selection for the
// tau-leaping simulation method".
template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
double
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
computeG(const std::size_t speciesIndex) const
{
  const std::size_t order = _highestOrder[speciesIndex];

  if (order == 1) {
    return 1.0;
  }
  else if (order == 2) {
    if (_highestIndividualOrder[speciesIndex] == 1) {
      return 2.0;
    }
    else if (_highestIndividualOrder[speciesIndex] == 2) {
      return 2.0 + 1.0 / (_state.getPopulation(speciesIndex) - 1);
    }
  }
  else if (order == 3) {
    if (_highestIndividualOrder[speciesIndex] == 1) {
      return 3.0;
    }
    else if (_highestIndividualOrder[speciesIndex] == 2) {
      return 1.5 * (2.0 + 1.0 / (_state.getPopulation(speciesIndex) - 1.0));
    }
    else if (_highestIndividualOrder[speciesIndex] == 3) {
      return 3.0 + 1.0 / (_state.getPopulation(speciesIndex) - 1)
             + 2.0 / (_state.getPopulation(speciesIndex) - 2);
    }
  }

  // Catch any other cases with an assertion failure.
  assert(false);
  return 0;
}

template < class _PropensitiesFunctor,
           bool _CorrectNegativePopulations >
inline
void
TauLeaping<_PropensitiesFunctor, _CorrectNegativePopulations>::
computeMuAndSigmaSquared()
{
  typedef typename State::StateChangeVectors::const_iterator const_iterator;

  // Initialize.
  std::fill(_mu.begin(), _mu.end(), 0.0);
  std::fill(_sigmaSquared.begin(), _sigmaSquared.end(), 0.0);

  double propensity;
  // Loop over the reactions.
  for (std::size_t m = 0; m < _propensities.size(); ++m) {
    propensity = _propensities[m];
#ifdef STLIB_DEBUG
    assert(propensity >= 0);
#endif
    // For each species that is modified by this reaction.
    for (const_iterator i = _state.getStateChangeVectors().begin(m);
         i != _state.getStateChangeVectors().end(m); ++i) {
      _mu[i->first] += i->second * propensity;
      _sigmaSquared[i->first] += i->second * i->second * propensity;
    }
  }
}

} // namespace stochastic
}
