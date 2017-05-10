// -*- C++ -*-

#if !defined(__stochastic_OdeReaction_ipp__)
#error This file is an implementation detail of OdeReaction.
#endif

namespace stlib
{
namespace stochastic
{

// Constructor.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
OdeReaction(const State& state,
            const PropensitiesFunctor& propensitiesFunctor,
            const double maxSteps) :
  Base(state, maxSteps),
  // Invalid value.
  _time(std::numeric_limits<double>::max()),
  _propensitiesFunctor(propensitiesFunctor),
  _propensities(state.getNumberOfReactions())
{
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
initialize(const std::vector<double>& populations, const double time)
{
  // Initialize the state.
  Base::initialize(populations);
  _time = time;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
setupRungeKuttaCashKarp()
{
  setupFixedRungeKuttaCashKarp();
  _solutionError.resize(_state.getNumberOfSpecies());
}

// Simulate until the end time is reached.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
simulateRungeKuttaCashKarp(const double epsilon, const double endTime)
{
  // For the initial time step, advance until it is expected that one
  // reaction fires.
  double dt, nextDt = 0;
  computePropensities(0.);
  const double propensitiesSum = std::accumulate(_propensities.begin(),
                                 _propensities.end(), 0.);
  if (propensitiesSum != 0) {
    dt = 1. / propensitiesSum;
  }
  else {
    dt = endTime - _time;
  }

  // Step until we reach the end time.
  while (_time < endTime) {
    // Check that we have not exceeded the allowed number of steps.
    if (! incrementStepCount()) {
      setStepCountError();
      return false;
    }

    const double initialDt = dt;
    bool finish = false;
    // Reduce dt if it will take us past the end time.
    if (_time + dt > endTime) {
      dt = endTime - _time;
      finish = true;
    }
    // Try a step.
    if (! stepRungeKuttaCashKarp(&dt, &nextDt, epsilon)) {
      // If we encounter an error return false.
      return false;
    }
    // If this is the last step.
    if (finish && dt == initialDt) {
      // Do this to avoid problems with round-off errors.
      _time = endTime;
    }
    dt = nextDt;
  }
  return true;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
setupFixedForward()
{
}

// Simulate with fixed size steps until the end time is reached.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
simulateFixedForward(const double dt, const double endTime)
{
  // Step until we reach the end time.
  while (_time < endTime) {
    // Try a step.
    if (! stepFixed(&OdeReaction::stepForward, dt, endTime)) {
      // If we encounter an error return false.
      return false;
    }
  }
  return true;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
setupFixedMidpoint()
{
  _p.resize(_state.getNumberOfSpecies());
}

// Simulate with fixed size steps until the end time is reached.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
simulateFixedMidpoint(const double dt, const double endTime)
{
  // Step until we reach the end time.
  while (_time < endTime) {
    // Try a step.
    if (! stepFixed(&OdeReaction::stepMidpoint, dt, endTime)) {
      // If we encounter an error return false.
      return false;
    }
  }
  return true;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
setupFixedRungeKutta4()
{
  _p.resize(_state.getNumberOfSpecies());
  _k1.resize(_state.getNumberOfReactions());
  _k2.resize(_state.getNumberOfReactions());
  _k3.resize(_state.getNumberOfReactions());
  _k4.resize(_state.getNumberOfReactions());
}

//! Simulate with fixed size steps until the end time is reached.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
simulateFixedRungeKutta4(const double dt, const double endTime)
{
  // Step until we reach the end time.
  while (_time < endTime) {
    // Try a step.
    if (! stepFixed(&OdeReaction::stepRungeKutta4, dt, endTime)) {
      // If we encounter an error return false.
      return false;
    }
  }
  return true;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
setupFixedRungeKuttaCashKarp()
{
  _p.resize(_state.getNumberOfSpecies());
  _k1.resize(_state.getNumberOfReactions());
  _k2.resize(_state.getNumberOfReactions());
  _k3.resize(_state.getNumberOfReactions());
  _k4.resize(_state.getNumberOfReactions());
  _k5.resize(_state.getNumberOfReactions());
  _k6.resize(_state.getNumberOfReactions());
}

//! Simulate with fixed size steps until the end time is reached.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
simulateFixedRungeKuttaCashKarp
(const double dt, const double endTime)
{
  // Step until we reach the end time.
  while (_time < endTime) {
    // Try a step.
    if (! stepFixed(&OdeReaction::stepRungeKuttaCashKarp, dt, endTime)) {
      // If we encounter an error return false.
      return false;
    }
  }
  return true;
}

#if 0
// Take a step. Return true if the state is valid.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
step(MemberFunctionPointer method, double dt, const double endTime)
{
  CONTNUE;
  // Decrease dt if it will take us past the end time.
  double time = _state.getTimeOffset() + dt;
  // If the time leap will take us past the end time.
  if (time > endTime) {
    dt = endTime - _state.getTimeOffset();
    // Advance the time to the ending time.
    time = endTime;
  }

  // Advance the state.
  (this->*method)(dt);
  // Advance the time.
  _state.setTimeOffset(time);
  // Return true if there are no negative populations.
  return _state.isValid();
}
#endif

// Take a step. Return true if the state is valid.
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
stepFixed(MemberFunctionPointer method, double dt, const double endTime)
{
  // Check that we have not exceeded the allowed number of steps.
  if (! incrementStepCount()) {
    setStepCountError();
    return false;
  }

  // Advance the time by dt.
  double t = _time;
  // If the time leap will take us past the end time.
  if (t + dt > endTime) {
    dt = endTime - _time;
    // Advance the time to the ending time.
    t = endTime;
  }
  else {
    t += dt;
  }

  // Advance the state.
  (this->*method)(dt);
  _time = t;
  // Return true if there are no negative populations.
  return _state.isValid();
}


template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
stepForward(const double dt)
{
  // Compute the propensities.
  computePropensities(0.);
  // Advance the state.
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(m, _propensities[m] * dt);
  }
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
stepMidpoint(const double dt)
{
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
#endif

  // Compute the propensities.
  computePropensities(0.);
  // Determine the midpoint populations.
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  const double half = 0.5 * dt;
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, _propensities[m] * half);
  }

  // Determine the midpoint propensities.
  computePropensities(_p, half);
  // Take a step with the midpoint propensities.
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(m, _propensities[m] * dt);
  }
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
stepRungeKutta4(const double dt)
{
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size());
  assert(_propensities.size() == _k1.size());
#endif
  // Compute the propensities.
  computePropensities(0.);

  // k1
  for (std::size_t i = 0; i != _k1.size(); ++i) {
    _k1[i] = dt * _propensities[i];
  }

  // k2
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, 0.5 * _k1[m]);
  }
  computePropensities(_p, 0.5 * dt);
  for (std::size_t i = 0; i != _k2.size(); ++i) {
    _k2[i] = dt * _propensities[i];
  }

  // k3
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, 0.5 * _k2[m]);
  }
  computePropensities(_p, 0.5 * dt);
  for (std::size_t i = 0; i != _k3.size(); ++i) {
    _k3[i] = dt * _propensities[i];
  }

  // k4
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, _k3[m]);
  }
  computePropensities(_p, dt);
  for (std::size_t i = 0; i != _k4.size(); ++i) {
    _k4[i] = dt * _propensities[i];
  }

  // Take a step with the average propensities.
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(m, (1. / 6.) * (_k1[m] +
                                        2 * (_k2[m] + _k3[m]) + _k4[m]));
  }
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
computeRungeKuttaCashKarp(const double dt)
{
#ifdef STLIB_DEBUG
  assert(_state.getNumberOfSpecies() == _p.size() &&
         _propensities.size() == _k1.size() &&
         _propensities.size() == _k2.size() &&
         _propensities.size() == _k3.size() &&
         _propensities.size() == _k4.size() &&
         _propensities.size() == _k5.size() &&
         _propensities.size() == _k6.size());
#endif
  const double
  a2 = 0.2,
  a3 = 0.3,
  a4 = 0.6,
  a5 = 1.0,
  a6 = 0.875,
  b21 = 0.2,
  b31 = 3. / 40.,
  b32 = 9. / 40.,
  b41 = 0.3,
  b42 = -0.9,
  b43 = 1.2,
  b51 = -11. / 54.,
  b52 = 2.5,
  b53 = -70. / 27.,
  b54 = 35. / 27.,
  b61 = 1631. / 55296.,
  b62 = 175. / 512.,
  b63 = 575. / 13824.,
  b64 = 44275. / 110592.,
  b65 = 253. / 4096.;

  // Initial propensities.
  computePropensities(0.);
  std::copy(_propensities.begin(), _propensities.end(), _k1.begin());

  // First step.
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, dt * b21 * _k1[m]);
  }

  // Second step.
  computePropensities(_p, a2 * dt);
  std::copy(_propensities.begin(), _propensities.end(), _k2.begin());
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, dt * (b31 * _k1[m] + b32 * _k2[m]));
  }

  // Third step.
  computePropensities(_p, a3 * dt);
  std::copy(_propensities.begin(), _propensities.end(), _k3.begin());
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, dt * (b41 * _k1[m] + b42 * _k2[m] +
                                      b43 * _k3[m]));
  }

  // Fourth step.
  computePropensities(_p, a4 * dt);
  std::copy(_propensities.begin(), _propensities.end(), _k4.begin());
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, dt * (b51 * _k1[m] + b52 * _k2[m] +
                                      b53 * _k3[m] + b54 * _k4[m]));
  }

  // Fifth step.
  computePropensities(_p, a5 * dt);
  std::copy(_propensities.begin(), _propensities.end(), _k5.begin());
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            _p.begin());
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_p, m, dt * (b61 * _k1[m] + b62 * _k2[m] +
                                      b63 * _k3[m] + b64 * _k4[m] +
                                      b65 * _k5[m]));
  }

  // Sixth step.
  computePropensities(_p, a6 * dt);
  std::copy(_propensities.begin(), _propensities.end(), _k6.begin());
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
solutionRungeKuttaCashKarp(const double dt)
{
  const double
  c1 = 37. / 378.,
  c3 = 250. / 621.,
  c4 = 125. / 594.,
  c6 = 512. / 1771.;

  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(m, dt * (c1 * _k1[m] + c3 * _k3[m] + c4 * _k4[m] +
                                 c6 * _k6[m]));
  }
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
double
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
errorRungeKuttaCashKarp(const double dt)
{
  const double
  c1 = 37. / 378.,
  c3 = 250. / 621.,
  c4 = 125. / 594.,
  c6 = 512. / 1771.,
  dc1 = c1 - 2825. / 27648.,
  dc3 = c3 - 18575. / 48384.,
  dc4 = c4 - 13525. / 55296.,
  dc5 = -277. / 14336.,
  dc6 = c6 - 0.25;
#if 0
  std::cout << "dt = " << dt << '\n';
  std::cout << "Populations:\n";
  for (std::size_t n = 0; n != _state.getNumberOfSpecies(); ++n) {
    std::cout << _state.getPopulation(n) << " ";
  }
  std::cout << '\n';
#endif
  double error = 0;
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    const double e = std::abs(dt * (dc1 * _k1[m] + dc3 * _k3[m] +
                                    dc4 * _k4[m] + dc5 * _k5[m] +
                                    dc6 * _k6[m])) /
                     std::max(1., _state.getReactionCount(m));
    //std::cout << m << " " << _propensities[m] << " " << e << '\n';
    error = std::max(error, e);
  }
  return error;
}

#if 0
template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
double
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
errorRungeKuttaCashKarp(const double dt)
{
  const double
  c1 = 37. / 378.,
  c3 = 250. / 621.,
  c4 = 125. / 594.,
  c6 = 512. / 1771.,
  dc1 = c1 - 2825. / 27648.,
  dc3 = c3 - 18575. / 48384.,
  dc4 = c4 - 13525. / 55296.,
  dc5 = -277. / 14336.,
  dc6 = c6 - 0.25;

  std::fill(_solutionError.begin(), _solutionError.end(), 0.);
  for (std::size_t m = 0; m != _state.getNumberOfReactions(); ++m) {
    _state.fireReaction(&_solutionError, m, dt * (dc1 * _k1[m] + dc3 * _k3[m] +
                        dc4 * _k4[m] + dc5 * _k5[m] +
                        dc6 * _k6[m]));
  }
#if 1
  // Here is a straight-forward method for determining the maximum relative
  // error.
  std::cout << "Populations:\n";
  for (std::size_t n = 0; n != _state.getNumberOfSpecies(); ++n) {
    std::cout << _state.getPopulation(n) << " ";
  }
  std::cout << '\n';

  double error = 0;
  for (std::size_t n = 0; n != _solutionError.size(); ++n) {
    const double e = std::abs(_solutionError[n]) /
                     std::max(1., _state.getPopulation(n));
    std::cout << n << " " << _propensities[n] << " " << e << '\n';
    error = std::max(error, e);
  }
  return error;
#else
  // Here is a more efficient method that avoids costly divisions.
  // e[i] / s[i] < e[j] / s[j];
  // e[i] * s[j] < e[j] * s[i];
  double error = std::abs(_solutionError[0]);
  double scale = std::max(1., _state.getPopulation(0));
  for (std::size_t n = 1; n != _solutionError.size(); ++n) {
    const double e = std::abs(_solutionError[n]);
    const double s = std::max(1., _state.getPopulation(n));
    if (error * s < e * scale) {
      error = e;
      scale = s;
    }
  }
  return error / scale;
#endif
}
#endif

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
bool
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
stepRungeKuttaCashKarp(double* dt, double* nextDt, const double epsilon)
{
  const double
  Safety = 0.9,
  PGrow = -0.2,
  PShrink = -0.25,
  // (5 / Safety)^(1 / PGrow)
  ErrorCondition = 1.89e-4;

  double scaledError;
  while (true) {
    // Determine the maximum relative error with the current step size.
    computeRungeKuttaCashKarp(*dt);
    scaledError = errorRungeKuttaCashKarp(*dt) / epsilon;
    // If the error is acceptable.
    if (scaledError <= 1.) {
      // Take the step.
      solutionRungeKuttaCashKarp(*dt);
      // Accept the step.
      _time += *dt;
      // CONTINUE: Fix negative populations.
      _state.fixNegativePopulations();
      break;
    }
    else {
      //std::cout << "Reduce the step size.\n";
      // Reduce the step size.
      const double candidateDt =
        Safety * (*dt) * std::pow(scaledError, PShrink);
      // Note: dt could be negative.
      *dt = (*dt >= 0. ? std::max(candidateDt, 0.1 * (*dt)) :
             std::min(candidateDt, 0.1 * (*dt)));
    }
    // Check for step size underflow.
    if (_time + *dt == _time) {
      std::ostringstream out;
      out << "Step size underflow: dt = " << *dt << ".";
      _error += out.str();
      return false;
    }
  }
  // Compute the next step size. Allow no more than a factor of 5 increase.
  if (scaledError > ErrorCondition) {
    *nextDt = Safety * (*dt) * std::pow(scaledError, PGrow);
    //std::cout << "Decrease the next step size to " << *nextDt << ".\n";
  }
  else {
    *nextDt = 5. * (*dt);
    //std::cout << "Increase the next step size to " << *nextDt << ".\n";
  }
  return true;
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
_computePropensities(std::false_type /*IsInhomogeneous*/,
                     const std::vector<double>& populations,
                     const double /*timeOffset*/)
{
  for (std::size_t m = 0; m < _propensities.size(); ++m) {
    //_propensities[m] = _propensitiesFunctor(m, populations);
    // CONTINUE
    _propensities[m] = std::max(0., _propensitiesFunctor(m, populations));
  }
}

template<bool _IsInhomogeneous, class _PropensitiesFunctor>
inline
void
OdeReaction<_IsInhomogeneous, _PropensitiesFunctor>::
_computePropensities(std::true_type /*IsInhomogeneous*/,
                     const std::vector<double>& populations,
                     const double timeOffset)
{
  _propensitiesFunctor(&_propensities, populations, _time + timeOffset);
}

} // namespace stochastic
}
