// -*- C++ -*-

#if !defined(__stochastic_TauLeapingImplicit_ipp__)
#error This file is an implementation detail of TauLeapingImplicit.
#endif

namespace stlib
{
namespace stochastic
{

// Constructor.
inline
TauLeapingImplicit::
TauLeapingImplicit(const State& state,
                   const PropensitiesFunctor& propensitiesFunctor,
                   const double maxSteps) :
  Base(state, maxSteps),
  _propensitiesFunctor(propensitiesFunctor),
  // Invalid value.
  _time(std::numeric_limits<double>::max()),
  _propensities(state.getNumberOfReactions()),
  // Construct.
  _discreteUniformGenerator(),
  _normalGenerator(&_discreteUniformGenerator),
  // CONTINUE: normal threshhold.
  _poissonGenerator(&_normalGenerator, 1000)
{
}

inline
void
TauLeapingImplicit::
initialize(const std::vector<double>& populations, const double time)
{
  Base::initialize(populations);
  _time = time;
}

// Try to take a step.  Return true if a step is taken.
inline
bool
TauLeapingImplicit::
stepFixed(MemberFunctionPointer method, double tau, const double endTime)
{
  // If we have reached the termination condition.
  if (_time >= endTime) {
    return false;
  }

  // Adjust the time step if necessary.
  double newTime = _time + tau;
  if (newTime >= endTime) {
    tau = endTime - _time;
    newTime = endTime;
  }

  // Take the step.
  (this->*method)(tau);
  // Advance the time.
  _time = newTime;
  return true;
}


inline
void
TauLeapingImplicit::
stepEuler(const double tau)
{
  const std::size_t N = _state.getNumberOfSpecies();
  const std::size_t M = _state.getNumberOfReactions();
  // Solve the following equation with the Newton Raphson method.
  // X(t + tau) = X(t) + sum_j(v_j (tau (X(t + tau) -X(t)) +
  //                                P_j(a_j(X(t)), tau)))
  // We rearrange to put in the form f = 0.
  // X(t + tau) - X(t) - sum_j(v_j (tau (X(t + tau) -X(t)) +
  //                                P_j(a_j(X(t)), tau))) = 0

  // The constant part of the equation, meaning the part that does not
  // depend on X(t + tau) is
  // constant = X(t) + sum_j(v_j (P_j(tau a_j(X(t))) - tau a_j(X(t))))
  // constant = X + sum_j(v_j (P_j - tau a_j))
  Eigen::VectorXd constant(N);
  // Initialize to X(t).
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            constant.data());
  // Compute the propensities at the initial time.
  computePropensities();
  // Compute the constant part and the intial value of the function
  // (when X(t+tau) = X(t)),
  // f = - sum_j P_j.
  // For each reaction.
  for (std::size_t m = 0; m != M; ++m) {
    const double p = _poissonGenerator(_propensities[m] * tau);
    // Increment the reaction count for this reaction channel. Note that this
    // only gives approximate results. The implicit tau-leaping method solves
    // a nonlinear equation to determine the species populations.
    _state.incrementReactionCounts(m, p);
    const double times = p - _propensities[m] * tau;
    for (State::StateChangeVectors::const_iterator i =
           _state.getStateChangeVectors().begin(m);
         i != _state.getStateChangeVectors().end(m); ++i) {
      constant[i->first] += times * i->second;
    }
  }

  // The initial guess for X(t + tau) is X(t).
  Eigen::VectorXd x(N);
  std::copy(_state.getPopulations().begin(), _state.getPopulations().end(),
            x.data());

  std::vector<double> populations(N);
  Eigen::VectorXd f;
  f.setZero(N);

  //
  // Newton-Raphson iterations.
  //
  const double tolerance = 0.1;
  for (std::size_t iteration = 0; iteration != 16; ++iteration) {
    // Evaluate the propensities for the current value of x.
    std::copy(x.data(), x.data() + N, populations.begin());
    computePropensities(populations);

    // Evaluate the function.
    f = x - constant;
    for (std::size_t m = 0; m != M; ++m) {
      for (State::StateChangeVectors::const_iterator i =
             _state.getStateChangeVectors().begin(m);
           i != _state.getStateChangeVectors().end(m); ++i) {
        f(i->first) -= i->second * tau * _propensities[m];
      }
    }

    // The Jacobian matrix.
    // X(t + tau) - sum_j(tau v_j a_j(X(t + tau)))
    Eigen::MatrixXd jacobian;
    jacobian.setIdentity(N, N);
    container::SparseVector<double> derivatives;
    // For each reaction.
    for (std::size_t m = 0; m != M; ++m) {
      const Reaction& r = _propensitiesFunctor.getReaction(m);
      r.computePropensityFunctionDerivatives
      (_propensities[m], _state.getPopulations(), &derivatives);
      // Loop over columns to update.
      for (container::SparseVector<double>::const_iterator j = derivatives.begin();
           j != derivatives.end(); ++j) {
        // Loop over rows to update.
        for (State::StateChangeVectors::const_iterator i =
               _state.getStateChangeVectors().begin(m);
             i != _state.getStateChangeVectors().end(m); ++i) {
          jacobian(i->first, j->first) -= tau * i->second * j->second;
        }
      }
    }

    // Solve for delta.
    f *= -1.;
    //std::cout << "Jacobian =\n" << jacobian << '\n'
    //<< "f = \n" << f << '\n';
    Eigen::VectorXd delta(N);
    // Old method.
    //jacobian.lu().solve(f, &delta);
    Eigen::FullPivLU<Eigen::MatrixXd> lu(jacobian);
    delta = lu.solve(f);
    x += delta;
    //std::cout << "delta =\n" << delta << '\n';
    if (delta.cwiseAbs().sum() < tolerance) {
      break;
    }
  }

  for (std::size_t n = 0; n != N; ++n) {
    // Ensure non-negativity and round to the nearest integer.
    _state.setPopulation(n, std::floor(std::max(0., x(n)) + 0.5));
  }
}

inline
void
TauLeapingImplicit::
computePropensities(const std::vector<double>& populations)
{
  for (std::size_t m = 0; m < _propensities.size(); ++m) {
    _propensities[m] = _propensitiesFunctor(m, populations);
  }
}

} // namespace stochastic
}
