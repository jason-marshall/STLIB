// -*- C++ -*-

/*!
  \file stochastic/OdeReaction.h
  \brief ODE integration of each reaction.
*/

#if !defined(__stochastic_OdeReaction_h__)
#define __stochastic_OdeReaction_h__

#include "stlib/stochastic/Solver.h"

#include <sstream>

namespace stlib
{
namespace stochastic
{

//! Perform ODE integration of each reaction.
/*!
  \param _IsInhomogenous Whether the propensities are time-inhomogeneous.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template<bool _IsInhomogenous, class _PropensitiesFunctor>
class OdeReaction : public Solver
{
protected:

  //! The base class.
  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;

  //
  // Private types.
  //
private:

  typedef void (OdeReaction::*MemberFunctionPointer)(double);

  //
  // Member data.
  //
private:

  double _time;
  PropensitiesFunctor _propensitiesFunctor;
  std::vector<double> _propensities;

  //! Used for Runge-Kutta and midpoint.
  std::vector<double> _p;
  //! Used for Runge-Kutta Cash-Karp
  std::vector<double> _solutionError;
  //! Used for Runge-Kutta.
  std::vector<double> _k1, _k2, _k3, _k4, _k5, _k6;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  OdeReaction();
  //! Copy constructor not implemented.
  OdeReaction(const OdeReaction&);
  //! Assignment operator not implemented.
  OdeReaction&
  operator=(const OdeReaction&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  OdeReaction(const State& state,
              const PropensitiesFunctor& propensitiesFunctor,
              double maxSteps);

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and set the time.
  void
  initialize(const std::vector<double>& populations, double time);

  //! Setup for a simulation with adaptive step size, Cash-Karp Runge-Kutta.
  void
  setupRungeKuttaCashKarp();

  //! Simulate until the end time is reached.
  bool
  simulateRungeKuttaCashKarp(double epsilon, double endTime);

  //! Setup for a simulation with fixed step, forward.
  void
  setupFixedForward();

  //! Simulate with fixed size steps until the end time is reached.
  bool
  simulateFixedForward(double dt, double endTime);

  //! Setup for a simulation with fixed step, midpoint.
  void
  setupFixedMidpoint();

  //! Simulate with fixed size steps until the end time is reached.
  bool
  simulateFixedMidpoint(double dt, double endTime);

  //! Setup for a simulation with fixed step, fourth order Runge-Kutta.
  void
  setupFixedRungeKutta4();

  //! Simulate with fixed size steps until the end time is reached.
  bool
  simulateFixedRungeKutta4(double dt, double endTime);

  //! Setup for a simulation with fixed step, Cash-Karp Runge-Kutta.
  void
  setupFixedRungeKuttaCashKarp();

  //! Simulate with fixed size steps until the end time is reached.
  bool
  simulateFixedRungeKuttaCashKarp(double dt, double endTime);

  //! Try to take a step.  Return false if the end time has been reached.
  bool
  step(MemberFunctionPointer method, double epsilon, double endTime);

  //! Take a step. Return true if the state is valid.
  bool
  stepFixed(MemberFunctionPointer method, double dt, double endTime);

private:

  void
  stepForward(double dt);

  void
  stepMidpoint(double dt);

  void
  stepRungeKutta4(double dt);

  //! Perform computations that enable computing a step or the error.
  void
  computeRungeKuttaCashKarp(double dt);

  //! Update the solution in the step.
  void
  solutionRungeKuttaCashKarp(double dt);

  //! Compute the relative error if we take the step.
  /*! The relative error is the maximum over the specise of
    error[n] / max(1, x[n]) where x[n] is the n_th species population and
    error[n] is the truncation error.
  */
  double
  errorRungeKuttaCashKarp(double dt);

  //! Take a step with the specified time step.
  void
  stepRungeKuttaCashKarp(const double dt)
  {
    computeRungeKuttaCashKarp(dt);
    solutionRungeKuttaCashKarp(dt);
  }

  //! Try to take a step with the specified time step and error tolerance.
  /*! If the step is too large, reduce until an acceptable step is found.
    Return true if a step can be taken.
  */
  bool
  stepRungeKuttaCashKarp(double* dt, double* nextDt, double epsilon);

  void
  computePropensities(const double timeOffset)
  {
    computePropensities(_state.getPopulations(), timeOffset);
  }

  //! Choose between the time-homogeneous and the time-inhomogeneous version.
  void
  computePropensities(const std::vector<double>& populations,
                      const double timeOffset)
  {
    _computePropensities(std::integral_constant<bool, _IsInhomogenous>(),
                         populations, timeOffset);
  }

  void
  _computePropensities(std::false_type /*IsInhomogeneous*/,
                       const std::vector<double>& populations,
                       const double /*timeOffset*/);

  void
  _computePropensities(std::true_type /*IsInhomogeneous*/,
                       const std::vector<double>& populations,
                       const double timeOffset);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the time.
  double
  getTime() const
  {
    return _time;
  }

  //@}
};

} // namespace stochastic
}

#define __stochastic_OdeReaction_ipp__
#include "stlib/stochastic/OdeReaction.ipp"
#undef __stochastic_OdeReaction_ipp__

#endif
