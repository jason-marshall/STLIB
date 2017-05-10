// -*- C++ -*-

/*!
  \file stochastic/TauLeapingImplicit.h
  \brief The implicit tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingImplicit_h__)
#define __stochastic_TauLeapingImplicit_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"

#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.h"

#include "Eigen/LU"
#include "Eigen/Core"

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using the implicit tau-leaping method.
class TauLeapingImplicit : public Solver
{
private:

  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef PropensitiesSingle<true> PropensitiesFunctor;
  //! A reaction.
  typedef PropensitiesFunctor::ReactionType Reaction;
  //! The Poisson generator.
  typedef numerical::PoissonGeneratorInvAcNormSure<> PoissonGenerator;
  //! The discrete uniform generator.
  typedef PoissonGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;
  //! The normal generator.
  typedef PoissonGenerator::NormalGenerator NormalGenerator;

  //
  // Private types.
  //
private:

  typedef void (TauLeapingImplicit::*MemberFunctionPointer)(double);

  //
  // Member data.
  //
private:

  PropensitiesFunctor _propensitiesFunctor;
  double _time;
  std::vector<double> _propensities;
  DiscreteUniformGenerator _discreteUniformGenerator;
  NormalGenerator _normalGenerator;
  PoissonGenerator _poissonGenerator;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TauLeapingImplicit();
  //! Copy constructor not implemented.
  TauLeapingImplicit(const TauLeapingImplicit&);
  //! Assignment operator not implemented.
  TauLeapingImplicit&
  operator=(const TauLeapingImplicit&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  TauLeapingImplicit(const State& state,
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

  //! Simulate with fixed size steps until the termination condition is reached.
  void
  simulateFixedEuler(const double tau, const double endTime)
  {
    // Step until we reach the end time or the maximum number of steps is
    // exceeded.
    while (true) {
      if (! incrementStepCount()) {
        setStepCountError();
        break;
      }
      if (! stepFixed(&TauLeapingImplicit::stepEuler, tau, endTime)) {
        break;
      }
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  bool
  stepFixed(MemberFunctionPointer method, double tau, double endTime);

protected:

  //! Record a step count error message. Record the current time.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    Base::setStepCountError();
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached at time = "
        << _time << ".";
    _error += out.str();
  }

private:

  void
  stepEuler(double tau);

  void
  computePropensities()
  {
    computePropensities(_state.getPopulations());
  }

  void
  computePropensities(const std::vector<double>& populations);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the state.
  const State&
  getState() const
  {
    return _state;
  }

  //! Return the time.
  double
  getTime() const
  {
    return _time;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a reference to the discrete, uniform generator.
  DiscreteUniformGenerator&
  getDiscreteUniformGenerator()
  {
    return _discreteUniformGenerator;
  }

  //@}
};

} // namespace stochastic
}

#define __stochastic_TauLeapingImplicit_ipp__
#include "stlib/stochastic/TauLeapingImplicit.ipp"
#undef __stochastic_TauLeapingImplicit_ipp__

#endif
