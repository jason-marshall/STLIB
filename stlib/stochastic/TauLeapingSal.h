// -*- C++ -*-

/*!
  \file stochastic/TauLeapingSal.h
  \brief The Step-Anticipation tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingSal_h__)
#define __stochastic_TauLeapingSal_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/PropensityTimeDerivatives.h"

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueBinaryHeap.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.h"

#include <set>

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using the Step-anticipation tau-leaping method.
class TauLeapingSal : public Solver
{
private:

  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef PropensitiesSingle<true> PropensitiesFunctor;
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

  typedef void (TauLeapingSal::*MemberFunctionPointer)(double);

  //
  // Member data.
  //
private:

  PropensitiesFunctor _propensitiesFunctor;
  PropensityTimeDerivatives _propensityTimeDerivatives;
  double _time;
  std::vector<double> _propensities;
  std::vector<double> _dpdt;
  std::vector<double> _rateConstants;
  // The number of firings for each reaction during a step.
  std::vector<double> _reactionFirings;
  DiscreteUniformGenerator _discreteUniformGenerator;
  NormalGenerator _normalGenerator;
  PoissonGenerator _poissonGenerator;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TauLeapingSal();
  //! Copy constructor not implemented.
  TauLeapingSal(const TauLeapingSal&);
  //! Assignment operator not implemented.
  TauLeapingSal&
  operator=(const TauLeapingSal&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  TauLeapingSal(const State& state,
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

  //! Simulate until the termination condition is reached.
  void
  simulateLinear(const double epsilon, const double endTime)
  {
    setError(epsilon);
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(&TauLeapingSal::stepLinear, epsilon, endTime)) {
    }
  }

  //! Simulate with fixed size steps until the termination condition is reached.
  void
  simulateFixedLinear(const double tau, const double endTime)
  {
    setError(0.01);
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (stepFixed(&TauLeapingSal::stepLinear, tau, endTime)) {
    }
  }

  //! Try to take a step. Return true if a step is taken.
  bool
  step(MemberFunctionPointer method, double epsilon, double endTime);


  //! Try to take a step. Return true if a step is taken.
  bool
  stepFixed(MemberFunctionPointer method, double tau, double endTime);

private:

  void
  fixNegativePopulations(double tau);

  void
  setError(const double error)
  {
    // The relative error in the mean is less than 0.1 * error.
    // continuityError / mean < 0.1 * error
    // 1 / mean < 0.1 * error
    // mean > 10 / error
    const double t = 10. / error;
    _poissonGenerator.setNormalThreshhold(t);
    // The relative error in neglecting the standard deviation is less
    // than 0.1 * error.
    // sqrt(mean) / mean < 0.1 * error
    // mean > 100 / error^2
    _poissonGenerator.setSureThreshhold(t * t);
  }

  void
  stepLinear(double tau);

  void
  computePropensities()
  {
    computePropensities(_state.getPopulations());
  }

  void
  computePropensities(const std::vector<double>& populations);

  double
  computeTau(double epsilon);

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

#define __stochastic_TauLeapingSal_ipp__
#include "stlib/stochastic/TauLeapingSal.ipp"
#undef __stochastic_TauLeapingSal_ipp__

#endif
