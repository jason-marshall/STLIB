// -*- C++ -*-

/*!
  \file stochastic/TauLeaping.h
  \brief The tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeaping_h__)
#define __stochastic_TauLeaping_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueBinaryHeap.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"
#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.h"

#include <set>

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using the tau-leaping method.
/*!
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
  \param _CorrectNegativePopulations If true, negative populations will be
  corrected. Otherwise, the simulation will fail if a species population
  becomes negative.
*/
template < class _PropensitiesFunctor = PropensitiesSingle<true>,
           bool _CorrectNegativePopulations = true >
class TauLeaping : public Solver
{
private:

  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;
  //! The Poisson generator.
  typedef numerical::PoissonGeneratorInvAcNormSure<> PoissonGenerator;
  //! The discrete uniform generator.
  typedef typename PoissonGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;
  //! The normal generator.
  typedef typename PoissonGenerator::NormalGenerator NormalGenerator;

  //
  // Private types.
  //
private:

  typedef void (TauLeaping::*MemberFunctionPointer)(double);

  //
  // Member data.
  //
private:

  PropensitiesFunctor _propensitiesFunctor;
  double _time;
  std::vector<double> _propensities;
  // The number of firings for each reaction during a step.
  std::vector<double> _reactionFirings;
  DiscreteUniformGenerator _discreteUniformGenerator;
  NormalGenerator _normalGenerator;
  PoissonGenerator _poissonGenerator;

  // The mean population change.
  std::vector<double> _mu;
  // The variance in the population change.
  std::vector<double> _sigmaSquared;
  std::vector<std::size_t> _highestOrder;
  std::vector<std::size_t> _highestIndividualOrder;
  //! Used for Runge-Kutta 4 and midpoint.
  std::vector<double> _p;
  //! Used for Runge-Kutta 4.
  std::vector<double> _k1, _k2, _k3, _k4;


  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TauLeaping();
  //! Copy constructor not implemented.
  TauLeaping(const TauLeaping&);
  //! Assignment operator not implemented.
  TauLeaping&
  operator=(const TauLeaping&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  TauLeaping(const State& state,
             const PropensitiesFunctor& propensitiesFunctor,
             double maxSteps);

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and set the time. Compute the orders for the species.
  /*!
    Let \c highestOrder be the highest order reaction in which the species
    appears.  Let \c highestIndividualOrder be the highest order of the
    species in a reaction.
    Suppose that the reactions are the following.  (We only use the reactants
    in computing the orders.)
    \f[
    x0 \rightarrow \cdots, \quad
    x1 + x2 \rightarrow \cdots, \quad
    x2 + 2 x3 \rightarrow \cdots, \quad
    3 x4 \rightarrow \cdots
    \f]
    Then the orders are the following.
    \verbatim
    highestOrder == {1, 2, 3, 3, 3}
    highestIndividualOrder == {1, 1, 1, 2, 3} \endverbatim
  */
  void
  initialize(const std::vector<double>& populations, double time);

  //! Simulate until the termination condition is reached.
  void
  simulateForward(const double epsilon, const double endTime)
  {
    setError(epsilon);
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(&TauLeaping::stepForward, epsilon, endTime)) {
    }
  }

  //! Simulate until the termination condition is reached.
  void
  simulateMidpoint(const double epsilon, const double endTime)
  {
    setError(epsilon);
    _p.resize(_state.getNumberOfSpecies());
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(&TauLeaping::stepMidpoint, epsilon, endTime)) {
    }
  }

  //! Simulate until the termination condition is reached.
  void
  simulateRungeKutta4(const double epsilon, const double endTime)
  {
    setError(epsilon);
    _p.resize(_state.getNumberOfSpecies());
    _k1.resize(_state.getNumberOfReactions());
    _k2.resize(_state.getNumberOfReactions());
    _k3.resize(_state.getNumberOfReactions());
    _k4.resize(_state.getNumberOfReactions());
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(&TauLeaping::stepRungeKutta4, epsilon, endTime)) {
    }
  }

  //! Simulate with fixed size steps until the termination condition is reached.
  void
  simulateFixedForward(const double tau, const double endTime)
  {
    setError(0.01);
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (stepFixed(&TauLeaping::stepForward, tau, endTime)) {
    }
  }

  //! Simulate with fixed size steps until the termination condition is reached.
  void
  simulateFixedMidpoint(const double tau, const double endTime)
  {
    setError(0.01);
    _p.resize(_state.getNumberOfSpecies());
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (stepFixed(&TauLeaping::stepMidpoint, tau, endTime)) {
    }
  }

  //! Simulate with fixed size steps until the termination condition is reached.
  void
  simulateFixedRungeKutta4(const double tau, const double endTime)
  {
    setError(0.01);
    _p.resize(_state.getNumberOfSpecies());
    _k1.resize(_state.getNumberOfReactions());
    _k2.resize(_state.getNumberOfReactions());
    _k3.resize(_state.getNumberOfReactions());
    _k4.resize(_state.getNumberOfReactions());
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (stepFixed(&TauLeaping::stepRungeKutta4, tau, endTime)) {
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  bool
  step(MemberFunctionPointer method, double epsilon, double endTime);


  //! Try to take a step.  Return true if a step is taken.
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
  stepForward(double tau);

  void
  stepMidpoint(double tau);

  void
  stepRungeKutta4(double tau);

  void
  computePropensities()
  {
    computePropensities(_state.getPopulations());
  }

  void
  computePropensities(const std::vector<double>& populations);

  double
  computeTau(double epsilon);

  //! Compute mu and sigma squared.
  void
  computeMuAndSigmaSquared();

  //! Compute the g described in "Efficient step size selection for the tau-leaping simulation method".
  double
  computeG(std::size_t speciesIndex) const;

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

#define __stochastic_TauLeaping_ipp__
#include "stlib/stochastic/TauLeaping.ipp"
#undef __stochastic_TauLeaping_ipp__

#endif
