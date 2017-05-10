// -*- C++ -*-

/*!
  \file stochastic/HybridDirectTauLeaping.h
  \brief The hybrid direct/tau-leaping method for SSA.
*/

#if !defined(__stochastic_HybridDirectTauLeaping_h__)
#define __stochastic_HybridDirectTauLeaping_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/TauLeapingDynamic.h"

#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "stlib/numerical/random/discrete/DiscreteGeneratorDynamic.h"

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using the hybrid direct/tau-leaping method.
/*!
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _PropensitiesFunctor = PropensitiesSingle<true> >
class HybridDirectTauLeaping : public Solver
{
private:

  typedef Solver Base;

  //
  // Public types.
  //
public:

  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;
  //! The tau-leaping algorithm.
  typedef TauLeapingDynamic TauLeaping;

  //! The exponential generator.
  typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
  //! The discrete, uniform generator.
  typedef typename ExponentialGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;
  //! The discrete, finite generator.
  typedef numerical::DiscreteGeneratorDynamic<DiscreteUniformGenerator>
  DiscreteGenerator;

  //
  // Private types.
  //
private:

  typedef bool (HybridDirectTauLeaping::*MemberFunctionPointer)(double);

  //
  // Member data.
  //
private:

  // State.
  double _time;
  PropensitiesFunctor _propensitiesFunctor;
  std::vector<double> _propensities;

  // Random number generators.
  DiscreteUniformGenerator _discreteUniformGenerator;
  ExponentialGenerator _exponentialGenerator;
  DiscreteGenerator _discreteGenerator;

  // Direct method.
  double _exponentialDeviate;
  std::size_t _directStepCount;
  // Reactions that will exhaust their reactants in few steps must be in the
  // direct group.
  double _volatileLimit;

  // Tau-leaping.
  TauLeaping _tauLeaping;
  std::size_t _tauLeapingStepCount;
  //! Used for Runge-Kutta 4 and midpoint.
  std::vector<double> _p;
  //! Used for Runge-Kutta 4.
  std::vector<double> _k1, _k2, _k3, _k4;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  HybridDirectTauLeaping();
  //! Copy constructor not implemented.
  HybridDirectTauLeaping(const HybridDirectTauLeaping&);
  //! Assignment operator not implemented.
  HybridDirectTauLeaping&
  operator=(const HybridDirectTauLeaping&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  HybridDirectTauLeaping(const State& state,
                         const PropensitiesFunctor& propensitiesFunctor,
                         const double epsilon,
                         const double maxSteps) :
    Base(state, maxSteps),
    _time(std::numeric_limits<double>::max()),
    _propensitiesFunctor(propensitiesFunctor),
    _propensities(state.getNumberOfReactions()),
    // Random number generators.
    _discreteUniformGenerator(),
    _exponentialGenerator(&_discreteUniformGenerator),
    _discreteGenerator(&_discreteUniformGenerator),
    // Direct method.
    _exponentialDeviate(-1),
    _directStepCount(0),
    // CONTINUE: This is reasonable, but I should experiment a bit.
    _volatileLimit(1.0 / epsilon),
    // Tau-leaping.
    _tauLeaping(state, propensitiesFunctor, &_discreteUniformGenerator,
                epsilon),
    _tauLeapingStepCount(0),
    _p()
  {
  }

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
  simulateForward(const double endTime)
  {
    // Step until no more reactions can fire or we reach the end time.
    while (step(&HybridDirectTauLeaping::stepForward, endTime)) {
    }
  }

  //! Simulate until the termination condition is reached.
  void
  simulateMidpoint(const double endTime)
  {
    _p.resize(_state.getNumberOfSpecies());
    // Step until no more reactions can fire or we reach the end time.
    while (step(&HybridDirectTauLeaping::stepMidpoint, endTime)) {
    }
  }

  //! Simulate until the termination condition is reached.
  void
  simulateRungeKutta4(const double endTime)
  {
    _p.resize(_state.getNumberOfSpecies());
    _k1.resize(_state.getNumberOfReactions());
    _k2.resize(_state.getNumberOfReactions());
    _k3.resize(_state.getNumberOfReactions());
    _k4.resize(_state.getNumberOfReactions());
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(&HybridDirectTauLeaping::stepRungeKutta4, endTime)) {
    }
  }

private:

  //! Try to take a step.  Return true if a step is taken.
  /*! Multiply _propensities by the step, tau. */
  bool
  step(MemberFunctionPointer method, double endTime);

  bool
  stepForward(double tau);

  bool
  stepMidpoint(double tau);

  bool
  stepRungeKutta4(double tau);

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

  //! Get the number of direct steps.
  std::size_t
  getDirectCount() const
  {
    return _directStepCount;
  }

  //! Get the number of tau-leaping steps.
  std::size_t
  getTauLeapingCount() const
  {
    return _tauLeapingStepCount;
  }

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return _discreteUniformGenerator;
  }

private:

  // Return true if the reaction is volatile.
  bool
  isVolatile(std::size_t index);

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

private:

  // Move the volatile and slow reactions to the direct group.
  void
  moveVolatileAndSlowToDirect(double minimumPropensity);

  // Compute the sum of the PMF for the direct group reactions.
  double
  computeDirectPmfSum()
  {
    const std::vector<std::size_t>& events =
      _discreteGenerator.getEvents();
    double sum = 0;
    for (std::size_t i = 0; i != events.size(); ++i) {
      sum += _propensities[events[i]];
    }
    return sum;
  }

  // CONTINUE
#if 0
  // Compute the time to the next direct reaction.
  double
  computeDirectStep()
  {
    double pmfSum = 0;
    for (std::size_t i = 0; i != _discreteGenerator.getEvents().size();
         ++i) {
      pmfSum += _propensities[_discreteGenerator.getEvents()[i]];
    }
    // If any reactions can fire.
    if (pmfSum != 0) {
      // Return the time to the next reaction.
      return _exponentialGenerator() / pmfSum;
    }
    // Otherwise return infinity.
    return std::numeric_limits<double>::max();
  }
#endif

  void
  computePropensities()
  {
    computePropensities(_state.getPopulations());
  }

  void
  computePropensities(const std::vector<double>& populations)
  {
    for (std::size_t m = 0; m < _propensities.size(); ++m) {
      _propensities[m] = _propensitiesFunctor(m, populations);
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name I/O.
  //@{
public:

  void
  print(std::ostream& out) const
  {
    _state.print(out);
    out << "Propensities:\n";
    for (std::size_t i = 0; i != _propensities.size(); ++i) {
      out << _propensities[i] << ' ';
    }
    out << '\n';
    out << "Exponential deviate = " << _exponentialDeviate << '\n';
    out << "Direct step count = " << _directStepCount << '\n';
    out << "Volatile limit = " << _volatileLimit << '\n';
    _tauLeaping.print(out);
    out << "Tau-leaping step count = " << _tauLeapingStepCount << '\n';
  }

  //@}
};

//@}

} // namespace stochastic
}

#define __stochastic_HybridDirectTauLeaping_ipp__
#include "stlib/stochastic/HybridDirectTauLeaping.ipp"
#undef __stochastic_HybridDirectTauLeaping_ipp__

#endif
