// -*- C++ -*-

/*!
  \file stochastic/InhomogeneousDirect.h
  \brief Base class for the direct method on time inhomogeneous problems.
*/

#if !defined(__stochastic_InhomogeneousDirect_h__)
#define __stochastic_InhomogeneousDirect_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/PropensitiesInhomogeneous.h"
#include "stlib/stochastic/TimeEpochOffset.h"

#include "stlib/numerical/random/exponential/ExponentialGeneratorZiggurat.h"
#include "stlib/numerical/random/discrete/linearSearch.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

namespace stlib
{
namespace stochastic
{

//! Base class for the direct method on time inhomogeneous problems.
class InhomogeneousDirect : public Solver
{
protected:

  //! The base class.
  typedef Solver Base;

  //
  // Protected types.
  //
protected:

  //! The propensities functor.
  typedef PropensitiesInhomogeneous<true> PropensitiesFunctor;
  //! The exponential generator.
  typedef numerical::ExponentialGeneratorZiggurat<> ExponentialGenerator;
  //! The discrete, uniform generator.
  typedef ExponentialGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;
  //! The continuous uniform generator.
  typedef numerical::ContinuousUniformGeneratorClosed<DiscreteUniformGenerator>
  ContinuousUniformGenerator;

  //
  // Public types.
  //
public:

  //! A set of reactions.
  typedef PropensitiesFunctor::ReactionSet ReactionSet;

  //
  // Member data.
  //
protected:

  //! The time is tracked with an epoch and offset.
  TimeEpochOffset _time;
  //! The propensities functor.
  PropensitiesFunctor _propensitiesFunctor;
  //! The discrete uniform random number generator.
  DiscreteUniformGenerator _discreteUniformGenerator;
  //! The exponential random number generator.
  ExponentialGenerator _exponentialGenerator;
  //! The continuous uniform generator.
  ContinuousUniformGenerator _continuousUniformGenerator;
  //! The time step.
  double _tau;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  InhomogeneousDirect();
  //! Copy constructor not implemented.
  InhomogeneousDirect(const InhomogeneousDirect&);
  //! Assignment operator not implemented.
  InhomogeneousDirect&
  operator=(const InhomogeneousDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  InhomogeneousDirect(const State& state, const ReactionSet& reactionSet,
                      const double maxSteps) :
    Base(state, maxSteps),
    _time(),
    // Copy.
    _propensitiesFunctor(reactionSet),
    // Construct.
    _discreteUniformGenerator(),
    _exponentialGenerator(&_discreteUniformGenerator),
    _continuousUniformGenerator(&_discreteUniformGenerator),
    // Initialize to an invalid value.
    _tau(std::numeric_limits<double>::max())
  {
  }

  // Default destructor is fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and reset the time.
  void
  initialize(const std::vector<double>& populations, const double time)
  {
    // Initialize the state.
    Base::initialize(populations);
    // Set the starting time.
    _time = time;
    // Compute the propensities at the starting time.
    computePropensities();
    // Compute the initial time step. Note that only the methods which collect
    // time series data use this value, the methods that collect histogram
    // data do not. This is here so that all methods can call a common
    // initialization function.
    _tau = computeTau();
  }

  std::size_t
  generateDiscreteDeviate()
  {
    const std::vector<double>& pmf = _propensitiesFunctor.propensities();
#ifdef STLIB_DEBUG
    assert(_propensitiesFunctor.sum() > 0);
#endif
    // Draw the deviate.
    std::size_t index;
    do {
      index = numerical::linearSearchChopDownUnguarded
              (pmf.begin(), pmf.end(),
               _continuousUniformGenerator() * _propensitiesFunctor.sum());
    }
    while (pmf[index] == 0);
    // Return the event.
    return index;
  }

  //! Simulate until the end time is reached.
  void
  simulate(const double endTime)
  {
    // Step until no more reactions can fire or we reach the end time.
    while (step(endTime)) {
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  bool
  step(const double endTime)
  {
    // If we have reached the end time.
    if (_time + _tau >= endTime) {
      return false;
    }

    // Check that we have not exceeded the allowed number of steps.
    if (! incrementStepCount()) {
      setStepCountError();
      return false;
    }

    // Advance the time.
    _time += _tau;
    // Recompute the propensities for the new time.
    computePropensities();
    // Fire a reaction if possible.
    if (_propensitiesFunctor.sum() > 0) {
      _state.fireReaction(generateDiscreteDeviate());
      // Recompute the propensities for the new populations.
      computePropensities();
    }
    // Compute the next time step.
    _tau = computeTau();
    return true;
  }

protected:

  //! Record a step count error message. Record the current time.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    Base::setStepCountError();
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached. "
        << " Time = " << _time << ".";
    _error += out.str();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return _discreteUniformGenerator;
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

  //! Compute the propensities using the current state and time.
  void
  computePropensities()
  {
    _propensitiesFunctor.set(_state.getPopulations(), _time);
  }

protected:

  //! Compute the time to the next reaction.
  double
  computeTau()
  {
    // If any reactions can fire.
    if (_propensitiesFunctor.sum() != 0) {
      const double mean = 1.0 / _propensitiesFunctor.sum();
      _time.updateEpoch(mean);
      // Return the time to the next reaction.
      return mean * _exponentialGenerator();
    }
    // Otherwise return infinity.
    return std::numeric_limits<double>::max();
  }

  // CONTINUE REMOVE
#if 0
  // Compute the time to the next reaction.
  double
  computeInitialTau()
  {
    // If any reactions can fire.
    if (_propensitiesFunctor.sum() != 0) {
      return _exponentialGenerator() / _propensitiesFunctor.sum();
    }
    // Otherwise return infinity.
    return std::numeric_limits<double>::max();
  }
#endif

  // CONTINUE REMOVE
#if 0
  // Compute the time to the next reaction.
  template<typename _TerminationCondition>
  double
  computeTau(const _TerminationCondition& terminationCondition)
  {
    // The maximum allowed ratio of the time offset to the mean time step
    // is 2^(53-32).
    const double maxRatio = 2097152;
    // If any reactions can fire.
    if (_propensitiesFunctor.sum() != 0) {
      const double mean = 1.0 / _propensitiesFunctor.sum();
      // If we will start losing random bits by adding to the time offset.
      if (mean * maxRatio < _state.getTimeOffset()) {
        // Start a new time epoch.
        _state.startNewEpoch();
        terminationCondition.startNewEpoch(_state.getTimeEpoch());
      }
      // Return the time to the next reaction.
      return mean * _exponentialGenerator();
    }
    // Otherwise return infinity.
    return std::numeric_limits<double>::max();
  }

  //! Compute the propensities at the specified time.
  void
  computePropensities(const double t)
  {
    _propensitiesFunctor.set(_state.getPopulations(), t);
  }
#endif

  //@}
};

//@}

} // namespace stochastic
}

#endif
