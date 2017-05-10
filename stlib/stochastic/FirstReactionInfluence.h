// -*- C++ -*-

/*!
  \file stochastic/FirstReactionInfluence.h
  \brief The first reaction method for SSA.
*/

#if !defined(__stochastic_FirstReactionInfluence_h__)
#define __stochastic_FirstReactionInfluence_h__

#include "stlib/stochastic/FirstReaction.h"

#include "stlib/container/StaticArrayOfArrays.h"

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using Gillespie's first reaction method.
/*!
  \param _ExponentialGenerator Random deviate generator for the exponential
  distribution. By default the ziggurat algorithm is used.
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.
*/
template < class _ExponentialGenerator =
           numerical::ExponentialGeneratorZiggurat<>,
           class _PropensitiesFunctor = PropensitiesSingle<true> >
class FirstReactionInfluence :
  public FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>
{
private:

  typedef FirstReaction<_ExponentialGenerator, _PropensitiesFunctor>
  Base;

  //
  // Public types.
  //
public:

  //! The exponential generator.
  typedef _ExponentialGenerator ExponentialGenerator;
  //! The propensities functor.
  typedef _PropensitiesFunctor PropensitiesFunctor;

  //! The discrete, uniform generator.
  typedef typename ExponentialGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  //
  // Member data.
  //
private:

  using Base::_state;
  using Base::_time;
  using Base::_propensitiesFunctor;
  //DiscreteUniformGenerator _discreteUniformGenerator;
  using Base::_exponentialGenerator;
  using Base::_timeToFirstReaction;
  using Base::_indexOfFirstReaction;

  container::StaticArrayOfArrays<std::size_t> _reactionInfluence;
  // Store the inverse propensities. This enables more efficient calculation
  // of the reaction times. (multiplication versus division)
  std::vector<double> _inversePropensities;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  FirstReactionInfluence();
  //! Copy constructor not implemented.
  FirstReactionInfluence(const FirstReactionInfluence&);
  //! Assignment operator not implemented.
  FirstReactionInfluence&
  operator=(const FirstReactionInfluence&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  FirstReactionInfluence(const State& state,
                         const PropensitiesFunctor& propensitiesFunctor,
                         const container::StaticArrayOfArrays<std::size_t>&
                         reactionInfluence,
                         const double maxSteps) :
    Base(state, propensitiesFunctor, maxSteps),
    // Copy.
    _reactionInfluence(reactionInfluence),
    // Allocate.
    _inversePropensities(state.getNumberOfReactions())
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
    Base::Base::initialize(populations);
    _time = time;

    // Compute the initial propensities and the initial time to the first
    // reaction.
    computePropensities();
    computeTimeToFirstReaction();
  }

  //! Simulate until the termination condition is reached.
  void
  simulate(const double endTime)
  {
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(endTime)) {
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  bool
  step(double endTime);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return a const reference to the state.
  using Base::getState;
  //! Return the current time.
  using Base::getTime;
  //! Return the step count.
  using Base::getStepCount;
  //! Return a const reference to the discrete, uniform generator.
  using Base::getDiscreteUniformGenerator;

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Return a reference to the discrete, uniform generator.
  //using Base::getDiscreteUniformGenerator;

private:

  double
  safelyInvert(const double x) const
  {
    if (x != 0) {
      return 1. / x;
    }
    else {
      return std::numeric_limits<double>::max();
    }
  }

  void
  computePropensities()
  {
    // Compute each propensity.
    for (std::size_t i = 0; i != _inversePropensities.size(); ++i) {
      _inversePropensities[i] =
        safelyInvert(_propensitiesFunctor(i, _state.getPopulations()));
    }
  }

  void
  updatePropensities(std::size_t reactionIndex);

  void
  computeTimeToFirstReaction()
  {
    // Start with infinity.
    _timeToFirstReaction = std::numeric_limits<double>::max();
    _indexOfFirstReaction = 0;
    double t;
    for (std::size_t i = 0; i != _inversePropensities.size(); ++i) {
      if (_inversePropensities[i] != std::numeric_limits<double>::max()) {
        t = _exponentialGenerator() * _inversePropensities[i];
        if (t < _timeToFirstReaction) {
          _timeToFirstReaction = t;
          _indexOfFirstReaction = i;
        }
      }
    }
  }

  //@}
};

//@}

} // namespace stochastic
}

#define __stochastic_FirstReactionInfluence_ipp__
#include "stlib/stochastic/FirstReactionInfluence.ipp"
#undef __stochastic_FirstReactionInfluence_ipp__

#endif
