// -*- C++ -*-

/*!
  \file stochastic/FirstReactionAbsoluteTime.h
  \brief The first reaction method for SSA.
*/

#if !defined(__stochastic_FirstReactionAbsoluteTime_h__)
#define __stochastic_FirstReactionAbsoluteTime_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/TimeEpochOffset.h"

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
class FirstReactionAbsoluteTime : public Solver
{
private:

  typedef Solver Base;

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

  TimeEpochOffset _time;
  PropensitiesFunctor _propensitiesFunctor;
  DiscreteUniformGenerator _discreteUniformGenerator;
  ExponentialGenerator _exponentialGenerator;
  double _timeOfFirstReaction;
  std::size_t _indexOfFirstReaction;
  container::StaticArrayOfArrays<std::size_t> _reactionInfluence;
  std::vector<double> _reactionTimes;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  FirstReactionAbsoluteTime();
  //! Copy constructor not implemented.
  FirstReactionAbsoluteTime(const FirstReactionAbsoluteTime&);
  //! Assignment operator not implemented.
  FirstReactionAbsoluteTime&
  operator=(const FirstReactionAbsoluteTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  FirstReactionAbsoluteTime(const State& state,
                            const PropensitiesFunctor& propensitiesFunctor,
                            const container::StaticArrayOfArrays<std::size_t>&
                            reactionInfluence,
                            const double maxSteps) :
    Base(state, maxSteps),
    // Copy.
    _time(),
    _propensitiesFunctor(propensitiesFunctor),
    // Construct.
    _discreteUniformGenerator(),
    _exponentialGenerator(&_discreteUniformGenerator),
    // Invalid value.
    _timeOfFirstReaction(-1),
    _indexOfFirstReaction(-1),
    // Copy.
    _reactionInfluence(reactionInfluence),
    // Allocate.
    _reactionTimes(state.getNumberOfReactions())
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
    _time = time;

    // Compute each reaction time.
    for (std::size_t i = 0; i != _reactionTimes.size(); ++i) {
      const double p = _propensitiesFunctor(i, _state.getPopulations());
      if (p != 0) {
        _reactionTimes[i] = _time.getOffset() + _exponentialGenerator() / p;
      }
      else {
        _reactionTimes[i] = std::numeric_limits<double>::max();
      }
    }

    computeTimeOfFirstReaction();
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
  step(const double endTime);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the current time.
  double
  getTime() const
  {
    // Convert the time epoch and offset to a single time.
    return _time;
  }

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

private:

  void
  updateReactionTimes(std::size_t reactionIndex);

  void
  computeTimeOfFirstReaction()
  {
    // Start with infinity.
    _timeOfFirstReaction = std::numeric_limits<double>::max();
    _indexOfFirstReaction = 0;
    for (std::size_t i = 0; i != _reactionTimes.size(); ++i) {
      if (_reactionTimes[i] < _timeOfFirstReaction) {
        _timeOfFirstReaction = _reactionTimes[i];
        _indexOfFirstReaction = i;
      }
    }
  }

  //@}
};

//@}

} // namespace stochastic
}

#define __stochastic_FirstReactionAbsoluteTime_ipp__
#include "stlib/stochastic/FirstReactionAbsoluteTime.ipp"
#undef __stochastic_FirstReactionAbsoluteTime_ipp__

#endif
