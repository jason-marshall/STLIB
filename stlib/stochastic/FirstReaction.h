// -*- C++ -*-

/*!
  \file stochastic/FirstReaction.h
  \brief The first reaction method for SSA.
*/

#if !defined(__stochastic_FirstReaction_h__)
#define __stochastic_FirstReaction_h__

#include "stlib/stochastic/Solver.h"
#include "stlib/stochastic/Propensities.h"
#include "stlib/stochastic/TimeEpochOffset.h"

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
class FirstReaction : public Solver
{
protected:

  //! The base class.
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
protected:

  //! The time is tracked with an epoch and offset.
  TimeEpochOffset _time;
  //! The propensities functor.
  PropensitiesFunctor _propensitiesFunctor;
  //! The discrete uniform random number generator.
  DiscreteUniformGenerator _discreteUniformGenerator;
  //! The exponential deviate generator.
  ExponentialGenerator _exponentialGenerator;
  //! The time to the first reaction.
  double _timeToFirstReaction;
  //! The index of the first reaction.
  std::size_t _indexOfFirstReaction;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  FirstReaction();
  //! Copy constructor not implemented.
  FirstReaction(const FirstReaction&);
  //! Assignment operator not implemented.
  FirstReaction&
  operator=(const FirstReaction&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  FirstReaction(const State& state,
                const PropensitiesFunctor& propensitiesFunctor,
                double maxSteps);

  // Default destructor is fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and time.
  void
  initialize(const std::vector<double>& populations, double time);

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

protected:

  //! Record a step count error message. Record the current time.
  /*! Override the same function from the base class. */
  void
  setStepCountError()
  {
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached at time = "
        << _time << ".";
    _error += out.str();
  }

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
  computeTimeToFirstReaction();

  //@}
};

//@}

} // namespace stochastic
}

#define __stochastic_FirstReaction_ipp__
#include "stlib/stochastic/FirstReaction.ipp"
#undef __stochastic_FirstReaction_ipp__

#endif
