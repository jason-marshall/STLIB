// -*- C++ -*-

/*!
  \file stochastic/Solver.h
  \brief Base class for solvers.
*/

#if !defined(__stochastic_Solver_h__)
#define __stochastic_Solver_h__

#include "stlib/stochastic/State.h"

#include <string>
#include <sstream>

namespace stlib
{
namespace stochastic
{

//! Base class for solvers.
/*!
  \param _PropensitiesFunctor Can calculate propensities as a function of the
  reaction index and the populations.

  Store the state, step count, maximum allowed steps,
  and an error message. The step count is stored as a double precision
  floating point value. This is because the integer type may be insufficient
  on 32-bit systems.
*/
class Solver
{
  //
  // Member data.
  //
protected:

  //! The state of the system.
  State _state;
  //! The number of steps.
  /*! Use double to count steps. On 32-bit architectures std::size_t may be
    insuffucient for some simulations. */
  double _stepCount;
  //! The maximum allow number of steps.
  double _maxSteps;
  //! String to record errors.
  std::string _error;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  Solver();
  //! Copy constructor not implemented.
  Solver(const Solver&);
  //! Assignment operator not implemented.
  Solver&
  operator=(const Solver&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  Solver(const State& state, const double maxSteps) :
    // Copy.
    _state(state),
    _stepCount(0),
    _maxSteps(maxSteps),
    _error()
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Initialize the state with the initial populations and reset the time.
  void
  initialize(const std::vector<double>& populations)
  {
    // Initialize the state.
    _state.setPopulations(populations);
    _state.resetReactionCounts();
    _stepCount = 0;
    _error.clear();
  }

  //! Increment the step count. Return true if successful. If not, set an error condition.
  bool
  incrementStepCount()
  {
    ++_stepCount;
    return _stepCount <= _maxSteps;
  }

protected:

  //! Record a step count error message.
  void
  setStepCountError()
  {
    std::ostringstream out;
    out << "The maximum step count " << _maxSteps << " was reached.";
    _error += out.str();
  }

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

  //! Return the step count.
  double
  getStepCount() const
  {
    return _stepCount;
  }

  //! Return the error string.
  const std::string&
  getError() const
  {
    return _error;
  }

  //@}
};

//@}

} // namespace stochastic
}

#endif
