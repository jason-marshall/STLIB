// -*- C++ -*-

/*!
  \file stochastic/InhomogeneousTimeSeriesAllReactionsDirect.h
  \brief Collect time series data with the direct method on time inhomogeneous problems.
*/

#if !defined(__stochastic_InhomogeneousTimeSeriesAllReactionsDirect_h__)
#define __stochastic_InhomogeneousTimeSeriesAllReactionsDirect_h__

#include "stlib/stochastic/InhomogeneousDirect.h"

namespace stlib
{
namespace stochastic
{

//! Collect time series data with the direct method on time inhomogeneous problems.
class InhomogeneousTimeSeriesAllReactionsDirect :
  public InhomogeneousDirect
{
  //
  // Private types.
  //
private:

  typedef InhomogeneousDirect Base;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  InhomogeneousTimeSeriesAllReactionsDirect();
  //! Copy constructor not implemented.
  InhomogeneousTimeSeriesAllReactionsDirect
  (const InhomogeneousTimeSeriesAllReactionsDirect&);
  //! Assignment operator not implemented.
  InhomogeneousTimeSeriesAllReactionsDirect&
  operator=(const InhomogeneousTimeSeriesAllReactionsDirect&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  InhomogeneousTimeSeriesAllReactionsDirect(const State& state,
      const ReactionSet& reactionSet,
      const double maxSteps) :
    Base(state, reactionSet, maxSteps)
  {
  }

  // Default destructor is fine.

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  using Base::simulate;

  //! Simulate until the end time is reached.
  /*!
    Record each reaction index and time.
  */
  template<typename _IntOutputIter, typename NumberOutputIter>
  void
  simulate(const double endTime, _IntOutputIter indices,
           NumberOutputIter times)
  {
    // Step until no more reactions can fire or we reach the termination
    // condition.
    while (step(endTime, indices, times)) {
    }
  }

  //! Try to take a step.  Return true if a step is taken.
  /*!
    Record the reaction index and time.
  */
  template<typename _IntOutputIter, typename NumberOutputIter>
  bool
  step(const double endTime, _IntOutputIter indices, NumberOutputIter times)
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
      const std::size_t reactionIndex = generateDiscreteDeviate();
      _state.fireReaction(reactionIndex);
      // Record the reaction index and time.
      *indices++ = reactionIndex;
      *times++ = _time;
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
};

//@}

} // namespace stochastic
}

#endif
