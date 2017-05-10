// -*- C++ -*-

/*!
  \file stochastic/TerminationCondition.h
  \brief Termination conditions for stochastic simulations.
*/

#if !defined(__stochastic_TerminationCondition_h__)
#define __stochastic_TerminationCondition_h__

namespace stlib
{
namespace stochastic
{

//! Terminate when the end time is reached or when a specified number of step have been taken.
class TerminationCondition
{
  //
  // Member data.
  //
protected:

  //! The end time.
  double _endTime;
  //! The maximum allowed number of steps.
  std::size_t _stepLimit;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TerminationCondition();
  //! Assignment operator not implemented.
  TerminationCondition&
  operator=(const TerminationCondition&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time and the step limit.
  TerminationCondition(const double endTime, const std::size_t stepLimit) :
    _endTime(endTime),
    _stepLimit(stepLimit) {}

  //! Copy constructor.
  TerminationCondition(const TerminationCondition& other) :
    _endTime(other._endTime),
    _stepLimit(other._stepLimit)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \return True if firing the next reaction would exceed the end time or the
    step count limit.
  */
  bool
  operator()(const double timeOfNextReaction, const std::size_t stepCount)
  const
  {
    return timeOfNextReaction >= _endTime || stepCount >= _stepLimit;
  }

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \return True if firing the next reaction would exceed the end time or the
    step count limit.
  */
  bool
  operator()(const double time, const double tau, const std::size_t stepCount)
  const
  {
    return tau == std::numeric_limits<double>::max() ||
           time + tau >= _endTime || stepCount >= _stepLimit;
  }

  //! Return the end time.
  double
  getEndTime() const
  {
    return _endTime;
  }

  //! Return the step limit.
  std::size_t
  getStepLimit() const
  {
    return _stepLimit;
  }

  //@}
};


// CONTINUE
#if 0
//! Terminate when the end time is reached or when a specified number of steps have been taken.
class TerminationConditionTimeStep : public TerminationCondition
{
  //
  // Private types.
  //
private:

  typedef TerminationCondition Base;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TerminationConditionTimeStep();
  //! Assignment operator not implemented.
  TerminationConditionTimeStep&
  operator=(const TerminationConditionTimeStep&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time and the step limit.
  TerminationConditionTimeStep(const double endTime,
                               const std::size_t stepLimit) :
    Base(endTime, stepLimit) {}

  //! Copy constructor.
  TerminationConditionTimeStep(const TerminationConditionTimeStep& other) :
    Base(other) {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \param time The current time.
    \param timeToNextReaction The time increment to the next reaction.
    \param stepCount The current step count.

    \return True if firing the next reaction would exceed the end time or the
    step count limit.
  */
  template<class _State>
  bool
  operator()(const double time, const double timeToNextReaction,
             const std::size_t stepCount) const
  {
    return time + timeToNextReaction >= Base::_endTime ||
           stepCount >= Base::_stepLimit;
  }

  //@}
};

//! Terminate when the end time is reached or when a specified number of steps have been taken.
class TerminationConditionEndTime : public TerminationCondition
{
  //
  // Private types.
  //
private:

  typedef TerminationCondition Base;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TerminationConditionEndTime();
  //! Assignment operator not implemented.
  TerminationConditionEndTime&
  operator=(const TerminationConditionEndTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time and the step limit.
  TerminationConditionEndTime(const double endTime,
                              const std::size_t stepLimit) :
    Base(endTime, stepLimit) {}

  //! Copy constructor.
  TerminationConditionEndTime(const TerminationConditionEndTime& other) :
    Base(other) {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \return True if firing the next reaction would exceed the end time or the
    reaction count limit.
  */
  template<class _State>
  bool
  operator()(const double timeOfNextReaction, const std::size_t stepCount)
  const
  {
    return timeOfNextReaction >= Base::_endTime ||
           stepCount >= Base::_stepLimit;
  }

  //@}
};

//! Terminate when the end time is reached or when a specified number of reactions have fired.
class TerminationConditionCurrentTime : public TerminationCondition
{
  //
  // Private types.
  //
private:

  typedef TerminationCondition Base;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TerminationConditionCurrentTime();
  //! Assignment operator not implemented.
  TerminationConditionCurrentTime&
  operator=(const TerminationConditionCurrentTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time and the reaction limit.
  TerminationConditionCurrentTime(const double endTime,
                                  const std::size_t stepLimit) :
    Base(endTime, stepLimit) {}

  //! Copy constructor.
  TerminationConditionCurrentTime
  (const TerminationConditionCurrentTime& other) :
    Base(other) {}

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate now.
  /*!
    \param state The current state.
  */
  template<class _State>
  bool
  operator()(const _State& state) const
  {
    return state.getTimeOffset() >= Base::_endTimeOffset ||
           state.getReactionCount() >= Base::_stepLimit;
  }

  //! Return true if the simulation should terminate now.
  /*!
    \param state The current state.
    \param time The current time.
  */
  template<class _State>
  bool
  operator()(const _State& state, const double time) const
  {
    return time >= Base::_endTimeOffset ||
           state.getReactionCount() >= Base::_stepLimit;
  }

  //@}
};
#endif

} // namespace stochastic
}

#endif
