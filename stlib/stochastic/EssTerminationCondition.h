// -*- C++ -*-

/*!
  \file stochastic/EssTerminationCondition.h
  \brief Termination conditions form exact stochastic simulations.
*/

#if !defined(__stochastic_EssTerminationCondition_h__)
#define __stochastic_EssTerminationCondition_h__

#include "stlib/stochastic/State.h"

#include <limits>

namespace stlib
{
namespace stochastic
{

//! Terminate when no more reactions can fire.
class EssTerminationConditionExhaust
{
  //
  // Public types.
  //
public:

  //
  // Default constructors will do.
  //

  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return false.
  /*!
    \param state The current state.

    \return false.
  */
  template<class _State>
  bool
  operator()(const _State& /*state*/) const
  {
    return false;
  }

  //! Return false.
  /*!
    \param state The current state.
    \param timeToNextReaction The time increment to the next reaction.

    \return false.
  */
  template<class _State>
  bool
  operator()(const _State& /*state*/, const double /*timeToNextReaction*/)
  const
  {
    return false;
  }

  //! Return infinity.
  double
  getEndTime() const
  {
    return std::numeric_limits<double>::max();
  }

  //@}
};




//! Terminate when the end time is reached.
class EssTerminationConditionEndTime
{
  //
  // Public types.
  //
public:

  //
  // Member data.
  //
private:

  double _endTime;
  double _endTimeOffset;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  EssTerminationConditionEndTime();
  //! Assignment operator not implemented.
  EssTerminationConditionEndTime&
  operator=(const EssTerminationConditionEndTime&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time.
  EssTerminationConditionEndTime(const double endTime) :
    _endTime(endTime),
    _endTimeOffset(endTime) {}

  //! Copy constructor.
  EssTerminationConditionEndTime(const EssTerminationConditionEndTime& other) :
    _endTime(other._endTime),
    _endTimeOffset(other._endTimeOffset) {}

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
    return state.getTimeOffset() >= _endTimeOffset;
  }

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \param state The current state.
    \param timeToNextReaction The time increment to the next reaction.

    \return True if firing the next reaction would exceed the end time for
    the simulation.
  */
  template<class _State>
  bool
  operator()(const _State& state, const double timeToNextReaction) const
  {
    return state.getTimeOffset() + timeToNextReaction > _endTimeOffset;
  }

  //! Return the end time.
  double
  getEndTime() const
  {
    return _endTime;
  }

  //! Start a new epoch.
  void
  startNewEpoch(const double timeEpoch)
  {
    _endTimeOffset = _endTime - timeEpoch;
  }

  //@}
};




//! Terminate when a specified number of reactions have fired.
class EssTerminationConditionReactionCount
{
  //
  // Public types.
  //
public:

  //
  // Member data.
  //
private:

  std::size_t _reactionLimit;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  EssTerminationConditionReactionCount();
  //! Assignment operator not implemented.
  EssTerminationConditionReactionCount&
  operator=(const EssTerminationConditionReactionCount&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the reaction limit.
  EssTerminationConditionReactionCount(const std::size_t reactionLimit) :
    _reactionLimit(reactionLimit) {}

  //! Copy constructor.
  EssTerminationConditionReactionCount
  (const EssTerminationConditionReactionCount& other) :
    _reactionLimit(other._reactionLimit)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate before the next step.
  /*!
    \param state The current state.

    \return True if taking another step would exceed the reaction count
    limit.
  */
  template<class _State>
  bool
  operator()(const _State& state) const
  {
    return state.getReactionCount() >= _reactionLimit;
  }

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \param state The current state.
    \param timeToNextReaction The time increment to the next reaction.

    \note The time to the next reaction is not used.  It is a function
    parameter for constistency with the other termination conditions.

    \return True if firing the next reaction would exceed the reaction count
    limit.
  */
  template<class _State>
  bool
  operator()(const _State& state, const double /*timeToNextReaction*/) const
  {
    return state.getReactionCount() >= _reactionLimit;
  }

  //! Return infinity.
  double
  getEndTime() const
  {
    return std::numeric_limits<double>::max();
  }

  //@}
};


//! Terminate when the end time is reached or when a specified number of reactions have fired.
class EssTerminationConditionEndTimeReactionCount
{
  //
  // Public types.
  //
public:

  //
  // Member data.
  //
private:

  double _endTime;
  mutable double _endTimeOffset;
  std::size_t _reactionLimit;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  EssTerminationConditionEndTimeReactionCount();
  //! Assignment operator not implemented.
  EssTerminationConditionEndTimeReactionCount&
  operator=(const EssTerminationConditionEndTimeReactionCount&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the end time and the reaction limit.
  EssTerminationConditionEndTimeReactionCount
  (const double endTime, const std::size_t reactionLimit) :
    _endTime(endTime),
    _endTimeOffset(endTime),
    _reactionLimit(reactionLimit) {}

  //! Copy constructor.
  EssTerminationConditionEndTimeReactionCount
  (const EssTerminationConditionEndTimeReactionCount& other) :
    _endTime(other._endTime),
    _endTimeOffset(other._endTimeOffset),
    _reactionLimit(other._reactionLimit)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Termination condition.
  //@{
public:

  //! Return true if the simulation should terminate before the next step.
  /*!
    \param state The current state.

    \return True if taking another step would exceed the end time or the
    reaction count limit.
  */
  template<class _State>
  bool
  operator()(const _State& state) const
  {
    return state.getTimeOffset() >= _endTimeOffset ||
           state.getReactionCount() >= _reactionLimit;
  }

  //! Return true if the simulation should terminate before the next reaction.
  /*!
    \param state The current state.
    \param timeToNextReaction The time increment to the next reaction.

    \return True if firing the next reaction would exceed the end time or the
    reaction count limit.
  */
  template<class _State>
  bool
  operator()(const _State& state, const double timeToNextReaction) const
  {
    return state.getTimeOffset() + timeToNextReaction > _endTimeOffset ||
           state.getReactionCount() >= _reactionLimit;
  }

  //! Return the end time.
  double
  getEndTime() const
  {
    return _endTime;
  }

  //! Start a new epoch.
  void
  startNewEpoch(const double timeEpoch) const
  {
    _endTimeOffset = _endTime - timeEpoch;
  }

  //@}
};


} // namespace stochastic
}

#endif
