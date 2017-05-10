// -*- C++ -*-

/*!
  \file stochastic/ReactionPriorityQueue.h
  \brief A reaction.
*/

#if !defined(__stochastic_ReactionPriorityQueue_h__)
#define __stochastic_ReactionPriorityQueue_h__

#include "stlib/ads/indexedPriorityQueue/IndexedPriorityQueueLinearSearch.h"
#include "stlib/numerical/random/exponential/Default.h"

#include <vector>

namespace stlib
{
namespace stochastic
{

//! The reaction priority queue.
/*!
  Recycling the reaction times is faster than generating new times.
*/
template < class _IndexedPriorityQueue =
           ads::IndexedPriorityQueueLinearSearch<>,
           class _ExponentialGenerator = numerical::EXPONENTIAL_GENERATOR_DEFAULT<>,
           bool _Recycle = true >
class ReactionPriorityQueue :
  private _IndexedPriorityQueue
{
  //
  // Private types.
  //
private:

  typedef _IndexedPriorityQueue Base;

  //
  // Enumerations.
  //
public:

  //! Whether to recycle the reaction times, and whether to use the propensities.
  enum {Recycle = _Recycle, UsesPropensities = Base::UsesPropensities};

  //
  // Public types.
  //
public:

  //! The discrete uniform random number generator.
  typedef typename _ExponentialGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;

  //
  // Member data.
  //
private:

  _ExponentialGenerator _exponential;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  ReactionPriorityQueue();
  //! Copy constructor not implemented.
  ReactionPriorityQueue(const ReactionPriorityQueue&);
  //! Assignment operator not implemented.
  const ReactionPriorityQueue&
  operator=(const ReactionPriorityQueue&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the size and the uniform generator.
  ReactionPriorityQueue(const std::size_t size,
                        DiscreteUniformGenerator* uniform) :
    Base(size),
    _exponential(uniform) {}

  //! Construct.  The base class uses hashing.
  ReactionPriorityQueue(const std::size_t size,
                        DiscreteUniformGenerator* uniform,
                        const std::size_t hashTableSize,
                        const double targetLoad) :
    Base(size, hashTableSize, targetLoad),
    _exponential(uniform) {}

  //! Store a pointer to the propensities in the indexed priority queue.
  void
  setPropensities(const std::vector<double>* propensities)
  {
    Base::setPropensities(propensities);
  }

  // Use the default destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the key of the specified element.
  using Base::get;

  //! Return the index of the top element.
  using Base::top;

  //! Return a const reference to the discrete, uniform generator.
  const DiscreteUniformGenerator&
  getDiscreteUniformGenerator() const
  {
    return *_exponential.getDiscreteUniformGenerator();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  using Base::popTop;

  //! Start a new time epoch by subtracting the specified time offset.
  void
  startNewEpoch(const double timeOffset)
  {
    Base::shift(- timeOffset);
  }

  //! Push the value into the queue.
  void
  push(const std::size_t index, const double time, const double propensity)
  {
#ifdef STLIB_DEBUG
    assert(propensity != 0);
#endif
    Base::push(index, time + _exponential() / propensity);
  }

  //! Push the top value into the queue.
  void
  pushTop(const double time, const double propensity)
  {
    if (propensity != 0) {
      Base::pushTop(time + _exponential() / propensity);
    }
    else {
      popTop();
    }
  }

  //! Push the top value into the queue using the inverse propensity.
  void
  pushTopInverse(const double time, const double inversePropensity)
  {
    Base::pushTop(time + _exponential() * inversePropensity);
  }

  //! Update the value in the queue.
  void
  update(const std::size_t index, const double time, const double oldPropensity,
         const double newPropensity)
  {
    update(index, time, oldPropensity, newPropensity,
           std::integral_constant<bool, Recycle>());
  }

  //! Clear the queue.
  using Base::clear;

  //! Set the constant used to balance costs.
  void
  setCostConstant(const double costConstant)
  {
    Base::setCostConstant(costConstant);
  }

private:

  //! Update the value in the queue.
  void
  update(const std::size_t index, const double time, const double oldPropensity,
         const double newPropensity, std::true_type /*Recycle*/)
  {
    if (oldPropensity != newPropensity) {
      if (oldPropensity == 0) {
        push(index, time, newPropensity);
      }
      else if (newPropensity == 0) {
        Base::pop(index);
      }
      else {
#ifdef STLIB_DEBUG
        assert(Base::get(index) >= time);
#endif
        Base::set(index, time + (oldPropensity / newPropensity) *
                  (Base::get(index) - time));
      }
    }
  }

  //! Update the value in the queue.
  void
  update(const std::size_t index, const double time, const double oldPropensity,
         const double newPropensity, std::false_type /*Recycle*/)
  {
    if (oldPropensity != newPropensity) {
      if (oldPropensity == 0) {
        push(index, time, newPropensity);
      }
      else if (newPropensity == 0) {
        Base::pop(index);
      }
      else {
        Base::set(index, time + _exponential() / newPropensity);
      }
    }
  }

  //@}
};

} // namespace stochastic
}

#endif
