// -*- C++ -*-

/*!
  \file stochastic/Propensities.h
  \brief Functors for computing propensities.
*/

#if !defined(__stochastic_Propensities_h__)
#define __stochastic_Propensities_h__

#include "stlib/stochastic/ReactionSet.h"

namespace stlib
{
namespace stochastic
{

//! Functor for computing a single propensity.
/*!
  \note This functor is expensive to copy.
*/
template<bool _IsDiscrete>
class PropensitiesSingle :
  public ReactionSet<_IsDiscrete>
{
  //
  // Public types.
  //
public:

  //! A set of reactions.
  typedef ReactionSet<_IsDiscrete> ReactionSetType;
  //! The reaction type.
  typedef typename ReactionSetType::ReactionType ReactionType;
  //! The result type.
  typedef double result_type;

  //
  // Private types.
  //
private:

  typedef ReactionSetType Base;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  PropensitiesSingle();
  //! Assignment operator not implemented.
  PropensitiesSingle&
  operator=(const PropensitiesSingle&);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the set of reactions.
  PropensitiesSingle(const ReactionSetType& reactionSet) :
    Base(reactionSet)
  {
  }

  // Use the default copy constructor and destructor.

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  using Base::getSize;
  using Base::getReaction;

  //! Return the specified propensity function.
  template<typename Container>
  result_type
  operator()(const std::size_t n, const Container& populations) const
  {
    return Base::computePropensity(n, populations);
  }

  //@}
};


//! Functor for computing all of the propensities.
/*!
  \note This functor is expensive to copy.
*/
template<bool _IsDiscrete>
class PropensitiesAll :
  public ReactionSet<_IsDiscrete>
{
  //
  // Public types.
  //
public:

  //! A set of reactions.
  typedef ReactionSet<_IsDiscrete> ReactionSetType;
  //! The reaction type.
  typedef typename ReactionSetType::ReactionType ReactionType;
  //! The result type.
  typedef double result_type;

  //
  // Private types.
  //
private:

  typedef ReactionSetType Base;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  PropensitiesAll();
  //! Assignment operator not implemented.
  PropensitiesAll&
  operator=(const PropensitiesAll&);


  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the set of reactions.
  PropensitiesAll(const ReactionSetType& reactionSet) :
    Base(reactionSet)
  {
  }

  //! Copy constructor.
  PropensitiesAll(const PropensitiesAll& other) :
    Base(other)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  using Base::getSize;
  using Base::getReaction;

  //! Compute the propensity functions.
  template<typename _Container, typename _RandomAccessIterator>
  void
  operator()(const _Container& populations,
             _RandomAccessIterator propensities) const
  {
    for (std::size_t i = 0; i != Base::getSize(); ++i) {
      propensities[i] = Base::computePropensity(i, populations);
    }
  }

  //@}
};

} // namespace stochastic
}

#endif
