// -*- C++ -*-
// Propensities.h

#if !defined(__Propensities_h__)
#define __Propensities_h__

#include <cstddef>
#include <cmath>

#include "stochastic/ReactionSet.h"

// Compute custom propensities.
template<bool _IsDiscrete>
class Propensities :
  public stochastic::ReactionSet<_IsDiscrete> {
private:

  // The number of reactions.
#include "PropensitiesNumberOfReactions.ipp"

  //
  // Public types.
  //
public:

  // The population number type.
  typedef double PopulationType;
  // The number type.
  typedef double Number;
  //! A set of reactions.
  typedef stochastic::ReactionSet<_IsDiscrete> ReactionSetType;
  //! The reaction type.
  typedef typename ReactionSetType::ReactionType ReactionType;
  // The result type.
  typedef Number result_type;

  //
  // Private types.
  //
private:

  //
  // Private types.
  //
private:

  // The base class.
  typedef ReactionSetType Base;
  // A pointer to a member function that computes a single propensity.
  typedef Number (Propensities::* PropensityMember)
  (const PopulationType*) const;

  //
  // Member data.
  //
private:

  PropensityMember _propensityFunctions[NumberOfReactions];
  
  //
  // Not implemented.
  //
private:  

  // Default constructor not implemented.
  Propensities();
  // Assignment operator not implemented.
  Propensities&
  operator=(const Propensities&);

  //--------------------------------------------------------------------------
  // Constructors etc.
public:

  // Constructor.
#include "PropensitiesConstructor.ipp"

  // Copy constructor.
  Propensities(const Propensities& other) :
    Base(other) {
    for (std::size_t i = 0; i != NumberOfReactions; ++i) {
      _propensityFunctions[i] = other._propensityFunctions[i];
    }
  }

  // Destructor.
  ~Propensities() 
  {}

  //--------------------------------------------------------------------------
  // Functor.
public:

  using Base::getSize;
  using Base::getReaction;

  // Return the specified propensity function.
  template<typename _Container>
  result_type
  operator()(const std::size_t n, const _Container& populations) const {
    return (this->*_propensityFunctions[n])(&populations[0]);
  }

  //--------------------------------------------------------------------------
  // Compute propensities.
private:

#include "PropensitiesMemberFunctions.ipp"

};

#endif
