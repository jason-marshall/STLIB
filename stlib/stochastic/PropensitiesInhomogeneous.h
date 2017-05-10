// -*- C++ -*-
// PropensitiesInhomogeneous.h

#if !defined(__PropensitiesInhomogeneous_h__)
#define __PropensitiesInhomogeneous_h__

#include <limits>
#include <numeric>
#include <vector>

#include "stlib/stochastic/ReactionSet.h"

//! Compute the propensities.
void
computePropensities(std::vector<double>* propensities,
                    const std::vector<double>& populations, double t);

namespace stlib
{
namespace stochastic
{

//! Compute custom propensities.
template<bool _IsDiscrete>
class PropensitiesInhomogeneous :
  public stochastic::ReactionSet<_IsDiscrete>
{
  //
  // Public types.
  //
public:

  //! A set of reactions.
  typedef stochastic::ReactionSet<_IsDiscrete> ReactionSet;
  //! The reaction type.
  typedef typename ReactionSet::ReactionType Reaction;

  //
  // Private types.
  //
private:

  // The base class.
  typedef ReactionSet Base;

  //
  // Member data.
  //
private:

  std::vector<double> _propensities;
  double _sum;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  PropensitiesInhomogeneous();
  // Assignment operator not implemented.
  PropensitiesInhomogeneous&
  operator=(const PropensitiesInhomogeneous&);

  //--------------------------------------------------------------------------
  // Constructors etc.
public:

  //! Constructor.
  PropensitiesInhomogeneous(const ReactionSet& reactionSet) :
    Base(reactionSet),
    // Fill with invalid values.
    _propensities(reactionSet.getSize(), -1.),
    _sum(-std::numeric_limits<double>::max())
  {
  }

  // Default copy constructor and destructor are fine.

  //--------------------------------------------------------------------------
  // Functor.
public:

  using Base::getSize;
  using Base::getReaction;

  //! Return the vector of propensities.
  const std::vector<double>&
  propensities() const
  {
    return _propensities;
  }

  //! Return the sum of the propensities.
  double
  sum() const
  {
    return _sum;
  }

  //! Compute the propensities and their sum for the specified populations and time.
  void
  set(const std::vector<double>& populations, const double t)
  {
    computePropensities(&_propensities, populations, t);
    _sum = std::accumulate(_propensities.begin(), _propensities.end(), 0.);
  }

  //! Compute the propensities.
  void
  operator()(std::vector<double>* propensities,
             const std::vector<double>& populations, const double t)
  {
    computePropensities(propensities, populations, t);
  }
};

} // namespace stochastic
}

#endif
