// -*- C++ -*-

/*!
  \file stochastic/PropensityTimeDerivatives.h
  \brief Derivatives of the propensity functions with respect to time.
*/

#if !defined(__stochastic_PropensityTimeDerivatives_h__)
#define __stochastic_PropensityTimeDerivatives_h__

#include "stlib/stochastic/State.h"
#include "stlib/stochastic/ReactionSet.h"

namespace stlib
{
namespace stochastic
{


//! Derivatives of the propensity functions with respect to time.
/*! This is implemented as a functor so that we don't have to allocate
  the vector of population derivatives each time. */
class PropensityTimeDerivatives
{
  //
  // Member data.
  //
private:

  //! The derivatives of the populations with respect to time.
  std::vector<double> _dxdt;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Construct from the number of species.
  PropensityTimeDerivatives(const std::size_t numberOfSpecies) :
    _dxdt(numberOfSpecies)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  //@{
public:

  //! Compute the first derivatives of the propensities w.r.t. time.
  /*!
   */
  void
  operator()(const State& state, const ReactionSet<true>& reactionSet,
             const std::vector<double>& propensities,
             std::vector<double>* first)
  {
#ifdef STLIB_DEBUG
    assert(first->size() == state.getNumberOfReactions());
    assert(reactionSet.getSize() == state.getNumberOfReactions());
#endif
    // The population time derivatives.
    state.populationDerivatives(propensities, &_dxdt);
    // The propensity time derivatives.
    for (std::size_t j = 0; j != reactionSet.getSize(); ++j) {
      (*first)[j] = reactionSet.getReaction(j).
                    timeDerivative(state.getPopulations(), _dxdt);
    }
  }

  //@}
};


} // namespace stochastic
}

#endif
