// -*- C++ -*-

/*!
  \file stochastic/reactionPropensityInfluence.h
  \brief Function to compute the reaction-propensity influence.
*/

#if !defined(__stochastic_reactionPropensityInfluence_h__)
#define __stochastic_reactionPropensityInfluence_h__

#include "stlib/stochastic/Reaction.h"

#include "stlib/container/StaticArrayOfArrays.h"

#include <set>
#include <map>

namespace stlib
{
namespace stochastic
{

//! Compute the reaction-propensity influence.
/*!
  \param numberOfSpecies The number of species.
  \param reactionsBeginning The beginning of the reactions.
  \param reactionsEnd The end of the reactions.
  \param influence For each reaction, this will list the reaction
  propensities that may be affected if the reaction fires.
  \param includeSelf Whether a reaction should include itself in the set
  of influenced reactions.
*/
template<typename ForwardIterator>
void
computeReactionPropensityInfluence
(std::size_t numberOfSpecies, ForwardIterator reactionsBeginning,
 ForwardIterator reactionsEnd,
 container::StaticArrayOfArrays<std::size_t>* influence, bool includeSelf);

} // namespace stochastic
}

#define __stochastic_reactionPropensityInfluence_ipp__
#include "stlib/stochastic/reactionPropensityInfluence.ipp"
#undef __stochastic_reactionPropensityInfluence_ipp__

#endif
