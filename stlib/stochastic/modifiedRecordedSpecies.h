// -*- C++ -*-

/*!
  \file stochastic/modifiedRecordedSpecies.h
  \brief Calculate the recorded species that are modified by each reaction.
*/

#if !defined(__stochastic_modifiedRecordedSpecies_h__)
#define __stochastic_modifiedRecordedSpecies_h__

#include "stlib/container/StaticArrayOfArrays.h"

#include <map>
#include <vector>

namespace stlib
{
namespace stochastic
{

//! Calculate the recorded species that are modified by each reaction.
template<typename _State>
inline
void
calculateModifiedRecordedSpecies
(const _State& state,
 const std::vector<std::size_t>& recordedSpecies,
 container::StaticArrayOfArrays<std::size_t>* modifiedRecordedSpecies)
{
  typedef std::map<std::size_t, std::size_t> Map;
  typedef Map::value_type value_type;
  typedef Map::const_iterator const_iterator;

  // The set of recorded species. We store pairs of species indices and
  // histogram indices.
  Map recorded;
  for (std::size_t i = 0; i != recordedSpecies.size(); ++i) {
    recorded.insert(value_type(recordedSpecies[i], i));
  }

  // The number of modified recorded species for each reaction.
  std::vector<std::size_t> sizes(state.getNumberOfReactions(), 0);
  // The indices of the recorded species (not the indices of the species).
  std::vector<std::size_t> indices;
  // The state change vectors.
  const typename _State::StateChangeVectors& scv =
    state.getStateChangeVectors();
  const_iterator r;
  // For each reaction.
  for (std::size_t i = 0; i != scv.getNumberOfArrays(); ++i) {
    // For each modified species.
    for (std::size_t j = 0; j != scv.size(i); ++j) {
      // Search for the species index.
      r = recorded.find(scv(i)[j].first);
      // If the species will be recorded.
      if (r != recorded.end()) {
        ++sizes[i];
        // The histogram index.
        indices.push_back(r->second);
      }
    }
  }
  modifiedRecordedSpecies->rebuild(sizes.begin(), sizes.end(),
                                   indices.begin(), indices.end());
}

} // namespace stochastic
}

#endif
