// -*- C++ -*-

#if !defined(__stochastic_TauLeapingDynamic_ipp__)
#error This file is an implementation detail of TauLeapingDynamic.
#endif

namespace stlib
{
namespace stochastic
{

inline
void
TauLeapingDynamic::
initialize(const ReactionSetType& reactionSet)
{
  typedef ReactionSetType::ReactionType::SparseVectorSizeType
  SparseVectorSizeType;
  typedef SparseVectorSizeType::const_iterator const_iterator;

  //
  // Compute the orders for the species.
  //
  // Initialize the arrays.
  std::fill(_highestOrder.begin(), _highestOrder.end(), 0);
  std::fill(_highestIndividualOrder.begin(), _highestIndividualOrder.end(), 0);

  std::size_t sum;
  // Loop over the reactions.
  for (std::size_t n = 0; n != reactionSet.getSize(); ++n) {
    const SparseVectorSizeType& reactants =
      reactionSet.getReaction(n).getReactants();
    // The sum of the reactant coefficients.
    sum = container::sum(reactants);
    // Loop over the reactant species.
    for (const_iterator i = reactants.begin(); i != reactants.end(); ++i) {
      if (sum > _highestOrder[i->first]) {
        _highestOrder[i->first] = sum;
      }
      if (i->second > _highestIndividualOrder[i->first]) {
        _highestIndividualOrder[i->first] = i->second;
      }
    }
  }
  //
  // Store the reactants for each reaction.
  //
  std::vector<std::size_t> sizes, indices;
  // Loop over the reactions.
  for (std::size_t n = 0; n != reactionSet.getSize(); ++n) {
    const SparseVectorSizeType& reactants =
      reactionSet.getReaction(n).getReactants();
    sizes.push_back(reactants.size());
    for (const_iterator i = reactants.begin(); i != reactants.end(); ++i) {
      indices.push_back(i->first);
    }
  }
  _reactants.rebuild(sizes.begin(), sizes.end(), indices.begin(),
                     indices.end());
}

inline
double
TauLeapingDynamic::
computeStep(const StateChangeVectors& listOfStateChangeVectors,
            const std::vector<double>& propensities,
            const std::vector<double>& populations)
{
#ifdef STLIB_DEBUG
  assert(_epsilon > 0);
#endif

  // Compute mu and sigmaSquared.
  computeMuAndSigmaSquared(listOfStateChangeVectors, propensities);

  // Initialize tau to infinity.
  double tau = std::numeric_limits<double>::max();

  // Take the minimum over the active species.
  double numerator, temp, a, b;
  for (std::size_t i = 0; i != _activeSpecies.size(); ++i) {
    const std::size_t n = _activeSpecies[i];
#ifdef STLIB_DEBUG
    // CONTINUE
    assert(_highestOrder[n] != 0);
    assert(populations[n] > 0);
#endif
    // CONTINUE: Tau-leaping should not be used for slow reactions.
    //numerator = std::max(_epsilon * populations[n] / computeG(n), 1.0);
    numerator = _epsilon * populations[n] / computeG(n, populations[n]);

    if (_mu[n] != 0) {
      a = numerator / std::abs(_mu[n]);
    }
    else {
      a = std::numeric_limits<double>::max();
    }

    if (_sigmaSquared[n] != 0) {
      b = numerator * numerator / _sigmaSquared[n];
    }
    else {
      b = std::numeric_limits<double>::max();
    }

    temp = std::min(a, b);
    if (temp < tau) {
      tau = temp;
    }
  }

  return tau;
}


// Compute the g described in "Efficient step size selection for the
// tau-leaping simulation method".
inline
double
TauLeapingDynamic::
computeG(const std::size_t speciesIndex, const double population)
const
{
  const std::size_t order = _highestOrder[speciesIndex];

  if (order == 1) {
    return 1.0;
  }
  else if (order == 2) {
    if (_highestIndividualOrder[speciesIndex] == 1) {
      return 2.0;
    }
    else if (_highestIndividualOrder[speciesIndex] == 2) {
      return 2.0 + 1.0 / (population - 1);
    }
  }
  else if (order == 3) {
    if (_highestIndividualOrder[speciesIndex] == 1) {
      return 3.0;
    }
    else if (_highestIndividualOrder[speciesIndex] == 2) {
      return 1.5 * (2.0 + 1.0 / (population - 1.0));
    }
    else if (_highestIndividualOrder[speciesIndex] == 3) {
      return 3.0 + 1.0 / (population - 1) + 2.0 / (population - 2);
    }
  }

  // Catch any other cases with an assertion failure.
  assert(false);
  return 0;
}

inline
void
TauLeapingDynamic::
computeMuAndSigmaSquared(const StateChangeVectors& stateChangeVectors,
                         const std::vector<double>& propensities)
{
  // Initialize.
  std::fill(_mu.begin(), _mu.end(), 0.0);
  std::fill(_sigmaSquared.begin(), _sigmaSquared.end(), 0.0);

  double propensity;
  std::size_t reaction;
  // Loop over the active reactions.
  for (std::size_t i = 0; i != _activeReactions.size(); ++i) {
    reaction = _activeReactions[i];
    propensity = propensities[reaction];
#ifdef STLIB_DEBUG
    assert(propensity >= 0);
#endif
    // For each species that is modified by this reaction.
    for (StateChangeVectors::const_iterator j =
           stateChangeVectors.begin(reaction);
         j != stateChangeVectors.end(reaction); ++j) {
      _mu[j->first] += j->second * propensity;
      _sigmaSquared[j->first] += j->second * j->second * propensity;
    }
  }
}

} // namespace stochastic
}
