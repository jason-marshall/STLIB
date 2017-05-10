// -*- C++ -*-

#if !defined(__stochastic_reactionPropensityInfluence_ipp__)
#error This file is an implementation detail of reactionPropensityInfluence.
#endif

namespace stlib
{
namespace stochastic
{

// Compute the reaction-propensity influence.
template<typename ForwardIterator>
inline
void
computeReactionPropensityInfluence
(const std::size_t numberOfSpecies,
 ForwardIterator reactionsBeginning, ForwardIterator reactionsEnd,
 container::StaticArrayOfArrays<std::size_t>* influence,
 const bool includeSelf)
{
  typedef container::SparseVector<std::size_t> SparseVector;
  typedef SparseVector::const_iterator const_iterator;

  // The index and coefficient of a species in a reaction.
  typedef std::map<std::size_t, std::size_t>::value_type IndexAndCoefficient;

  //
  // First build a mapping from species to reactions. For each species,
  // list the reaction propensities that depend on it.
  //
  container::StaticArrayOfArrays<std::size_t> speciesToReactions;
  if (numberOfSpecies != 0) {
    // Determine the sizes for the array of arrays.
    std::vector<std::size_t> sizes(numberOfSpecies, 0);
    // For each reaction.
    for (ForwardIterator reaction = reactionsBeginning;
         reaction != reactionsEnd; ++reaction) {
      const std::vector<std::size_t>& dependencies =
        reaction->getDependencies();
      // For each dependency.
      for (std::size_t i = 0; i != dependencies.size(); ++i) {
        ++sizes[dependencies[i]];
      }
    }

    // Allocate memory.
    speciesToReactions.rebuild(sizes.begin(), sizes.end());

    // Record the mapping from reactants to reactions.
    std::fill(sizes.begin(), sizes.end(), 0);
    std::size_t reactionIndex = 0;
    for (ForwardIterator reaction = reactionsBeginning;
         reaction != reactionsEnd; ++reaction, ++reactionIndex) {
      const std::vector<std::size_t>& dependencies =
        reaction->getDependencies();
      // For each dependency.
      for (std::size_t i = 0; i != dependencies.size(); ++i) {
        const std::size_t n = dependencies[i];
        speciesToReactions(n, sizes[n]) = reactionIndex;
        ++sizes[n];
      }
    }
  }

  // The number of influences for each reaction.
  std::vector<std::size_t> sizes;
  // The reaction indices.
  std::vector<std::size_t> reactionIndices;

  // We store the reactants and products in a std::map for easy searching.
  std::map<std::size_t, std::size_t> reactants, products;
  // The influenced species and reactions for a reaction.
  std::set<std::size_t> influencedSpecies, influencedReactions;

  // For each reaction.
  std::size_t reactionIndex = 0;
  for (ForwardIterator reaction = reactionsBeginning; reaction != reactionsEnd;
       ++reaction, ++reactionIndex) {
    //
    // First determine the influence on the species.  If a reaction changes the
    // population of a species, then it influences the species.  I check
    // the case that a species is both a reactant and product, but does
    // not change in population.  For example, the reaction
    // x_0 + x_1 -> x_0 + x_2
    // influences species 1 and 2, but not species 0.
    //

    // Get the reactants.
    reactants.clear();
    {
      const SparseVector& sv = reaction->getReactants();
      for (const_iterator i = sv.begin(); i != sv.end(); ++i) {
        reactants.insert(IndexAndCoefficient(i->first, i->second));
      }
    }
    // Get the products.
    products.clear();
    {
      const SparseVector& sv = reaction->getProducts();
      for (const_iterator i = sv.begin(); i != sv.end(); ++i) {
        products.insert(IndexAndCoefficient(i->first, i->second));
      }
    }
    influencedSpecies.clear();
    influencedReactions.clear();

    // First check the reactants.
    std::map<std::size_t, std::size_t>::const_iterator j;
    // For each reactant.
    for (std::map<std::size_t, std::size_t>::const_iterator i =
           reactants.begin(); i != reactants.end(); ++i) {
      // See if the same species is also a product.
      j = products.find(i->first);
      // If the species is not a product or if the product coefficient differs
      // from the reactant coefficient.
      if (j == products.end() || j->second != i->second) {
        // The species is influenced.
        influencedSpecies.insert(i->first);
      }
    }

    // Then check the products.
    // For each product.
    for (std::map<std::size_t, std::size_t>::const_iterator i =
           products.begin(); i != products.end(); ++i) {
      // See if the same species is also a reactant.
      j = reactants.find(i->first);
      // If the species is not a reactant or if the reactant coefficient
      // differs from the product coefficient.
      if (j == reactants.end() || j->second != i->second) {
        // The species is influenced.
        influencedSpecies.insert(i->first);
      }
    }

    // Now we have the species that are influenced by this reaction.  We then
    // determine the reaction propensities that are influenced.

    for (std::set<std::size_t>::const_iterator species =
           influencedSpecies.begin();
         species != influencedSpecies.end(); ++species) {
      for (std::size_t n = 0; n != speciesToReactions.size(*species); ++n) {
        const std::size_t r = speciesToReactions(*species, n);
        if (! includeSelf && r == reactionIndex) {
          continue;
        }
        influencedReactions.insert(r);
      }
    }

    // Now we have the reactions whose propensities are influenced by
    // this reaction.  Record the number of influenced reactions and their
    // indices.
    sizes.push_back(influencedReactions.size());
    for (std::set<std::size_t>::const_iterator i = influencedReactions.begin();
         i != influencedReactions.end(); ++i) {
      reactionIndices.push_back(*i);
    }
  }

  // Build the static array of arrays.
  influence->rebuild(sizes.begin(), sizes.end(), reactionIndices.begin(),
                     reactionIndices.end());
}

} // namespace stochastic
}
