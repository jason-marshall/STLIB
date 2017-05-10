// -*- C++ -*-

/*!
  \file stochastic/TauLeapingDynamic.h
  \brief The tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingDynamic_h__)
#define __stochastic_TauLeapingDynamic_h__

#include "stlib/stochastic/State.h"
#include "stlib/stochastic/ReactionSet.h"

#include "stlib/numerical/random/poisson/PoissonGeneratorInvAcNormSure.h"
#include "stlib/container/StaticArrayOfArrays.h"

#include <set>

namespace stlib
{
namespace stochastic
{

//! Perform a stochastic simulation using the tau-leaping method.
/*!
  \param _State The state of the simulation: reactions and populations.
*/
class TauLeapingDynamic
{
  //
  // Public types.
  //
public:

  //! The state change vectors.
  typedef State::StateChangeVectors StateChangeVectors;
  //! The set of reactions.
  typedef ReactionSet<true> ReactionSetType;
  //! The Poisson generator.
  typedef numerical::PoissonGeneratorInvAcNormSure<> PoissonGenerator;
  //! The discrete uniform generator.
  typedef PoissonGenerator::DiscreteUniformGenerator
  DiscreteUniformGenerator;
  //! The normal generator.
  typedef PoissonGenerator::NormalGenerator NormalGenerator;

  //
  // Member data.
  //
private:

  NormalGenerator _normalGenerator;
  PoissonGenerator _poissonGenerator;

  // The active reactions.
  std::vector<std::size_t> _activeReactions;
  // The reactants for each reaction.
  container::StaticArrayOfArrays<std::size_t> _reactants;
  // The active species (reactants of the active reactions).
  std::vector<std::size_t> _activeSpecies;
  // The mean population change.
  std::vector<double> _mu;
  // The variance in the population change.
  std::vector<double> _sigmaSquared;
  std::vector<std::size_t> _highestOrder;
  std::vector<std::size_t> _highestIndividualOrder;
  double _epsilon;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  TauLeapingDynamic();
  //! Copy constructor not implemented.
  TauLeapingDynamic(const TauLeapingDynamic&);
  //! Assignment operator not implemented.
  TauLeapingDynamic&
  operator=(const TauLeapingDynamic&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct.
  TauLeapingDynamic(const State& state, const ReactionSetType& reactionSet,
                    DiscreteUniformGenerator* discreteUniformGenerator,
                    const double epsilon) :
    _normalGenerator(discreteUniformGenerator),
    // CONTINUE: normal threshhold.
    _poissonGenerator(&_normalGenerator, 1000),
    // Initially there are no active reactions.
    _activeReactions(),
    _reactants(),
    _activeSpecies(),
    _mu(state.getNumberOfSpecies()),
    _sigmaSquared(state.getNumberOfSpecies()),
    _highestOrder(state.getNumberOfSpecies()),
    _highestIndividualOrder(state.getNumberOfSpecies()),
    _epsilon()
  {
    initialize(reactionSet);
    // Set epsilon, the normal threshhold, and the sure threshhold.
    setEpsilon(epsilon);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Simulation.
  //@{
public:

  //! Compute the maximum allowed time step.
  double
  computeStep(const StateChangeVectors& listOfStateChangeVectors,
              const std::vector<double>& propensities,
              const std::vector<double>& populations);

  //! Generate a Poisson deviate with the specified mean.
  PoissonGenerator::result_type
  generatePoisson(const double mean)
  {
    return _poissonGenerator(mean);
  }

private:

  //! Compute the orders for the species.
  /*!
    Let \c highestOrder be the highest order reaction in which the species
    appears.  Let \c highestIndividualOrder be the highest order of the
    species in a reaction.
    Suppose that the reactions are the following.  (We only use the reactants
    in computing the orders.)
    \f[
    x0 \rightarrow \cdots, \quad
    x1 + x2 \rightarrow \cdots, \quad
    x2 + 2 x3 \rightarrow \cdots, \quad
    3 x4 \rightarrow \cdots
    \f]
    Then the orders are the following.
    \verbatim
    highestOrder == {1, 2, 3, 3, 3}
    highestIndividualOrder == {1, 1, 1, 2, 3} \endverbatim
  */
  void
  initialize(const ReactionSetType& reactionSet);

  //! Compute mu and sigma squared.
  void
  computeMuAndSigmaSquared(const StateChangeVectors& listOfStateChangeVectors,
                           const std::vector<double>& propensities);

  //! Compute the g described in "Efficient step size selection for the tau-leaping simulation method".
  double
  computeG(std::size_t speciesIndex, double population) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the value of epsilon.
  double
  getEpsilon() const
  {
    return _epsilon;
  }

  //! Get the active reactions.
  const std::vector<std::size_t>&
  getActiveReactions() const
  {
    return _activeReactions;
  }

  //! Return true if there are no active reactions.
  bool
  empty() const
  {
    return _activeReactions.empty();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Set the value of epsilon.
  void
  setEpsilon(const double epsilon)
  {
    _epsilon = epsilon;
    // The relative error in the mean is less than 0.1 * error.
    // continuityError / mean < 0.1 * error
    // 1 / mean < 0.1 * error
    // mean > 10 / error
    const double t = 10. / epsilon;
    _poissonGenerator.setNormalThreshhold(t);
    // The relative error in neglecting the standard deviation is less
    // than 0.1 * error.
    // sqrt(mean) / mean < 0.1 * error
    // mean > 100 / error^2
    _poissonGenerator.setSureThreshhold(t * t);
  }

  //! Clear the active set.
  void
  clear()
  {
    _activeReactions.clear();
    _activeSpecies.clear();
  }

  //! Insert a reaction to the active set.
  void
  insert(const std::size_t index)
  {
    _activeReactions.push_back(index);
    computeActiveSpecies();
  }

  //! Remove a reaction from the active set.
  void
  erase(const std::size_t index)
  {
    const std::ptrdiff_t i = std::find(_activeReactions.begin(),
                                       _activeReactions.end(), index)
                             - _activeReactions.begin();
#ifdef STLIB_DEBUG
    assert(std::size_t(i) != _activeReactions.size());
#endif
    _activeReactions[i] = _activeReactions.back();
    _activeReactions.pop_back();
    computeActiveSpecies();
  }

private:

  //! Compute the active species from the active reactions.
  void
  computeActiveSpecies()
  {
    typedef container::StaticArrayOfArrays<std::size_t>::const_iterator
    const_iterator;
    _activeSpecies.clear();
    std::set<std::size_t> active;
    for (std::vector<std::size_t>::const_iterator i = _activeReactions.begin();
         i != _activeReactions.end(); ++i) {
      for (const_iterator j = _reactants.begin(*i); j != _reactants.end(*i);
           ++j) {
        active.insert(*j);
      }
    }
    std::copy(active.begin(), active.end(),
              std::back_inserter(_activeSpecies));
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name I/O.
  //@{
public:

  void
  print(std::ostream& out) const
  {
    out << "Active reactions for tau-leaping:\n";
    for (std::size_t i = 0; i != _activeReactions.size(); ++i) {
      out << _activeReactions[i] << ' ';
    }
    out << '\n';

    out << "Active species for tau-leaping:\n";
    for (std::size_t i = 0; i != _activeSpecies.size(); ++i) {
      out << _activeSpecies[i] << ' ';
    }
    out << '\n';

    out << "Mean population change:\n";
    for (std::size_t i = 0; i != _mu.size(); ++i) {
      out << _mu[i] << ' ';
    }
    out << '\n';

    out << "Variance in population change:\n";
    for (std::size_t i = 0; i != _sigmaSquared.size(); ++i) {
      out << _sigmaSquared[i] << ' ';
    }
    out << '\n';

    out << "Highest order:\n";
    for (std::size_t i = 0; i != _highestOrder.size(); ++i) {
      out << _highestOrder[i] << ' ';
    }
    out << '\n';

    out << "Highest individual order:\n";
    for (std::size_t i = 0; i != _highestIndividualOrder.size(); ++i) {
      out << _highestIndividualOrder[i] << ' ';
    }
    out << '\n';

    out << "Epsilon = " << _epsilon << '\n';
  }

  //@}
};

} // namespace stochastic
}

#define __stochastic_TauLeapingDynamic_ipp__
#include "stlib/stochastic/TauLeapingDynamic.ipp"
#undef __stochastic_TauLeapingDynamic_ipp__

#endif
