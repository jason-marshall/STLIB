// -*- C++ -*-

/*!
  \file stochastic/State.h
  \brief The state of the stochastic simulation.
*/

#if !defined(__stochastic_State_h__)
#define __stochastic_State_h__

#include "stlib/container/SparseVector.h"
#include "stlib/container/StaticArrayOfArrays.h"
#include "stlib/ext/vector.h"

namespace stlib
{
namespace stochastic
{

USING_STLIB_EXT_VECTOR_MATH_OPERATORS;
USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//----------------------------------------------------------------------------
//! The state of the stochastic simulation.
/*!
  Hold the reaction count, the populations, and the state change
  vectors.

  The data structure for the state change vectors stores pairs of indices
  (std::size_t) and stoichiometries (double). One could also use
  pairs of pointers and stoichiometries, the pointers would point to
  addresses in the populataion array. However, there is no performance
  advantage in the former method. Array indexing is about as fast as pointer
  dereferencing.

  Species populations are represented with the double precision floating point
  number type.
*/
class State
{
  //
  // Public types.
  //
public:

  //! The sparse array of state change vectors.
  typedef container::StaticArrayOfArrays<std::pair<std::size_t, double> >
  StateChangeVectors;

  //
  // Member data.
  //
private:

  //! The total number of reaction firings.
  double _totalReactionCount;
  //! The populations of the species.
  std::vector<double> _populations;
  //! The number of reaction firings.
  std::vector<double> _reactionCounts;
  //! The state change vectors.
  StateChangeVectors _stateChangeVectors;

  //
  // Not implemented.
  //
private:

  //! Default constructor not implemented.
  State();
  //! Assignment operator not implemented.
  State&
  operator=(const State&);

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the number of species and the state change vectors.
  template<typename _ForwardIterator>
  State(const std::size_t numberOfSpecies, _ForwardIterator reactionsBegin,
        _ForwardIterator reactionsEnd) :
    _totalReactionCount(0),
    // Initialize the populations to zero.
    _populations(numberOfSpecies, 0),
    // Initialize the reaction counts.
    _reactionCounts(std::distance(reactionsBegin, reactionsEnd), 0.),
    _stateChangeVectors()
  {
    buildStateChangeVectors(reactionsBegin, reactionsEnd);
  }

  //! Construct from the populations and the state change vectors.
  template<typename _ForwardIterator>
  State(const std::vector<double>& populations,
        _ForwardIterator reactionsBegin, _ForwardIterator reactionsEnd) :
    _totalReactionCount(0),
    // Initialize the populations to zero.
    _populations(populations),
    // Initialize the reaction counts.
    _reactionCounts(std::distance(reactionsBegin, reactionsEnd), 0.),
    _stateChangeVectors()
  {
    buildStateChangeVectors(reactionsBegin, reactionsEnd);
  }

  // Use the default copy constructor and destructor.

private:

  //! Build the state change vectors.
  template<typename _ForwardIterator>
  void
  buildStateChangeVectors(_ForwardIterator reactionsBegin,
                          _ForwardIterator reactionsEnd);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Get the total number of reaction firings.
  double
  getReactionCount() const
  {
    return _totalReactionCount;
  }

  //! Get the number of reaction firings for the specified reaction.
  double
  getReactionCount(const std::size_t reactionIndex) const
  {
    return _reactionCounts[reactionIndex];
  }

  //! Get the vector of reaction counts.
  const std::vector<double>&
  getReactionCounts() const
  {
    return _reactionCounts;
  }

  //! Get the number of species.
  std::size_t
  getNumberOfSpecies() const
  {
    return _populations.size();
  }

  //! Get the number of reactions.
  std::size_t
  getNumberOfReactions() const
  {
    return _reactionCounts.size();
  }

  //! Get the populations.
  const std::vector<double>&
  getPopulations() const
  {
    return _populations;
  }

  //! Get the specified population.
  double
  getPopulation(const std::size_t speciesIndex) const
  {
    return _populations[speciesIndex];
  }

  //! Get the state change vectors.
  const StateChangeVectors&
  getStateChangeVectors() const
  {
    return _stateChangeVectors;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Validity.
  //@{
public:

  //! Return true if the state is valid.
  bool
  isValid() const
  {
    // Check that there are no negative populations.
    for (std::size_t i = 0; i != _populations.size(); ++i) {
      if (_populations[i] < 0) {
        return false;
      }
    }
    return true;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulators.
  //@{
public:

  //! Reset the reaction counts.
  void
  resetReactionCounts()
  {
    _totalReactionCount = 0;
    std::fill(_reactionCounts.begin(), _reactionCounts.end(), 0);
  }

  //! Set the populations.
  void
  setPopulations(const std::vector<double>& populations)
  {
    assert(_populations.size() == populations.size());
    // Copy the populations.
    std::copy(populations.begin(), populations.end(), _populations.begin());
  }

  //! Set the specified population.
  void
  setPopulation(const std::size_t index, const double population)
  {
    _populations[index] = population;
  }

  //! Offset the populations.
  void
  offsetPopulations(const std::vector<double>& change)
  {
    _populations += change;
  }

  //! Increment the specified reaction counts.
  void
  incrementReactionCounts(const std::size_t reactionIndex,
                          const double numberOfTimes)
  {
    _totalReactionCount += numberOfTimes;
    _reactionCounts[reactionIndex] += numberOfTimes;
  }

  //! Fix any negative populations (by making them zero).
  void
  fixNegativePopulations()
  {
    // Fix the negative populations.
    const std::size_t end = _populations.size();
    for (std::size_t n = 0; n != end; ++n) {
      if (_populations[n] < 0) {
        _populations[n] = 0;
      }
    }
  }

  //! Fire the specified reaction.
  void
  fireReaction(const std::size_t n);

  //! Un-fire the specified reaction.
  void
  unFireReaction(const std::size_t n);

  //! Fire the specified reaction the specified number of times.
  void
  fireReaction(const std::size_t reactionIndex,
               const double numberOfTimes);

  //! Fire the specified reaction the specified number of times to update the specified populations.
  /*! Do not increment the reaction counts. */
  void
  fireReaction(std::vector<double>* populations,
               const std::size_t reactionIndex,
               const double numberOfTimes) const;

  //@}
  //--------------------------------------------------------------------------
  //! \name Mathematical functions.
  //@{
public:

  //! Compute the time derivatives of the populations.
  /*! Of course, we are assuming a deterministic evolution of the
    populations. The i_th component of \f$\mathrm{d}\mathrm{x}/\mathrm{d}t\f$
    is obtained by summing the propensity contribution from each reaction.
    \[
    \frac{\mathrm{d} x_i}{\mathrm{d}t} =
    \sum_j a_j(\mathbf{x}) \nu_{ij}
    \]

    \param propensities The values of the propensity functions.
    \param dxdt The time derivative of the species populations.
  */
  void
  populationDerivatives(const std::vector<double>& propensities,
                        std::vector<double>* dxdt) const
  {
#ifdef STLIB_DEBUG
    assert(propensities.size() == getNumberOfReactions());
    assert(dxdt->size() == getNumberOfSpecies());
#endif
    std::fill(dxdt->begin(), dxdt->end(), 0);
    // Loop over the reactions.
    for (std::size_t j = 0; j != _stateChangeVectors.getNumberOfArrays();
         ++j) {
      // Loop over the entries in the state change vector for the j_th
      // reaction.
      for (std::size_t k = 0; k != _stateChangeVectors.size(j); ++k) {
        const std::pair<std::size_t, double>& p = _stateChangeVectors(j, k);
        (*dxdt)[p.first] += propensities[j] * p.second;
      }
    }
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name I/O
  //@{
public:

  void
  print(std::ostream& out) const
  {
    out << "Total reaction count = " << getReactionCount() << '\n';
    out << "Populations:\n";
    for (std::size_t i = 0; i != getNumberOfSpecies(); ++i) {
      out << getPopulation(i) << ' ';
    }
    out << '\n';
    out << "Reaction counts:\n";
    for (std::size_t i = 0; i != getNumberOfReactions(); ++i) {
      out << getReactionCount(i) << ' ';
    }
    out << '\n';
  }

  //@}
};

//! Return true if the states are equal.
/*! \relates State */
bool
operator==(const State& x, const State& y);

//! Return true if the states are not equal.
/*! \relates State */
inline
bool
operator!=(const State& x, const State& y)
{
  return !(x == y);
}

//! Write the populations in ascii format.
/*! \relates State */
void
writePopulationsAscii(std::ostream& out, const State& x);

//! Read the populations in ascii format.
/*! \relates State */
void
readPopulationsAscii(std::istream& in, State* x);

//! Write the state in ascii format.
/*! \relates State */
void
writeAscii(std::ostream& out, const State& x);

} // namespace stochastic
}

#define __stochastic_State_ipp__
#include "stlib/stochastic/State.ipp"
#undef __stochastic_State_ipp__

#endif
