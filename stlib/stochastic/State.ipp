// -*- C++ -*-

#if !defined(__stochastic_State_ipp__)
#error This file is an implementation detail of State.
#endif

namespace stlib
{
namespace stochastic
{

template<typename _ForwardIterator>
inline
void
State::
buildStateChangeVectors(_ForwardIterator reactionsBegin,
                        _ForwardIterator reactionsEnd)
{
  // Dynamic container for the state change vectors.
  typedef container::SparseVector<double> SparseVector;
  std::vector<SparseVector> scv(std::distance(reactionsBegin, reactionsEnd));

  // Construct the state change vectors from the reactions.
  SparseVector reactants, products;
  for (std::size_t i = 0; i != scv.size(); ++i, ++reactionsBegin) {
    // Convert from size type to number type.
    reactants = reactionsBegin->getReactants();
    products = reactionsBegin->getProducts();
    // Take the difference.
    scv[i] = products - reactants;
  }

  // Build the static data structure.
  _stateChangeVectors = StateChangeVectors(scv);
}

// Fire the specified reaction.
inline
void
State::
fireReaction(const std::size_t n)
{
  ++_totalReactionCount;
  ++_reactionCounts[n];
  for (StateChangeVectors::const_iterator i =
         _stateChangeVectors.begin(n); i != _stateChangeVectors.end(n); ++i) {
    _populations[i->first] += i->second;
  }
}

// Un-fire the specified reaction.
inline
void
State::
unFireReaction(const std::size_t n)
{
  --_totalReactionCount;
  --_reactionCounts[n];
  for (StateChangeVectors::const_iterator i =
         _stateChangeVectors.begin(n); i != _stateChangeVectors.end(n); ++i) {
    _populations[i->first] -= i->second;
  }
}

// Fire the specified reaction the specified number of times.
inline
void
State::
fireReaction(const std::size_t reactionIndex, const double numberOfTimes)
{
  _totalReactionCount += numberOfTimes;
  _reactionCounts[reactionIndex] += numberOfTimes;
  for (StateChangeVectors::const_iterator i =
         _stateChangeVectors.begin(reactionIndex);
       i != _stateChangeVectors.end(reactionIndex); ++i) {
    _populations[i->first] += numberOfTimes * i->second;
  }
}

// Fire the specified reaction the specified number of times to update the specified populations.
inline
void
State::
fireReaction(std::vector<double>* populations, const std::size_t reactionIndex,
             const double numberOfTimes) const
{
  for (StateChangeVectors::const_iterator i =
         _stateChangeVectors.begin(reactionIndex);
       i != _stateChangeVectors.end(reactionIndex); ++i) {
    (*populations)[i->first] += numberOfTimes * i->second;
  }
}

//----------------------------------------------------------------------------
// Free functions.
//----------------------------------------------------------------------------

// Return true if the states are equal.
inline
bool
operator==(const State& x, const State& y)
{
  return x.getReactionCount() == y.getReactionCount() &&
         x.getReactionCounts() == y.getReactionCounts() &&
         x.getPopulations() == y.getPopulations() &&
         x.getStateChangeVectors() == y.getStateChangeVectors();
}


// Write the populations in ascii format.
inline
void
writePopulationsAscii(std::ostream& out, const State& x)
{
  out << x.getPopulations();
}


// Read the populations in ascii format.
inline
void
readPopulationsAscii(std::istream& in, State* state)
{
  std::vector<double> populations;
  in >> populations;
  state->setPopulations(populations);
}


// Write the state in ascii format.
inline
void
writeAscii(std::ostream& out, const State& state)
{
  out << state.getPopulations();
}

} // namespace stochastic
}
