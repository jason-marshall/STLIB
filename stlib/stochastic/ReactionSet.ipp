// -*- C++ -*-

#if !defined(__stochastic_ReactionSet_ipp__)
#error This file is an implementation detail of ReactionSet.
#endif

namespace stlib
{
namespace stochastic
{

//----------------------------------------------------------------------------
// Manipulators.
//----------------------------------------------------------------------------


template<bool _IsDiscrete>
template<typename _InputIterator>
inline
void
ReactionSet<_IsDiscrete>::
rebuild(_InputIterator reactionsBeginning, _InputIterator reactionsEnd)
{
  _reactions.clear();
  _reactions.insert(_reactions.end(), reactionsBeginning, reactionsEnd);
}


//----------------------------------------------------------------------------
// Free functions
//----------------------------------------------------------------------------

// Return true if the states are equal.
template<bool _IsDiscrete>
inline
bool
operator==(const ReactionSet<_IsDiscrete>& x,
           const ReactionSet<_IsDiscrete>& y)
{
  if (x.getSize() != y.getSize()) {
    return false;
  }
  for (std::size_t n = 0; n != x.getSize(); ++n) {
    if (x.getReaction(n) != y.getReaction(n)) {
      return false;
    }
  }

  return true;
}


// Write the reactions in ascii format.
template<bool _IsDiscrete>
inline
void
writeAscii(std::ostream& out, const ReactionSet<_IsDiscrete>& x)
{
  out << x.getSize() << "\n";
  for (std::size_t n = 0; n != x.getSize(); ++n) {
    writeAscii(out, x.getReaction(n));
  }
}


// Read the reactions in ascii format.
template<bool _IsDiscrete>
inline
void
readAscii(std::istream& in, ReactionSet<_IsDiscrete>* x)
{
  typedef typename ReactionSet<_IsDiscrete>::ReactionType ReactionType;

  std::size_t numberOfReactions;
  in >> numberOfReactions;

  std::vector<ReactionType> reactions(numberOfReactions);
  for (std::size_t n = 0; n != numberOfReactions; ++n) {
    readAscii(in, &reactions[n]);
  }

  x->rebuild(reactions.begin(), reactions.end());
}


// Read the reactants and products in ascii format.
template<bool _IsDiscrete>
inline
void
readReactantsAndProductsAscii(std::istream& in, ReactionSet<_IsDiscrete>* x)
{
  std::size_t numberOfReactions;
  in >> numberOfReactions;
  readReactantsAndProductsAscii(in, numberOfReactions, x);
}

// Read the reactants and products in ascii format.
template<bool _IsDiscrete>
inline
void
readReactantsAndProductsAscii(std::istream& in,
                              const std::size_t numberOfReactions,
                              ReactionSet<_IsDiscrete>* x)
{
  typedef typename ReactionSet<_IsDiscrete>::ReactionType ReactionType;
  std::vector<ReactionType> reactions(numberOfReactions);
  for (std::size_t n = 0; n != numberOfReactions; ++n) {
    readReactantsAscii(in, &reactions[n]);
    readProductsAscii(in, &reactions[n]);
    readDependenciesAscii(in, &reactions[n]);
  }

  x->rebuild(reactions.begin(), reactions.end());
}

// Read the rate constants in ascii format.
template<bool _IsDiscrete>
inline
void
readRateConstantsAscii(std::istream& in, ReactionSet<_IsDiscrete>* x)
{
  std::vector<double> rateConstants;
  in >> rateConstants;
  assert(rateConstants.size() == x->getSize());
  x->setRateConstants(rateConstants.begin(), rateConstants.end());
}

} // namespace stochastic
}
