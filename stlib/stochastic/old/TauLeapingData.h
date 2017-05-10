// -*- C++ -*-

/*!
  \file stochastic/TauLeapingData.h
  \brief Holds data for the tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingData_h__)
#define __stochastic_TauLeapingData_h__

#include "ReactionSet.h"

namespace stochastic {

//! This class holds the data necessary to take a tau-leaping step.
/*!
  \param _State The state of the simulation: reactions, populations, and time.
*/
template<class _State>
class TauLeapingData {
   //
   // Public types.
   //
public:

   //! The state.
   typedef _State State;
   //! The number type.
   typedef typename State::Number Number;

   //
   // Member data.
   //
private:

   ads::Array<1, Number> _mu;
   ads::Array<1, Number> _sigmaSquared;
   ads::Array<1, int> _highestOrder;
   ads::Array<1, int> _highestIndividualOrder;
   Number _epsilon;

   //
   // Not implemented.
   //
private:

   // Default constructor not implemented.
   TauLeapingData();

   // Copy constructor not implemented.
   TauLeapingData(const TauLeapingData&);

   // Assignment operator not implemented.
   TauLeapingData&
   operator=(const TauLeapingData&);

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct from the number of species and epsilon.
   TauLeapingData(const int numberOfSpecies, const Number epsilon) :
      _mu(numberOfSpecies),
      _sigmaSquared(numberOfSpecies),
      _highestOrder(numberOfSpecies),
      _highestIndividualOrder(numberOfSpecies),
      _epsilon(epsilon) {}

   //! Destructor.  Free internally allocated memory.
   ~TauLeapingData() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the value of epsilon.
   Number
   getEpsilon() const {
      return _epsilon;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the value of epsilon.
   void
   setEpsilon(const Number epsilon) {
      _epsilon = epsilon;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Mathematical Functions.
   //@{
public:

   //! Compute the orders for the species.
   /*!
     \param state The state of the simulation.

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
     highestIndividualOrder == {1, 1, 1, 2, 3}
     \endverbatim
   */
   void
   initialize(const State& state);

   Number
   computeTau(const State& state);

   //@}
   //--------------------------------------------------------------------------
   // Private functions.
private:

   //! Compute mu and sigma squared.
   void
   computeMuAndSigmaSquared(const State& state);

   //! Compute the g described in "Efficient step size selection for the tau-leaping simulation method".
   Number
   computeG(const State& state, int speciesIndex) const;
};

} // namespace stochastic

#define __stochastic_TauLeapingData_ipp__
#include "TauLeapingData.ipp"
#undef __stochastic_TauLeapingData_ipp__

#endif
