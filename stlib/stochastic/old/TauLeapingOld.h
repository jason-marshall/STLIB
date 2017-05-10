// -*- C++ -*-

/*!
  \file stochastic/TauLeaping.h
  \brief The tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeaping_h__)
#define __stochastic_TauLeaping_h__

#include "TauLeapingData.h"

namespace stochastic {

//! The sequential tau-leaping algorithm.
template<typename T, typename UnaryFunctor>
class TauLeaping {

public:

   //
   // Public types.
   //

   //! The number type.
   typedef T Number;
   //! The simulation state.
   typedef State<Number> State;

private:

   //
   // Member data.
   //

   ads::Array<1, int> _highestOrder;
   ads::Array<1, int> _highestIndividualOrder;
   Number _epsilon;
#ifdef _OPENMP
   /*!
     The first array holds the accumulated values.  The rest are used for
     computing intermediate results within each thread.
   */
   ads::Array<1, ads::Array<1, Number> > _mu;
   /*!
     The first array holds the accumulated values.  The rest are used for
     computing intermediate results within each thread.
   */
   ads::Array<1, ads::Array<1, Number> > _sigmaSquared;
   //! Poisson random number generator.  One for each thread.
   ads::Array<1, UnaryFunctor> _poisson;
   /*!
     The first element holds the time leap.  The rest are used for
     computing intermediate results within each thread.
   */
   ads::Array<1, Number> _tau;
#else
   ads::Array<1, Number> _mu;
   ads::Array<1, Number> _sigmaSquared;
   //! Poisson random number generator.
   UnaryFunctor _poisson;
   Number _tau;
#endif

private:

   //
   // Not implemented.
   //

   // Default constructor not implemented.
   TauLeaping();

   // Copy constructor not implemented.
   TauLeaping(const TauLeaping&);

   // Assignment operator not implemented.
   TauLeaping&
   operator=(const TauLeaping&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Construct from the number of species, epsilon and a seed for the random number generator.
   /*!
     \note This may only be called from the master thread.
   */
   TauLeaping(int numberOfSpecies, Number epsilon, int seedValue);

   //! Destructor.  Free internally allocated memory.
   /*!
     \note This may only be called from the master thread.
   */
   ~TauLeaping() {
#ifdef _OPENMP
      assert(omp_get_thread_num() == 0);
#endif
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{

   //! Get the value of epsilon.
   Number
   getEpsilon() const {
      return _epsilon;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{

   //! Set the value of epsilon.
   void
   setEpsilon(const Number epsilon) {
      _epsilon = epsilon;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Mathematical Functions.
   //@{

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

   //! Take a step with the tau-leaping method.
   void
   step(State* state, Number maximumTime = std::numeric_limits<Number>::max());

   //! Advance the simulation to the specified time.
   /*!
     \pre You must call initialize() before using this function.
     \return The number of steps taken.
   */
   int
   simulate(State* state, Number maximumTime);

   //! Advance the simulation by the specified number of steps.
   /*!
     \pre You must call initialize() before using this function.
   */
   void
   simulate(State* state, int numberOfSteps);

   //! Seed the random number generator.
   /*!
     \note Only affects the generator for this thread.
   */
   void
   seed(const int seedValue) {
#ifdef _OPENMP
      _poisson[omp_get_thread_num()].seed(seedValue);
#else
      _poisson.seed(seedValue);
#endif
   }

   //@}
   //--------------------------------------------------------------------------
   // Private functions.

private:

   //! Compute the time leap.
   Number
   computeTau(const State& state);

   //! Compute the time leap using only the master thread.
   /*!
     \note This must be called from all threads because it has a barrier.
   */
   Number
   computeTauSingle(const State& state);

#ifdef _OPENMP
   //! Compute the time leap using all of the threads.
   Number
   computeTauMulti(const State& state);
#endif

   //! Compute mu and sigma squared on the master thread.
   /*!
     \note This may only be called by the master thread.  This allows me to
     implement the function without barriers.
   */
   void
   computeMuAndSigmaSquaredSingle(const State& state);

#ifdef _OPENMP
   //! Compute mu and sigma squared using all of the threads.
   void
   computeMuAndSigmaSquaredMulti(const State& state);
#endif

   //! Compute the g described in "Efficient step size selection for the tau-leaping simulation method".
   Number
   computeG(const State& state, int speciesIndex) const;
};


//! Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
int
simulateWithTauLeaping(State<T>* state, T epsilon, T maximumTime, int seed);


//! Take the specified number of steps.
template<typename UnaryFunctor, typename T>
void
simulateWithTauLeaping(State<T>* state, T epsilon, int numberOfSteps, int seed);

} // namespace stochastic

#define __stochastic_TauLeaping_ipp__
#include "TauLeaping.ipp"
#undef __stochastic_TauLeaping_ipp__

#endif
