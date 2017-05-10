// -*- C++ -*-

/*!
  \file stochastic/TauLeapingThreaded.h
  \brief The tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingThreaded_h__)
#define __stochastic_TauLeapingThreaded_h__

#include "TauLeapingData.h"

namespace stochastic {

//! The sequential tau-leaping algorithm.
template<typename T, typename UnaryFunctor>
class TauLeapingThreaded : public TauLeapingData<T> {

private:

   //
   // Private types.
   //

   typedef TauLeapingData<T> Base;

public:

   //
   // Public types.
   //

   //! The number type.
   typedef typename Base::Number Number;
   //! The simulation state.
   typedef typename Base::State State;

private:

   //
   // Member data.
   //

   ads::Array<1, ads::Array<1, Number> > _muArray;
   ads::Array<1, ads::Array<1, Number> > _sigmaSquaredArray;
   ads::Array<1, UnaryFunctor> _poisson;

private:

   //
   // Not implemented.
   //

   // Default constructor not implemented.
   TauLeapingThreaded();

   // Copy constructor not implemented.
   TauLeapingThreaded(const TauLeapingThreaded&);

   // Assignment operator not implemented.
   TauLeapingThreaded&
   operator=(const TauLeapingThreaded&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Construct from the number of species, epsilon and a seed for the random number generator.
   /*!
     \pre Must be called from the master thread.
   */
   TauLeapingThreaded(const int numberOfSpecies, const Number epsilon,
                      const int seed) :
      Base(numberOfSpecies, epsilon),
      _muArray(omp_get_num_procs()),
      _sigmaSquaredArray(omp_get_num_procs()),
      _poisson(omp_get_num_procs()) {
      // Must be called from the master thread.
      assert(omp_get_thread_num() == 0);
      // CONTINUE: Address the seeding problem later.
      for (int i = 0; i != _poisson.size(); ++i) {
         _poisson.seed(seed + i);
      }
   }

   //! Destructor.  Free internally allocated memory.
   /*!
     \pre Must be called from the master thread.
   */
   ~TauLeapingThreaded() {
      assert(omp_get_thread_num() == 0);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{

   //! Get the value of epsilon.
   using Base::getEpsilon;

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{

   //! Set the value of epsilon.
   void
   setEpsilon(const Number epsilon) {
      if (omp_get_thread_num() == 0) {
         Base::setEpsilon(epsilon);
      }
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Mathematical Functions.
   //@{

   //! Initialize the data structure in preparation for stepping.
   void
   initialize(const State& state) {
      if (omp_get_thread_num() == 0) {
         Base::initialize(state);
      }
   }

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

   //! Seed the random number generator for this thread.
   void
   seed(const int seedValue) {
      _poisson[omp_get_thread_num()].seed(seedValue);
   }

   //@}
};


//! Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
int
simulateWithTauLeapingThreaded(State<T>* state, T epsilon, T maximumTime,
                               int seed);


//! Take the specified number of steps.
template<typename UnaryFunctor, typename T>
void
simulateWithTauLeapingThreaded(State<T>* state, T epsilon, int numberOfSteps,
                               int seed);

} // namespace stochastic

#define __stochastic_TauLeapingThreaded_ipp__
#include "TauLeapingThreaded.ipp"
#undef __stochastic_TauLeapingThreaded_ipp__

#endif
