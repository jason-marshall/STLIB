// -*- C++ -*-

/*!
  \file stochastic/TauLeapingSerial.h
  \brief The tau-leaping method for SSA.
*/

#if !defined(__stochastic_TauLeapingSerial_h__)
#define __stochastic_TauLeapingSerial_h__

#include "TauLeapingData.h"

namespace stochastic {

//! The sequential tau-leaping algorithm.
template<typename T, typename UnaryFunctor>
class TauLeapingSerial : public TauLeapingData<T> {

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

   UnaryFunctor _poisson;

private:

   //
   // Not implemented.
   //

   // Default constructor not implemented.
   TauLeapingSerial();

   // Copy constructor not implemented.
   TauLeapingSerial(const TauLeapingSerial&);

   // Assignment operator not implemented.
   TauLeapingSerial&
   operator=(const TauLeapingSerial&);

public:

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{

   //! Construct from the number of species, epsilon and a seed for the random number generator.
   TauLeapingSerial(const int numberOfSpecies, const Number epsilon,
                    const int seed) :
      Base(numberOfSpecies, epsilon),
      _poisson(seed) {}

   //! Destructor.  Free internally allocated memory.
   ~TauLeapingSerial() {}

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
   using Base::setEpsilon;

   //@}
   //--------------------------------------------------------------------------
   //! \name Mathematical Functions.
   //@{

   //! Initialize the data structure in preparation for stepping.
   using Base::initialize;

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
   void
   seed(const int seedValue) {
      _poisson.seed(seedValue);
   }

   //@}
};


//! Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
int
simulateWithTauLeapingSerial(State<T>* state, T epsilon, T maximumTime,
                             int seed);


//! Take the specified number of steps.
template<typename UnaryFunctor, typename T>
void
simulateWithTauLeapingSerial(State<T>* state, T epsilon, int numberOfSteps,
                             int seed);

} // namespace stochastic

#define __stochastic_TauLeapingSerial_ipp__
#include "TauLeapingSerial.ipp"
#undef __stochastic_TauLeapingSerial_ipp__

#endif
