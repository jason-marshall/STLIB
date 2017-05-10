// -*- C++ -*-

#if !defined(__stochastic_TauLeapingSerial_ipp__)
#error This file is an implementation detail of TauLeapingSerial.
#endif

namespace stochastic {

//---------------------------------------------------------------------------
// TauLeapingSerial
//---------------------------------------------------------------------------

template<typename T, typename UnaryFunctor>
inline
void
TauLeapingSerial<T, UnaryFunctor>::
step(State* state, const Number maximumTime) {
   assert(state->getTime() < maximumTime);

   // Compute the propensity functions.
   state->computePropensityFunctions();

   // Compute the time leap.
   T tau = computeTau(*state);
   // If the time leap will take us past the maximum time.
   if (state->getTime() + tau > maximumTime) {
      tau = maximumTime - state->getTime();
      // Advance the time to the ending time.
      state->setTime(maximumTime);
   }
   else {
      // Advance the time by tau.
      state->advanceTime(tau);
   }

   // Advance the state.
   // For each reaction.
   for (int m = 0; m != state->getNumberOfReactions(); ++m) {
      state->fireReaction(m, _poisson(state->getPropensityFunction(m) * tau));
   }
}


// Advance the simulation to the specified time.
template<typename T, typename UnaryFunctor>
inline
int
TauLeapingSerial<T, UnaryFunctor>::
simulate(State* state, const Number maximumTime) {
   int numberOfSteps = 0;
   do {
      step(state, maximumTime);
      ++numberOfSteps;
   }
   while (state->getTime() < maximumTime);
   return numberOfSteps;
}


// Advance the simulation by the specified number of steps.
template<typename T, typename UnaryFunctor>
inline
void
TauLeapingSerial<T, UnaryFunctor>::
simulate(State* state, const int numberOfSteps) {
   assert(numberOfSteps >= 0);
   for (int n = 0; n != numberOfSteps; ++n) {
      step(state);
   }
}


//---------------------------------------------------------------------------
// Interface functions.
//---------------------------------------------------------------------------


// Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
inline
int
simulateWithTauLeapingSerial(State<T>* state, const T epsilon,
                             const T maximumTime, const int seed) {
   // Construct the tau-leaping data structure.
   TauLeapingSerial<T, UnaryFunctor>
   tauLeaping(state->getNumberOfSpecies(), epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);
   // Advance the simulation.
   return tauLeaping.simulate(state, maximumTime);
}



// Take the specified number of steps.
template<typename UnaryFunctor, typename T>
inline
void
simulateWithTauLeapingSerial(State<T>* state, const T epsilon,
                             const int numberOfSteps, const int seed) {
   // Construct the tau-leaping data structure.
   TauLeapingSerial<T, UnaryFunctor>
   tauLeaping(state->getNumberOfSpecies(), epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);
   // Advance the simulation.
   tauLeaping.simulate(state, numberOfSteps);
}

} // namespace stochastic {
