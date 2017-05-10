// -*- C++ -*-

#if !defined(__stochastic_TauLeaping_ipp__)
#error This file is an implementation detail of TauLeaping.
#endif

namespace stochastic {

//---------------------------------------------------------------------------
// Constructors etc.
//---------------------------------------------------------------------------


#ifdef _OPENMP
template<typename T, typename UnaryFunctor>
inline
TauLeaping<T, UnaryFunctor>::
TauLeaping(const int numberOfSpecies, const Number epsilon,
           const int seedValue) :
   _highestOrder(numberOfSpecies),
   _highestIndividualOrder(numberOfSpecies),
   _epsilon(epsilon),
   _mu(),
   _sigmaSquared(),
   _poisson(),
   _tau() {
   assert(omp_get_thread_num() == 0);

   int numberOfThreads = 0;
#pragma omp parallel
   if (omp_get_thread_num() == 0) {
      numberOfThreads = omp_get_num_threads();
   }
   assert(numberOfThreads > 0);

   _mu.resize(numberOfThreads);
   _sigmaSquared.resize(numberOfThreads);
   _poisson.resize(numberOfThreads);
   _tau.resize(numberOfThreads);

   for (int i = 0; i != _mu.size(); ++i) {
      _mu[i].resize(numberOfSpecies);
      _sigmaSquared[i].resize(numberOfSpecies);
      // CONTIINUE: Improve the seeding across the threads.
      _poisson[i].seed(seedValue + i);
   }
}
#else
template<typename T, typename UnaryFunctor>
inline
TauLeaping<T, UnaryFunctor>::
TauLeaping(const int numberOfSpecies, const Number epsilon,
           const int seedValue) :
   _highestOrder(numberOfSpecies),
   _highestIndividualOrder(numberOfSpecies),
   _epsilon(epsilon),
   _mu(numberOfSpecies),
   _sigmaSquared(numberOfSpecies),
   _poisson(seedValue),
   _tau() {}
#endif

//---------------------------------------------------------------------------
//
//---------------------------------------------------------------------------



template<typename T, typename UnaryFunctor>
inline
void
TauLeaping<T, UnaryFunctor>::
initialize(const State& state) {
   typedef typename State::Reaction Reaction;

#ifdef _OPENMP
   // If this is not the master thread.
   if (omp_get_thread_num() != 0) {
      // Do nothing.
      return;
   }
#endif

   // Check that the arrays are the correct size.
   assert(state.getNumberOfSpecies() == _highestOrder.size());
   assert(state.getNumberOfSpecies() == _highestIndividualOrder.size());

   // Initialize the arrays.
   _highestOrder = 0;
   _highestIndividualOrder = 0;

   int sum, index, order;
   // Loop over the reactions.
   for (int n = 0; n != state.getNumberOfReactions(); ++n) {
      const Reaction& reaction = state.getReaction(n);
      // The sum of the reactant coefficients.
      sum = ads::computeSum(reaction.getReactants());
      // Loop over the reactant species.
      for (int i = 0; i != reaction.getReactants().size(); ++i) {
         index = reaction.getReactants().getIndex(i);
         order = reaction.getReactants()[i];
         if (sum > _highestOrder[index]) {
            _highestOrder[index] = sum;
         }
         if (order > _highestIndividualOrder[index]) {
            _highestIndividualOrder[index] = order;
         }
      }
   }
}



template<typename T, typename UnaryFunctor>
inline
void
TauLeaping<T, UnaryFunctor>::
step(State* state, const Number maximumTime) {
#ifdef _OPENMP
   const int ThreadNumber = omp_get_thread_num();
#else
   const int ThreadNumber = 0;
#endif

   assert(state->getTime() < maximumTime);

   // Compute the propensity functions.
   state->computePropensityFunctions();

   // Compute the time leap.
   T tau = computeTau(*state);
   // If this is the master thread.
   if (ThreadNumber == 0) {
      // If the time leap will take us past the maximum time.
      if (tau == std::numeric_limits<Number>::max() ||
            state->getTime() + tau > maximumTime) {
         tau = maximumTime - state->getTime();
         // Advance the time to the ending time.
         state->setTime(maximumTime);
      }
      else {
         // Advance the time by tau.
         state->advanceTime(tau);
      }
   }

   // Initialize the populations to prepare for firing.
   state->initializePopulations();

   // Advance the state.
#ifdef _OPENMP
   // For each reaction in this thread.
   const ads::IndexRange<1>& indexRange =
      state->getReactionIndexRange(ThreadNumber);
   const int Begin = indexRange.lbound();
   const int End = indexRange.ubound();
   for (int m = Begin; m != End; ++m) {
      state->fireReaction(m, _poisson[ThreadNumber]
                          (state->getPropensityFunction(m) * tau));
   }
#else
   // For each reaction.
   const int End = state->getNumberOfReactions();
   for (int m = 0; m != End; ++m) {
      state->fireReaction(m, _poisson(state->getPropensityFunction(m) * tau));
   }
#endif

   // Accumulate the populations from the different threads.
   // "Fix" any populations that have become negative.
   state->accumulateAndFixNegativePopulations();
}


// Advance the simulation to the specified time.
template<typename T, typename UnaryFunctor>
inline
int
TauLeaping<T, UnaryFunctor>::
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
TauLeaping<T, UnaryFunctor>::
simulate(State* state, const int numberOfSteps) {
   assert(numberOfSteps >= 0);
   for (int n = 0; n != numberOfSteps; ++n) {
      step(state);
      // If the simulation time has gone to infinity.
      if (state->getTime() == std::numeric_limits<Number>::max()) {
         break;
      }
   }
}



template<typename T, typename UnaryFunctor>
inline
typename TauLeaping<T, UnaryFunctor>::Number
TauLeaping<T, UnaryFunctor>::
computeTau(const State& state) {
#ifdef _OPENMP
   if (state.isNumberOfSpeciesSmall()) {
      return computeTauSingle(state);
   }
   // else
   return computeTauMulti(state);
#else
   return computeTauSingle(state);
#endif
}




template<typename T, typename UnaryFunctor>
inline
typename TauLeaping<T, UnaryFunctor>::Number
TauLeaping<T, UnaryFunctor>::
computeTauSingle(const State& state) {
#ifdef _OPENMP
   const int ThreadNumber = omp_get_thread_num();
#else
   const int ThreadNumber = 0;
#endif

   if (ThreadNumber == 0) {
#ifdef _OPENMP
      Number& tau = _tau[0];
      ads::Array<1, Number>& mu = _mu[0];
      ads::Array<1, Number>& sigmaSquared = _sigmaSquared[0];
#else
      Number& tau = _tau;
      ads::Array<1, Number>& mu = _mu;
      ads::Array<1, Number>& sigmaSquared = _sigmaSquared;
#endif

      assert(_epsilon > 0);

      // Compute mu and sigmaSquared.
      computeMuAndSigmaSquaredSingle(state);

      // Initialize tau to infinity.
      tau = std::numeric_limits<T>::max();

      // Take the minimum over the species.
      T numerator, temp, a, b;
      // Loop over all species.
      const int NumberOfSpecies = state.getNumberOfSpecies();
      for (int n = 0; n != NumberOfSpecies; ++n) {
         // If the n_th species is not a reactant in any reaction.
         if (_highestOrder[n] == 0) {
            // This species does not affect tau.
            continue;
         }
         numerator = std::max(_epsilon * state.getPopulation(n) /
                              computeG(state, n), 1.0);

         if (mu[n] != 0) {
            a = numerator / std::abs(mu[n]);
         }
         else {
            a = std::numeric_limits<T>::max();
         }

         if (sigmaSquared[n] != 0) {
            b = numerator * numerator / sigmaSquared[n];
         }
         else {
            b = std::numeric_limits<T>::max();
         }

         temp = std::min(a, b);
         if (temp < tau) {
            tau = temp;
         }
      }
   }

#ifdef _OPENMP
   {
      // Wait until _tau[0] has the correct value.
#pragma omp barrier
   }
   return _tau[0];
#else
   return _tau;
#endif
}




#ifdef _OPENMP
template<typename T, typename UnaryFunctor>
inline
typename TauLeaping<T, UnaryFunctor>::Number
TauLeaping<T, UnaryFunctor>::
computeTauMulti(const State& state) {
   const int ThreadNumber = omp_get_thread_num();

   assert(_epsilon > 0);

   // Compute mu and sigmaSquared.
   computeMuAndSigmaSquaredMulti(state);

   // Initialize tau to infinity.
   Number& tau = _tau[ThreadNumber];
   tau = std::numeric_limits<T>::max();

   // Take the minimum over the species.
   T numerator, temp, a, b;
   // Loop over the species for this thread.
   const ads::IndexRange<1>& indexRange =
      state.getSpeciesIndexRange(ThreadNumber);
   const int begin = indexRange.lbound();
   const int end = indexRange.ubound();
   for (int n = begin; n != end; ++n) {
      // If the n_th species is not a reactant in any reaction.
      if (_highestOrder[n] == 0) {
         // This species does not affect tau.
         continue;
      }
      numerator = std::max(_epsilon * state.getPopulation(n) /
                           computeG(state, n), 1.0);

      if (_mu[0][n] != 0) {
         a = numerator / std::abs(_mu[0][n]);
      }
      else {
         a = std::numeric_limits<T>::max();
      }

      if (_sigmaSquared[0][n] != 0) {
         b = numerator * numerator / _sigmaSquared[0][n];
      }
      else {
         b = std::numeric_limits<T>::max();
      }

      temp = std::min(a, b);
      if (temp < tau) {
         tau = temp;
      }
   }

   {
      // Wait for the tau's to be computed in each thread.
#pragma omp barrier
   }

   // Take the minimum over each thread.
   // If this is the master thread.
   if (ThreadNumber == 0) {
      // This is inexpensive, so it is not worthwile to use recursive doubling.
      _tau[0] = ads::computeMinimum(_tau);
   }

   {
      // Wait until _tau[0] has the correct value.
#pragma omp barrier
   }

   return _tau[0];
}
#endif


// Compute the g described in "Efficient step size selection for the
// tau-leaping simulation method".
template<typename T, typename UnaryFunctor>
inline
typename TauLeaping<T, UnaryFunctor>::Number
TauLeaping<T, UnaryFunctor>::
computeG(const State& state, const int speciesIndex) const {
   const int order = _highestOrder[speciesIndex];

   if (order == 1) {
      return 1.0;
   }
   else if (order == 2) {
      if (_highestIndividualOrder[speciesIndex] == 1) {
         return 2.0;
      }
      else if (_highestIndividualOrder[speciesIndex] == 2) {
         return 2.0 + 1.0 / (state.getPopulation(speciesIndex) - 1);
      }
   }
   else if (order == 3) {
      if (_highestIndividualOrder[speciesIndex] == 1) {
         return 3.0;
      }
      else if (_highestIndividualOrder[speciesIndex] == 2) {
         return 1.5 *(2.0 + 1.0 / (state.getPopulation(speciesIndex) - 1.0));
      }
      else if (_highestIndividualOrder[speciesIndex] == 3) {
         return 3.0 + 1.0 / (state.getPopulation(speciesIndex) - 1)
                + 2.0 / (state.getPopulation(speciesIndex) - 2);
      }
   }

   // Catch any other cases with an assertion failure.
   assert(false);
   return 0;
}





template<typename T, typename UnaryFunctor>
inline
void
TauLeaping<T, UnaryFunctor>::
computeMuAndSigmaSquaredSingle(const State& state) {
   typedef typename State::SparseArrayInt SparseArrayInt;

#ifdef _OPENMP
   const int ThreadNumber = omp_get_thread_num();
   assert(ThreadNumber == 0);
   // The arrays for this thread.
   ads::Array<1, Number>& mu = _mu[ThreadNumber];
   ads::Array<1, Number>& sigmaSquared = _sigmaSquared[ThreadNumber];
#else
   ads::Array<1, Number>& mu = _mu;
   ads::Array<1, Number>& sigmaSquared = _sigmaSquared;
#endif

   // Make sure the array is the correct size.
   assert(state.getNumberOfSpecies() == mu.size());
   assert(state.getNumberOfSpecies() == sigmaSquared.size());

   // Initialize.
   mu = 0.0;
   sigmaSquared = 0.0;

   T propensityFunction, value;
   int index;
   // For all reactions.
   const int NumberOfReactions = state.getNumberOfReactions();
   for (int m = 0; m != NumberOfReactions; ++m) {
      propensityFunction = state.getPropensityFunction(m);
      // CONTINUE
      if (!(propensityFunction >= 0)) {
         std::cerr << m << " " << propensityFunction << "\n";
         for (int m = 0; m != NumberOfReactions; ++m) {
            std::cerr << state.getPropensityFunction(m) << "\n";
         }
      }
      assert(propensityFunction >= 0);
      const SparseArrayInt& stateChange = state.getStateChangeVector(m);
      for (int i = 0; i != stateChange.size(); ++i) {
         index = stateChange.getIndex(i);
         value = stateChange[i];
         mu[index] += value * propensityFunction;
         sigmaSquared[index] += value * value * propensityFunction;
      }
   }
}



#ifdef _OPENMP
template<typename T, typename UnaryFunctor>
inline
void
TauLeaping<T, UnaryFunctor>::
computeMuAndSigmaSquaredMulti(const State& state) {
   typedef typename State::SparseArrayInt SparseArrayInt;

   const int ThreadNumber = omp_get_thread_num();

   // The arrays for this thread.
   ads::Array<1, Number>& mu = _mu[ThreadNumber];
   ads::Array<1, Number>& sigmaSquared = _sigmaSquared[ThreadNumber];

   // Make sure the array is the correct size.
   assert(state.getNumberOfSpecies() == mu.size());
   assert(state.getNumberOfSpecies() == sigmaSquared.size());

   // Initialize.
   mu = 0.0;
   sigmaSquared = 0.0;

   T propensityFunction, value;
   int index;
   // For each reaction in this thread.
   const ads::IndexRange<1>& indexRange =
      state.getReactionIndexRange(ThreadNumber);
   const int begin = indexRange.lbound();
   const int end = indexRange.ubound();
   for (int m = begin; m != end; ++m) {
      propensityFunction = state.getPropensityFunction(m);
      assert(propensityFunction >= 0);
      const SparseArrayInt& stateChange = state.getStateChangeVector(m);
      for (int i = 0; i != stateChange.size(); ++i) {
         index = stateChange.getIndex(i);
         value = stateChange[i];
         mu[index] += value * propensityFunction;
         sigmaSquared[index] += value * value * propensityFunction;
      }
   }

   {
      // Wait for all of the threads to compute mu and sigmaSquared.
#pragma omp barrier
   }

   // If this is the master thread.
   if (ThreadNumber == 0) {
      // Sum the results from each of the threads.
      // CONTINUE: Recursive double may pay off starting with 8 threads and
      // a large number of species.
      for (int i = 1; i != _mu.size(); ++i) {
         _mu[0] += _mu[i];
         _sigmaSquared[0] += _sigmaSquared[i];
      }
   }

   {
      // Wait for _mu[0] and sigmaSquared[0] to get the correct values.
#pragma omp barrier
   }
}
#endif


//---------------------------------------------------------------------------
// Interface functions.
//---------------------------------------------------------------------------


// Advance the state to the specified time.
template<typename UnaryFunctor, typename T>
inline
int
simulateWithTauLeaping(State<T>* state, const T epsilon, const T maximumTime,
                       const int seed) {
#ifdef _OPENMP
   const int ThreadNumber = omp_get_thread_num();
#else
   const int ThreadNumber = 0;
#endif

   // Construct the tau-leaping data structure.
   TauLeaping<T, UnaryFunctor>
   tauLeaping(state->getNumberOfSpecies(), epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);
   int numberOfSteps = 0;
   // Advance the simulation.
#pragma omp parallel
   {
      const int n = tauLeaping.simulate(state, maximumTime);
      // If this is the master thread.
      if (ThreadNumber == 0) {
         numberOfSteps = n;
      }
   }
   return numberOfSteps;
}



// Take the specified number of steps.
template<typename UnaryFunctor, typename T>
inline
void
simulateWithTauLeaping(State<T>* state, const T epsilon,
                       const int numberOfSteps, const int seed) {
   // Construct the tau-leaping data structure.
   TauLeaping<T, UnaryFunctor>
   tauLeaping(state->getNumberOfSpecies(), epsilon, seed);
   // Initialize the data structure.
   tauLeaping.initialize(*state);
   // Advance the simulation.
#pragma omp parallel
   tauLeaping.simulate(state, numberOfSteps);
}

} // namespace stochastic
