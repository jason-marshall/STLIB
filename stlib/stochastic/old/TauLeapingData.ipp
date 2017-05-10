// -*- C++ -*-

#if !defined(__stochastic_TauLeapingData_ipp__)
#error This file is an implementation detail of TauLeapingData.
#endif

namespace stochastic {

template<typename T>
inline
void
TauLeapingData<T>::
initialize(const State& state) {
   typedef typename State::Reaction Reaction;

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


template<typename T>
inline
typename TauLeapingData<T>::Number
TauLeapingData<T>::
computeTau(const State& state) {
#ifdef DEBUG_stochastic_TauLeapingData
   assert(_epsilon > 0);
#endif

   // Compute mu and sigmaSquared.
   computeMuAndSigmaSquared(state);

   // Initialize tau to infinity.
   T tau = std::numeric_limits<T>::max();

   // Take the minimum over the species.
   T numerator, temp, a, b;
   for (int n = 0; n < state.getNumberOfSpecies(); ++n) {
      // If the n_th species is not a reactant in any reaction.
      if (_highestOrder[n] == 0) {
         // This species does not affect tau.
         continue;
      }
      numerator = std::max(_epsilon * state.getPopulation(n) /
                           computeG(state, n), 1.0);

      if (_mu[n] != 0) {
         a = numerator / std::abs(_mu[n]);
      }
      else {
         a = std::numeric_limits<T>::max();
      }

      if (_sigmaSquared[n] != 0) {
         b = numerator * numerator / _sigmaSquared[n];
      }
      else {
         b = std::numeric_limits<T>::max();
      }

      temp = std::min(a, b);
      if (temp < tau) {
         tau = temp;
      }
   }

   return tau;
}


// Compute the g described in "Efficient step size selection for the
// tau-leaping simulation method".
template<typename T>
inline
typename TauLeapingData<T>::Number
TauLeapingData<T>::
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


template<typename T>
inline
void
TauLeapingData<T>::
computeMuAndSigmaSquared(const State& state) {
   typedef ads::SparseArray<1, int> SparseArrayInt;

   // Make sure the array is the correct size.
   assert(state.getNumberOfSpecies() == _mu.size());
   assert(state.getNumberOfSpecies() == _sigmaSquared.size());

   // Initialize.
   _mu = 0.0;
   _sigmaSquared = 0.0;

   T propensityFunction, value;
   int index;
   // Loop over the reactions.
   for (int m = 0; m < state.getNumberOfReactions(); ++m) {
      propensityFunction = state.computePropensityFunction(m);
      assert(propensityFunction >= 0);
      const SparseArrayInt& stateChange = state.getStateChangeVector(m);
      for (int i = 0; i != stateChange.size(); ++i) {
         index = stateChange.getIndex(i);
         value = stateChange[i];
         _mu[index] += value * propensityFunction;
         _sigmaSquared[index] += value * value * propensityFunction;
      }
   }
}

} // namespace stochastic
