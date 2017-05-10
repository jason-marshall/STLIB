// -*- C++ -*-

#if !defined(__stochastic_StateThreaded_ipp__)
#error This file is an implementation detail of StateThreaded.
#endif

namespace stochastic {

//----------------------------------------------------------------------------
// Constructors.
//----------------------------------------------------------------------------

template<typename T>
inline
StateThreaded<T>::
StateThreaded() :
   Base(),
   _time(0),
   _totalReactionCounts(),
   _populations(),
   _reactionIndexRanges(),
   _speciesIndexRanges(),
   _smallNumberOfReactions(1000),
   _smallNumberOfSpecies(1000) {
   assert(omp_get_thread_num() == 0);

   int numberOfThreads = 0;
#pragma omp parallel
   if (omp_get_thread_num() == 0) {
      numberOfThreads = omp_get_num_threads();
   }
   assert(numberOfThreads > 0);

   _totalReactionCounts.resize(numberOfThreads);
   _totalReactionCounts = 0;
   _populations.resize(numberOfThreads);
   _reactionIndexRanges.resize(numberOfThreads);
   _speciesIndexRanges.resize(numberOfThreads);
}


template<typename T>
inline
StateThreaded<T>::
StateThreaded(const int numberOfSpecies) :
   Base(),
   _time(0),
   _totalReactionCounts(),
   _populations(),
   _reactionIndexRanges(),
   _speciesIndexRanges(),
   _smallNumberOfReactions(1000),
   _smallNumberOfSpecies(1000) {
   assert(omp_get_thread_num() == 0);

   int numberOfThreads = 0;
#pragma omp parallel
   if (omp_get_thread_num() == 0) {
      numberOfThreads = omp_get_num_threads();
   }
   assert(numberOfThreads > 0);

   _totalReactionCounts.resize(numberOfThreads);
   _totalReactionCounts = 0;
   _populations.resize(numberOfThreads);
   _reactionIndexRanges.resize(numberOfThreads);
   _speciesIndexRanges.resize(numberOfThreads);

   for (int i = 0; i != _populations.size(); ++i) {
      _populations[i].resize(numberOfSpecies);
   }
   computeSpeciesIndexRanges();
}



template<typename T>
inline
StateThreaded<T>::
StateThreaded(const int numberOfSpecies, const int numberOfReactions) :
   Base(numberOfReactions),
   _time(0),
   _totalReactionCounts(),
   _populations(),
   _reactionIndexRanges(),
   _speciesIndexRanges(),
   _smallNumberOfReactions(1000),
   _smallNumberOfSpecies(1000) {
   assert(omp_get_thread_num() == 0);

   int numberOfThreads = 0;
#pragma omp parallel
   if (omp_get_thread_num() == 0) {
      numberOfThreads = omp_get_num_threads();
   }
   assert(numberOfThreads > 0);

   _totalReactionCounts.resize(numberOfThreads);
   _totalReactionCounts = 0;
   _populations.resize(numberOfThreads);
   _reactionIndexRanges.resize(numberOfThreads);
   _speciesIndexRanges.resize(numberOfThreads);

   for (int i = 0; i != _populations.size(); ++i) {
      _populations[i].resize(numberOfSpecies);
   }
   computeSpeciesIndexRanges();
}


//----------------------------------------------------------------------------
// Manipulators.
//----------------------------------------------------------------------------


template<typename T>
inline
void
StateThreaded<T>::
setPopulations(const ads::Array<1, int>& populations) {
   assert(omp_get_thread_num() == 0);
   // Set the populations.
   _populations[0] = populations;
   // Resize the arrays that are used in the other threads.
   for (int i = 1; i != _populations.size(); ++i) {
      _populations[i].resize(populations.size());
   }
   computeSpeciesIndexRanges();
}


template<typename T>
inline
void
StateThreaded<T>::
insertReaction(const Reaction& reaction) {
#ifdef DEBUG_stochastic_StateThreaded
   assert(omp_get_thread_num() == 0);
#endif
   Base::insertReaction(reaction);
   computeReactionIndexRanges();
}


template<typename T>
inline
void
StateThreaded<T>::
clearReactions() {
#ifdef DEBUG_stochastic_StateThreaded
   assert(omp_get_thread_num() == 0);
#endif
   Base::clearReactions();
   computeReactionIndexRanges();
}


template<typename T>
inline
void
StateThreaded<T>::
initializePopulations() {
   const int ThreadNumber = omp_get_thread_num();
   // If this is not the master thread.
   if (ThreadNumber != 0) {
      _populations[ThreadNumber] = 0;
   }
   // There is no need for a barrier here.  We are working with data that is
   // logically private to this thread.
}


template<typename T>
inline
void
StateThreaded<T>::
accumulateAndFixNegativePopulations() {
   {
      // Wait for any previous calculations to complete.
#pragma omp barrier
   }

   const int ThreadNumber = omp_get_thread_num();
   // CONTINUE: Eventually I will want to do this with recursive doubling.
   // The range of species for this thread.
   const int begin = _speciesIndexRanges[ThreadNumber].lbound();
   const int end = _speciesIndexRanges[ThreadNumber].ubound();
   // Accumulate the reaction counts and the populations to the root.
   for (int i = 1; i != _populations.size(); ++i) {
      _totalReactionCounts[0] += _totalReactionCounts[i];
      _totalReactionCounts[i] = 0;
      for (int n = begin; n != end; ++n) {
         _populations[0][n] += _populations[i][n];
      }
   }
   // Fix the negative populations.
   for (int n = begin; n != end; ++n) {
      if (_populations[0][n] < 0) {
         _populations[0][n] = 0;
      }
   }

   {
      // Wait for the populations to be summed.
#pragma omp barrier
   }
}


//----------------------------------------------------------------------------
// Private member functions
//----------------------------------------------------------------------------


template<typename T>
inline
void
StateThreaded<T>::
computeReactionIndexRanges() {
   assert(omp_get_thread_num() == 0);

   const int NumberOfThreads = _reactionIndexRanges.size();
   int lower, upper;
   for (int i = 0; i != NumberOfThreads; ++i) {
      numerical::partitionRange(getNumberOfReactions(), NumberOfThreads, i,
                                &lower, &upper);
      _reactionIndexRanges[i].set_lbound(lower);
      _reactionIndexRanges[i].set_ubound(upper);
   }
}


template<typename T>
inline
void
StateThreaded<T>::
computeSpeciesIndexRanges() {
   assert(omp_get_thread_num() == 0);

   const int NumberOfThreads = _speciesIndexRanges.size();
   int lower, upper;
   for (int i = 0; i != NumberOfThreads; ++i) {
      numerical::partitionRange(getNumberOfSpecies(), NumberOfThreads, i,
                                &lower, &upper);
      _speciesIndexRanges[i].set_lbound(lower);
      _speciesIndexRanges[i].set_ubound(upper);
   }
}

//----------------------------------------------------------------------------
// Free functions.
//----------------------------------------------------------------------------

template<typename T>
inline
bool
isValid(const StateThreaded<T>& state) {
   // Check that there are no negative populations.
   if (ads::computeMinimum(state.getPopulations()) < 0) {
      return false;
   }
   // Check the reactions.
   for (int i = 0; i != state.getNumberOfReactions(); ++i) {
      if (! isValid(state.getReaction(i), state.getNumberOfSpecies())) {
         return false;
      }
   }
   return true;
}


// Return true if the states are equal.
template<typename T>
inline
bool
operator==(const StateThreaded<T>& x, const StateThreaded<T>& y) {
   // Check the time.
   if (x.getTime() != y.getTime()) {
      return false;
   }
   // Check the reactions.
   if (static_cast<const StateReactions<T>&>(x) !=
         static_cast<const StateReactions<T>&>(y)) {
      return false;
   }
   // Check the populations.
   if (x.getPopulations() != y.getPopulations()) {
      return false;
   }

   return true;
}


// Write the populations in ascii format.
template<typename T>
inline
void
writePopulationsAscii(std::ostream& out, const StateThreaded<T>& x) {
   out << x.getPopulations();
}


// Read the populations in ascii format.
template<typename T>
inline
void
readPopulationsAscii(std::istream& in, StateThreaded<T>* state) {
   ads::Array<1, int> populations;
   in >> populations;
   state->setPopulations(populations);
}


// Write the state in ascii format.
template<typename T>
inline
void
writeAscii(std::ostream& out, const StateThreaded<T>& state) {
   // The time.
   out << state.getTime() << "\n";
   out << state.getPopulations();
   writeReactionsAscii(out, state);
}


// Read the state in ascii format.
template<typename T>
inline
void
readAscii(std::istream& in, StateThreaded<T>* state) {
   // The time.
   T time;
   in >> time;
   state->setTime(time);

   // The populations.
   ads::Array<1, int> populations;
   in >> populations;
   state->setPopulations(populations);

   // The reactions.
   readReactionsAscii(in, state);
}

} // namespace stochastic {
