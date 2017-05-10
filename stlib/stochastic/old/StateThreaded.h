// -*- C++ -*-

/*!
  \file stochastic/StateThreaded.h
  \brief The state of the stochastic simulation.
*/

#if !defined(__stochastic_StateThreaded_h__)
#define __stochastic_StateThreaded_h__

#ifndef _OPENMP
#error This class may only be used with OpenMP enabled.
#endif

#include "StateReactions.h"

#include "stlib/numerical/partition.h"

#include <omp.h>

namespace stochastic {

//! The state of the stochastic simulation.
/*!
  Hold the time, the reaction count, the populations, and the state change
  vectors.

  \param _PopulationsContainer is the container for the populations.
  By default this is an ads::Array.
  \param _ScvContainer is the container for the state change vectors.
  By default this is an array of sparse arrays.
  \param T The number type.  The default it is \c double .
*/
template < class _PopulationsContainer = ads::Array<1, int>,
         class _ScvContainer = ads::Array<1, ads::SparseArray<1, int> >,
         typename T = double >
class StateThreaded {
   //
   // Public types.
   //
public:

   //! The number type.
   typedef T Number;
   //! The container for the populations.
   typedef _PopulationsContainer PopulationsContainer;
   //! The container for the state change vectors.
   typedef _ScvContainer ScvContainer;

   //
   // Member data.
   //
private:

   //! The time.
   Number _time;
   //! The total number of reaction firings.
   /*!
     The first element holds the count.  The rest are used for computing
     intermediate results within each thread.

     CONTINUE.  This probably causes a mild false sharing problem.
   */
   ads::Array<1, std::size_t> _totalReactionCounts;
   //! The populations of the species.  An array for each thread.
   /*!
     The first array holds the populations.  The rest are used for computing
     intermediate results within each thread.

     CONTINUE.  This probably causes a mild false sharing problem.  It is
     mild because the Poisson random variable is computed between population
     updates.  I should figure out if this is an important issue and if so,
     fix it.
   */
   ads::Array<1, ads::Array<1, int> > _populations;
   //! The number of reaction firings.
   ads::Array<1, ads::Array<1, std::size_t> > _reactionCounts;
   //! The state change vectors.
   ScvContainer _stateChangeVectors;

   //! The reaction index ranges for each thread.
   ads::Array<1, ads::IndexRange<1> > _reactionIndexRanges;
   //! The species index ranges for each thread.
   ads::Array<1, ads::IndexRange<1> > _speciesIndexRanges;
   //! Threshhold for a small number of reactions.
   int _smallNumberOfReactions;
   //! Threshhold for a small number of species.
   int _smallNumberOfSpecies;

   // CONTINUE HERE
   //
   // Not implemented.
   //
private:

   //! Copy constructor not implemented.
   StateThreaded(const StateThreaded&);
   //! Assignment operator not implemented.
   StateThreaded&
   operator=(const StateThreaded&);

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Default constructor.
   /*!
     \note This may only be called from the master thread.
   */
   StateThreaded();

   //! Construct from the number of species.
   /*!
     \note This may only be called from the master thread.
   */
   StateThreaded(int numberOfSpecies);

   //! Construct from the number of species.  Reserve memory for the specified number of reactions.
   /*!
     \note This may only be called from the master thread.
   */
   StateThreaded(int numberOfSpecies, int numberOfReactions);

   //! Destructor.
   /*!
     \note This may only be called from the master thread.
   */
   ~StateThreaded() {
      assert(omp_get_thread_num() == 0);
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the time.
   Number
   getTime() const {
      return _time;
   }

   //! Get the number of reaction firings.
   std::size_t
   getReactionCount() const {
      return _totalReactionCounts[0];
   }

   //! Get the number of species.
   int
   getNumberOfSpecies() const {
      return _populations[0].size();
   }

   //! Get the number of reactions.
   using Base::getNumberOfReactions;

   //! Get the populations.
   const ads::Array<1, int>&
   getPopulations() const {
      return _populations[0];
   }

   //! Get the specified population.
   int
   getPopulation(const int speciesIndex) const {
      return _populations[0][speciesIndex];
   }

   //! Get the specified reaction.
   using Base::getReaction;

   //! Get the beginning of the range of reactions.
   using Base::getReactionsBeginning;

   //! Get the end of the range of reactions.
   using Base::getReactionsEnd;

   //! Get the specified state change vector.
   using Base::getStateChangeVector;

   //! Compute the propensity function for the specified reaction.
   Number
   computePropensityFunction(const int reactionIndex) const {
      return getReaction(reactionIndex).computePropensityFunction
             (_populations[0]);
   }

   //! Get the reaction index range for the specified thread.
   const ads::IndexRange<1>&
   getReactionIndexRange(const int threadNumber) const {
      return _reactionIndexRanges[threadNumber];
   }

   //! Get the species index range for the specified thread.
   const ads::IndexRange<1>&
   getSpeciesIndexRange(const int threadNumber) const {
      return _speciesIndexRanges[threadNumber];
   }

   //! Get the threshhold for a small number of reactions.
   int
   getSmallNumberOfReactions() const {
      return _smallNumberOfReactions;
   }

   //! Get the threshhold for a small number of species.
   int
   getSmallNumberOfSpecies() const {
      return _smallNumberOfSpecies;
   }

   //! Return true if the number of reactions is small.
   bool
   isNumberOfReactionsSmall() const {
      return getNumberOfReactions() <= _smallNumberOfReactions;
   }

   //! Return true if the number of species is small.
   bool
   isNumberOfSpeciesSmall() const {
      return getNumberOfSpecies() <= _smallNumberOfSpecies;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the time.
   void
   setTime(const Number time) {
#ifdef DEBUG_stochastic_StateThreaded
      assert(omp_get_thread_num() == 0);
#endif
      _time = time;
   }

   //! Advance the time and return the new time.
   Number
   advanceTime(const Number increment) {
#ifdef DEBUG_stochastic_StateThreaded
      assert(omp_get_thread_num() == 0);
#endif
      return _time += increment;
   }

   // CONTINUE: Add functions for inserting a range of reactions.  This will
   // avoid the inefficiency of calling computeReactionIndexRanges for each
   // reaction.

   //! Insert a reaction.
   void
   insertReaction(const Reaction& reaction);

   //! Clear the reactions.
   void
   clearReactions();

   //! Set the populations.
   /*!
     \note This may change the number of species.
   */
   void
   setPopulations(const ads::Array<1, int>& populations);

   //! Set the specified population.
   void
   setPopulation(const int index, const int population) {
#ifdef DEBUG_stochastic_StateThreaded
      assert(omp_get_thread_num() == 0);
#endif
      _populations[0][index] = population;
   }

   //! Offset the populations.
   void
   offsetPopulations(const ads::Array<1, int>& change) {
#ifdef DEBUG_stochastic_StateThreaded
      assert(omp_get_thread_num() == 0);
#endif
      _populations[0] += change;
   }

   //! Initialize the populations for firing reactions.
   /*!
     The master thread holds the populations.  The rest of the threads have
     buffers.  To initialize, we set all of the buffer elements to zero.
   */
   void
   initializePopulations();

   //! Fire the specified reaction.
   void
   fireReaction(const int n) {
      const int ThreadNumber = omp_get_thread_num();
      ++_totalReactionCounts[ThreadNumber];
      _populations[ThreadNumber] += getStateChangeVector(n);
   }

   //! Fire the specified reaction the specified number of times.
   void
   fireReaction(const int reactionIndex, const int numberOfTimes) {
      const int ThreadNumber = omp_get_thread_num();
      _totalReactionCounts[ThreadNumber] += numberOfTimes;
      ads::scaleAdd(&_populations[ThreadNumber], numberOfTimes,
                    getStateChangeVector(reactionIndex));
   }

   // CONTINUE: I will need to replace this with the correct solution.
   //! Accumulate the populations after firing reactions and fix any negative populations (by making them zero).
   void
   accumulateAndFixNegativePopulations();

   //! Set the threshhold for a small number of reactions.
   void
   setSmallNumberOfReactions(const int n) {
      _smallNumberOfReactions = n;
   }

   //! Get the threshhold for a small number of species.
   void
   setSmallNumberOfSpecies(const int n) {
      _smallNumberOfSpecies = n;
   }

   //@}
   //--------------------------------------------------------------------------
   // Private member functions.
private:

   void
   computeReactionIndexRanges();

   void
   computeSpeciesIndexRanges();
};

//--------------------------------------------------------------------------
//! \defgroup stochastic_StateThreadedFunctions Free functions for StateThreaded.
//@{

//! Return true if the state is valid.
/*!
  \relates StateThreaded

  The species populations must be non-negative.  The reactions must be
  consistent with the number of species.
*/
template<typename T>
bool
isValid(const StateThreaded<T>& state);

//! Return true if the states are equal.
/*! \relates StateThreaded */
template<typename T>
bool
operator==(const StateThreaded<T>& x, const StateThreaded<T>& y);

//! Return true if the states are not equal.
/*! \relates StateThreaded */
template<typename T>
inline
bool
operator!=(const StateThreaded<T>& x, const StateThreaded<T>& y) {
   return !(x == y);
}

//! Write the reactions in ascii format.
/*! \relates StateThreaded */
template<typename T>
inline
void
writeReactionsAscii(std::ostream& out, const StateThreaded<T>& x) {
   writeReactionsAscii(out, static_cast<const StateReactions<T>&>(x));
}

//! Read the reactions in ascii format.
/*! \relates StateThreaded */
template<typename T>
inline
void
readReactionsAscii(std::istream& in, StateThreaded<T>* x) {
   readReactionsAscii(in, static_cast<StateReactions<T>*>(x));
}

//! Write the populations in ascii format.
/*! \relates StateThreaded */
template<typename T>
void
writePopulationsAscii(std::ostream& out, const StateThreaded<T>& x);

//! Read the populations in ascii format.
/*! \relates StateThreaded */
template<typename T>
void
readPopulationsAscii(std::istream& in, StateThreaded<T>* x);

//! Write the state in ascii format.
/*! \relates StateThreaded */
template<typename T>
void
writeAscii(std::ostream& out, const StateThreaded<T>& x);

//! Read the state in ascii format.
/*! \relates StateThreaded */
template<typename T>
void
readAscii(std::istream& in, StateThreaded<T>* x);

//@}

} // namespace stochastic

#define __stochastic_StateThreaded_ipp__
#include "StateThreaded.ipp"
#undef __stochastic_StateThreaded_ipp__

#endif
