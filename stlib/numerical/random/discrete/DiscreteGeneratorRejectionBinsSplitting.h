// -*- C++ -*-

/*!
  \file numerical/random/discrete/DiscreteGeneratorRejectionBinsSplitting.h
  \brief discrete deviate.  Rejection with bins and splitting.
*/

#if !defined(__numerical_DiscreteGeneratorRejectionBinsSplitting_h__)
#define __numerical_DiscreteGeneratorRejectionBinsSplitting_h__

#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplittingStatic.h"

#include <boost/config.hpp>

namespace stlib
{
namespace numerical {

//! Discrete deviate.  Bins with splitting.
/*!
  Dynamic specialization.
*/
template < bool ExactlyBalance = true,
         class Generator = DISCRETE_UNIFORM_GENERATOR_DEFAULT >
class DiscreteGeneratorRejectionBinsSplitting :
   public DiscreteGeneratorRejectionBinsSplittingStatic<ExactlyBalance, Generator> {
   //
   // Private types.
   //
private:

   //! The static version of this data structure.
   typedef DiscreteGeneratorRejectionBinsSplittingStatic
   <ExactlyBalance, Generator> Base;

   //
   // Public constants.
   //
public:

   //! The sum of the PMF is automatically updated.
   BOOST_STATIC_CONSTEXPR bool AutomaticUpdate = true;

   //
   // Public types.
   //
public:

   //! The discrete uniform generator.
   typedef typename Base::DiscreteUniformGenerator DiscreteUniformGenerator;
   //! The number type.
   typedef typename Base::Number Number;
   //! The argument type.
   typedef typename Base::argument_type argument_type;
   //! The result type.
   typedef typename Base::result_type result_type;

   //
   // Member data.
   //

private:

   //! An upper bound on the height of the bins.
   using Base::_heightUpperBound;
   //! The indices of the deviate in the bin.
   using Base::_deviateIndices;

   //! The sum of the PMF.
   using Base::_sum;
   //! Probability mass function.  (This is scaled and may not sum to unity.)
   using Base::_pmf;

   //! The error in the sum of the PMF.
   Number _error;
   //! The minimum allowed efficiency.
   Number _minimumEfficiency;
   //! The minimum allowed efficiency is the initial efficiency times this factor.
   Number _minimumEfficiencyFactor;

   //
   // Not implemented.
   //

private:

   //! Default constructor not implemented.
   DiscreteGeneratorRejectionBinsSplitting();

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
public:

   //! Construct using the uniform generator.
   explicit
   DiscreteGeneratorRejectionBinsSplitting
   (DiscreteUniformGenerator* generator) :
      Base(generator),
      _error(0),
      _minimumEfficiency(-1),
      // CONTINUE
      _minimumEfficiencyFactor(0.75) {}

   //! Construct from the probability mass function.
   template<typename ForwardIterator>
   DiscreteGeneratorRejectionBinsSplitting
   (DiscreteUniformGenerator* generator,
    ForwardIterator begin, ForwardIterator end) :
      Base(generator),
      _error(0),
      _minimumEfficiency(-1),
      _minimumEfficiencyFactor(0.75) {
      initialize(begin, end);
   }

   //! Copy constructor.
   /*!
     \note The discrete, uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorRejectionBinsSplitting
   (const DiscreteGeneratorRejectionBinsSplitting& other) :
      Base(other),
      _error(other._error),
      _minimumEfficiency(other._minimumEfficiency),
      _minimumEfficiencyFactor(other._minimumEfficiencyFactor) {}

   //! Assignment operator.
   /*!
     \note The discrete,uniform generator is not copied.  Only the pointer
     to it is copied.
   */
   DiscreteGeneratorRejectionBinsSplitting&
   operator=(const DiscreteGeneratorRejectionBinsSplitting& other) {
      if (this != &other) {
         Base::operator=(other);
         _error = other._error;
         _minimumEfficiency = other._minimumEfficiency;
         _minimumEfficiencyFactor = other._minimumEfficiencyFactor;
      }
      return *this;
   }

   //! Destructor.
   /*!
     The memory for the discrete, uniform generator is not freed.
   */
   ~DiscreteGeneratorRejectionBinsSplitting() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Random number generation.
   //@{
public:

   //! Seed the uniform random number generator.
   using Base::seed;

   //! Return a discrete deviate.
   result_type
   operator()() {
      if (computeEfficiency() < getMinimumEfficiency()) {
         rebuild();
      }
      return Base::operator()();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! Get the probability with the specified index.
   using Base::operator[];

   //! Get the number of possible deviates.
   using Base::size;

   //! Get the sum of the probability mass functions.
   using Base::sum;

   //! Compute the efficiency of the method.
   using Base::computeEfficiency;

   //! Compute the bin height for the given probability.
   using Base::computeBinHeight;

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      // Recompute the PMF sum if necessary.
      update();
      return Base::isValid();
   }

   //! Get the minimum allowed efficiency.
   Number
   getMinimumEfficiency() const {
      return _minimumEfficiency;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end);

   //! Set the number of index bits.
   using Base::setIndexBits;

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value);

   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   set(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         set(i, iterator[i]);
      }
   }

   //! Set the minimum allowed efficiency factor.
   void
   setMinimumEfficiencyFactor(const Number efficiency) const {
      _minimumEfficiencyFactor = efficiency;
   }

private:

   //! Update the data structure following calls to set().
   void
   update();

   //! Repair the data structure.
   /*!
     Recompute the PMF data.
   */
   void
   repair();

   //! Rebuild the bins.
   void
   rebuild();

   //! Update the minimum allowed efficiency.
   void
   updateMinimumAllowedEfficiency() {
      // Set the minimum allowed efficiency.
      _minimumEfficiency = computeEfficiency() * _minimumEfficiencyFactor;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
public:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const;

   //@}
};

} // namespace numerical
}

#define __numerical_random_DiscreteGeneratorRejectionBinsSplitting_ipp__
#include "stlib/numerical/random/discrete/DiscreteGeneratorRejectionBinsSplitting.ipp"
#undef __numerical_random_DiscreteGeneratorRejectionBinsSplitting_ipp__

#endif
