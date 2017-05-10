// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfAndSumOrderedPairPointer.h
  \brief PMF and its sum.
*/

#if !defined(__numerical_DgPmfAndSumOrderedPairPointer_h__)
#define __numerical_DgPmfAndSumOrderedPairPointer_h__

#include "stlib/numerical/random/discrete/DgPmfOrderedPairPointer.h"

namespace stlib
{
namespace numerical {

//! Ordered probability mass function and its sum.
/*!
*/
template<bool _Guarded>
class DgPmfAndSumOrderedPairPointer :
   public DgPmfOrderedPairPointer<_Guarded> {
   //
   // Private types.
   //
private:

   typedef DgPmfOrderedPairPointer<_Guarded> Base;

   //
   // Protected types.
   //
protected:

   //! The number type.
   typedef typename Base::Number Number;
   //! The value/index pair type.
   typedef typename Base::value_type value_type;
   //! Const iterator.
   typedef typename Base::const_iterator const_iterator;
   //! Iterator.
   typedef typename Base::iterator iterator;

   //
   // Nested classes.
   //
protected:

   using typename Base::ValueLess;
   using typename Base::ValueGreater;

   //
   // Member data.
   //
protected:

   //! Value/index pairs for the events in the PMF.
   using Base::_pmfPairs;
   //! The sum of the PMF.
   Number _sum;
   //! The error in the sum of the PMF.
   Number _error;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfAndSumOrderedPairPointer() :
      Base(),
      _sum(0),
      _error(0) {}

   //! Copy constructor.
   DgPmfAndSumOrderedPairPointer(const DgPmfAndSumOrderedPairPointer& other) :
      Base(other),
      _sum(other._sum),
      _error(other._error) {}

   //! Assignment operator.
   DgPmfAndSumOrderedPairPointer&
   operator=(const DgPmfAndSumOrderedPairPointer& other) {
      if (this != &other) {
         Base::operator=(other);
         _sum = other._sum;
         _error = other._error;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfAndSumOrderedPairPointer() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Get the probability with the specified index.
   using Base::operator[];
   //! Get the beginning of the PMF.
   using Base::begin;
   //! Get the end of the PMF.
   using Base::end;
   //! Get the number of possible deviates.
   using Base::size;

   //! Get the sum of the probability mass functions.
   Number
   sum() const {
      return _sum;
   }

   //! Return true if the sum of the PMF is positive.
   bool
   isValid() {
      update();
      return _sum > 0;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
protected:

   bool
   operator==(const DgPmfAndSumOrderedPairPointer& other) const {
      return Base::operator==(other) && _sum == other._sum &&
             _error == other._error;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Get the beginning of the PMF.
   //using Base::begin;
   //! Get the end of the PMF.
   //using Base::end;

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      // Update the error in the PMF sum.
      _error += (_sum + value + operator[](index)) *
                std::numeric_limits<Number>::epsilon();
      // Update the PMF sum with the difference between the new and old values.
      _sum += value - operator[](index);
      // Set the PMF value.
      Base::set(index, value);
   }

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      // Initialize the PMF.
      Base::initialize(begin, end);
      // Compute the sum.
      repair();
   }

protected:

   //! Recompute the sum of the PMF if necessary.
   void
   update() {
      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_error > allowedRelativeError * _sum) {
         repair();
      }
   }

private:

   //! Recompute the sum of the PMF.
   void
   repair() {
      // Compute the PMF sum.
      _sum = 0;
      const const_iterator finish = end();
      for (const_iterator i = begin(); i != finish; ++i) {
         _sum += i->first;
      }
      // The initial error in the sum.
      _error = size() * _sum * std::numeric_limits<Number>::epsilon();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      Base::print(out);
      out << "PMF sum = " << _sum << "\n"
          << "Error in the PMF sum = " << _error << "\n";
   }

   //@}
};

} // namespace numerical
}

#endif
