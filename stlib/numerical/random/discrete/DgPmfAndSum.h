// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfAndSum.h
  \brief PMF and its sum.
*/

#if !defined(__numerical_DgPmfAndSum_h__)
#define __numerical_DgPmfAndSum_h__

#include "stlib/numerical/random/discrete/DgPmf.h"

namespace stlib
{
namespace numerical {

//! PMF and its sum.
/*!
*/
template<bool _Guarded>
class DgPmfAndSum :
   public DgPmf<_Guarded> {
   //
   // Private types.
   //
private:

   typedef DgPmf<_Guarded> Base;

   //
   // Protected types.
   //
protected:

   //! The number type.
   typedef double Number;
   //! Const iterator.
   typedef typename Base::const_iterator const_iterator;
   //! Iterator.
   typedef typename Base::iterator iterator;

   //
   // Member data.
   //
protected:

   //! Probability mass function.  (This is scaled and may not sum to unity.)
   using Base::_pmf;
   //! The sum of the PMF.
   Number _sum;
   //! The error in the sum of the PMF.
   Number _error;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfAndSum() :
      Base(),
      _sum(0),
      _error(0) {}

   //! Copy constructor.
   DgPmfAndSum(const DgPmfAndSum& other) :
      Base(other),
      _sum(other._sum),
      _error(other._error) {}

   //! Assignment operator.
   DgPmfAndSum&
   operator=(const DgPmfAndSum& other) {
      if (this != &other) {
         Base::operator=(other);
         _sum = other._sum;
         _error = other._error;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfAndSum() {}

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
   operator==(const DgPmfAndSum& other) const {
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
      _error += (_sum + value + _pmf[index]) *
                std::numeric_limits<Number>::epsilon();
      // Update the PMF sum with the difference between the new and old values.
      _sum += value - _pmf[index];
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

private:

   //! Recompute the sum of the PMF if necessary.
   void
   update() {
      // The allowed relative error is 2^-32.
      const Number allowedRelativeError = 2.3283064365386963e-10;
      if (_error > allowedRelativeError * _sum) {
         repair();
      }
   }

   //! Recompute the sum of the PMF.
   void
   repair() {
      _sum = std::accumulate(begin(), end(), 0.0);
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
