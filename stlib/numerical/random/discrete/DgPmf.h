// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmf.h
  \brief Probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmf_h__)
#define __numerical_DgPmf_h__

#include "stlib/ext/vector.h"

#include <limits>

namespace stlib
{
namespace numerical {

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! Probability mass function for a discrete generator.
/*!
  Manage the probability mass function.
*/
template<bool _Guarded>
class DgPmf {
   //
   // Protected types.
   //
protected:

   //! The number type.
   typedef double Number;
   //! The array type.
   typedef std::vector<Number> Container;
   //! Const iterator.
   typedef Container::const_iterator const_iterator;
   //! Iterator.
   typedef Container::iterator iterator;

   //
   // Member data.
   //
protected:

   //! Probability mass function.  (This is scaled and may not sum to unity.)
   Container _pmf;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmf() :
      // The PMF array is empty.
      _pmf(std::size_t(_Guarded)) {
      if (_Guarded) {
         // The guard element.
         _pmf.back() = 0.5 * std::numeric_limits<Number>::max();
      }
   }

   //! Copy constructor.
   DgPmf(const DgPmf& other) :
      _pmf(other._pmf) {}

   //! Assignment operator.
   DgPmf&
   operator=(const DgPmf& other) {
      if (this != &other) {
         _pmf = other._pmf;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmf() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Get the probability with the specified index.
   Number
   operator[](const std::size_t i) const {
      return _pmf[i];
   }

   //! Get the beginning of the PMF.
   const_iterator
   begin() const {
      return _pmf.begin();
   }

   //! Get the end of the PMF.
   const_iterator
   end() const {
      return _pmf.end() - _Guarded;
   }

   //! Get the number of possible deviates.
   std::size_t
   size() const {
      return _pmf.size() - _Guarded;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
protected:

   bool
   operator==(const DgPmf& other) const {
      return _pmf == other._pmf;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Get the beginning of the value/index pairs.
   iterator
   begin() {
      return _pmf.begin();
   }

   //! Get the end of the value/index pairs.
   iterator
   end() {
      return _pmf.end() - _Guarded;
   }

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      _pmf[index] = value;
   }

#if 0
   //! Set the probability mass functions.
   template<typename _RandomAccessIterator>
   void
   set(_RandomAccessIterator iterator) {
      for (std::size_t i = 0; i != _pmf.size(); ++i) {
         _pmf[i] = iterator[i];
      }
   }
#endif

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator begin, ForwardIterator end) {
      _pmf.resize(std::distance(begin, end) + _Guarded);
      if (_Guarded) {
         // The guard element.
         _pmf.back() = 0.5 * std::numeric_limits<Number>::max();
      }
      std::copy(begin, end, _pmf.begin());
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "PMF = \n" << _pmf << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif
