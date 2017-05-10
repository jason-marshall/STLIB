// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgPmfOrderedPairPointer.h
  \brief Ordered probability mass function for a discrete generator.
*/

#if !defined(__numerical_DgPmfOrderedPairPointer_h__)
#define __numerical_DgPmfOrderedPairPointer_h__

#include "stlib/ads/iterator/TransformIterator.h"
#include "stlib/ads/functor/select.h"
#include "stlib/ext/pair.h"
#include "stlib/ext/vector.h"

#include <limits>

namespace stlib
{
namespace numerical
{

USING_STLIB_EXT_VECTOR_IO_OPERATORS;

//! Ordered probability mass function for a discrete generator.
/*!
  CONTINUE
*/
template<bool _Guarded>
class DgPmfOrderedPairPointer {
   //
   // Protected types.
   //
protected:

   //! The number type.
   typedef double Number;
   //! A pair of a PMF value and index.
   typedef std::pair<Number, std::size_t> PairValueIndex;
   //! The container of value/index pairs.
   typedef std::vector<PairValueIndex> Container;
   //! The value/index pair type.
   typedef Container::value_type value_type;
   //! Const iterator to value/index pairs.
   typedef Container::const_iterator const_iterator;
   //! Iterator to value/index pairs.
   typedef Container::iterator iterator;
   //! Const iterator to PMF values.
   typedef ads::TransformIterator < const_iterator,
           ads::Select1st<PairValueIndex> >
           PmfConstIterator;

   //
   // Nested classes.
   //
protected:

   //! Less then comparison.
   class ValueLess :
      public std::binary_function<value_type, value_type, bool> {
   public:
      //! Less then comparison.
      bool
      operator()(const value_type& x, const value_type& y) const {
         return x.first < y.first;
      }
   };

   //! Greater than comparison.
   class ValueGreater :
      public std::binary_function<value_type, value_type, bool> {
   public:
      //! Greater than comparison.
      bool
      operator()(const value_type& x, const value_type& y) const {
         return x.first > y.first;
      }
   };

   //
   // Member data.
   //
protected:

   //! Value/index pairs for the events in the PMF.
   Container _pmfPairs;
   //! The rank of the elements in _pmf array.
   /*!
     This is useful for manipulating the _pmf array by index.  \c *_pointers[i]
     is the i_th element in the original PMF array.
   */
   std::vector<iterator> _pointers;

   //--------------------------------------------------------------------------
   //! \name Constructors etc.
   //@{
protected:

   //! Default constructor.
   DgPmfOrderedPairPointer() :
      // The arrays are empty.
      _pmfPairs(std::size_t(_Guarded)),
      _pointers() {
      if (_Guarded) {
         // The guard element.
         _pmfPairs.back().first = 0.5 * std::numeric_limits<Number>::max();
         _pmfPairs.back().second = 0;
      }
   }

   //! Copy constructor.
   DgPmfOrderedPairPointer(const DgPmfOrderedPairPointer& other) :
      _pmfPairs(other._pmfPairs),
      _pointers(other._pointers) {}

   //! Assignment operator.
   DgPmfOrderedPairPointer&
   operator=(const DgPmfOrderedPairPointer& other) {
      if (this != &other) {
         _pmfPairs = other._pmfPairs;
         _pointers = other._pointers;
      }
      return *this;
   }

   //! Destructor.
   ~DgPmfOrderedPairPointer() {}

   //@}
   //--------------------------------------------------------------------------
   //! \name Accessors.
   //@{
protected:

   //! Get the number of possible deviates.
   std::size_t
   size() const {
      return _pointers.size();
   }

   //! Get the beginning of the value/index pairs.
   const_iterator
   begin() const {
      return _pmfPairs.begin();
   }

   //! Get the end of the value/index pairs.
   const_iterator
   end() const {
      return _pmfPairs.end() - _Guarded;
   }

   //! Get the beginning of the PMF.
   PmfConstIterator
   pmfBegin() const {
      return PmfConstIterator(_pmfPairs.begin());
   }

   //! Get the end of the PMF.
   PmfConstIterator
   pmfEnd() const {
      return PmfConstIterator(_pmfPairs.end());
   }

   //! Get the probability with the specified index.
   Number
   operator[](const std::size_t i) const {
      return _pointers[i]->first;
   }

   //! Return the position of the specified event in the ordered array.
   std::size_t
   position(const std::size_t i) const {
      return _pointers[i] - _pmfPairs.begin();
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Equality.
   //@{
protected:

   bool
   operator==(const DgPmfOrderedPairPointer& other) const {
      return _pmfPairs == other._pmfPairs && _pointers == other._pointers;
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
protected:

   //! Get the beginning of the value/index pairs.
   iterator
   begin() {
      return _pmfPairs.begin();
   }

   //! Get the end of the value/index pairs.
   iterator
   end() {
      return _pmfPairs.end() - _Guarded;
   }

   //! Set the probability mass function with the specified index.
   void
   set(std::size_t index, Number value) {
      _pointers[index]->first = value;
   }

   //! Initialize the probability mass function.
   template<typename ForwardIterator>
   void
   initialize(ForwardIterator start, ForwardIterator finish) {
      const std::size_t size = std::distance(start, finish);
      // Allocate memory for the value/index pair container.
      _pmfPairs.resize(size + _Guarded);
      if (_Guarded) {
         // The guard element.
         _pmfPairs.back().first = 0.5 * std::numeric_limits<Number>::max();
         _pmfPairs.back().second = size;
      }
      // Copy the PMF.
      for (std::size_t i = 0; i != size; ++i) {
         _pmfPairs[i].first = *start++;
         _pmfPairs[i].second = i;
      }
      // Allocate memory for the pointers.
      _pointers.resize(size);
      // Set the pointers.
      computePointers();
   }

   //! Compute the pointers.
   void
   computePointers() {
      for (iterator i = begin(); i != end(); ++i) {
         _pointers[i->second] = i;
      }
   }

   //@}
   //--------------------------------------------------------------------------
   //! \name File I/O.
   //@{
protected:

   //! Print information about the data structure.
   void
   print(std::ostream& out) const {
      out << "Value/index pairs = \n" << _pmfPairs << '\n'
          << "Ranks = \n";
      for (std::size_t i = 0; i != _pointers.size(); ++i) {
         out << position(i) << ' ';
      }
      out << '\n';
   }

   //@}
};

} // namespace numerical
}

#endif
