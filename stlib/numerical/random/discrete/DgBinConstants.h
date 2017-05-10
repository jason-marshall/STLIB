// -*- C++ -*-

/*!
  \file numerical/random/discrete/DgBinConstants.h
  \brief Policy classes for handling bin constants.
*/

#if !defined(__numerical_DgBinConstants_h__)
#define __numerical_DgBinConstants_h__

#include <limits>

#include <cassert>
#include <cmath>
#include <cstddef>

namespace stlib
{
namespace numerical {

//! Bin constants can be set at run time.
class DgBinConstants {
   //
   // Public types.
   //
public:

   //! The number type.
   typedef double Number;

   //
   // Member data.
   //
private:

   std::size_t _indexBits;
   std::size_t _numberOfBins;
   unsigned _indexMask;
   Number _maxHeightInverse;

   //-------------------------------------------------------------------------
   /*! \name Constructors, etc.
     The big four are protected so that only derived classes can use them.
   */
   //@{
protected:

   //! Default constructor.
   /*!
     The default number of index bits is 8.
   */
   DgBinConstants() {
      setIndexBits(8);
   }

   //! Copy constructor.
   DgBinConstants(const DgBinConstants& other) :
      _indexBits(other._indexBits),
      _numberOfBins(other._numberOfBins),
      _indexMask(other._indexMask),
      _maxHeightInverse(other._maxHeightInverse) {}

   //! Assignment operator.
   DgBinConstants&
   operator=(const DgBinConstants& other) {
      if (&other != this) {
         _indexBits = other._indexBits;
         _numberOfBins = other._numberOfBins;
         _indexMask = other._indexMask;
         _maxHeightInverse = other._maxHeightInverse;
      }
      return *this;
   }

   //! Destructor.
   ~DgBinConstants() {}

   //@}
   //-------------------------------------------------------------------------
   //! \name Accessors.
   //@{
public:

   //! The number of bits used for indexing the bins.
   std::size_t
   getIndexBits() const {
      return _indexBits;
   }

   //! The number of bins. 2^IndexBits.
   std::size_t
   getNumberOfBins() const {
      return _numberOfBins;
   }

   //! The mask for extracting the index.
   /*!
     An unsigned representation of NumberOfBins - 1.
   */
   unsigned
   getIndexMask() const {
      return _indexMask;
   }

   //! The inverse of the maximum height.  1 / (2^(32 - IndexBits) - 1).
   Number
   getMaxHeightInverse() const {
      return _maxHeightInverse;
   }

   //@}
   //-------------------------------------------------------------------------
   //! \name Manipulators.
   //@{
public:

   //! Set the number of index bits.
   void
   setIndexBits(const std::size_t indexBits) {
      assert(indexBits < 32);
      _indexBits = indexBits;
      _numberOfBins = std::size_t(std::pow(Number(2), int(_indexBits)));
      _indexMask = _numberOfBins - 1;
      _maxHeightInverse = 1.0 / (std::numeric_limits<unsigned>::max() /
                                 _numberOfBins - 1);
   }

   //@}
};


} // namespace numerical
}

#endif
