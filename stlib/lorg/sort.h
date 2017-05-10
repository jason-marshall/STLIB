// -*- C++ -*-

/*!
  \file lorg/sort.h
  \brief Sort a vector of code/value pairs.
*/

#if !defined(__lorg_sort_h__)
#define __lorg_sort_h__

#include "stlib/ads/algorithm/insertion_sort.h"

#include <boost/config.hpp>

#include <array>
#include <limits>
#include <vector>

#include <cstring>
#include <cassert>

namespace stlib
{
namespace lorg
{

//! Radix/Counting/Insertion sort.
/*!
  \param _Integer An unsigned integer.
  \param _T Arbitrary associated data.
*/
template<typename _Integer, typename _T>
class RciSort
{
  //
  // Types.
  //
public:

  //! The value type for the vector.
  typedef std::pair<_Integer, _T> Value;

  //
  // Constants.
  //
private:

  BOOST_STATIC_CONSTEXPR int RadixBits = 8;
  BOOST_STATIC_CONSTEXPR std::size_t Radix = 1 << RadixBits;
  BOOST_STATIC_CONSTEXPR _Integer Mask = Radix - 1;

  //
  // Nested classes.
  //
private:

  // Using a comparison function for the insertion sort is slightly slower.
#if 0
  struct
      LessThanFirst {
    bool
    operator()(const Value& x, const Value& y)
    {
      return x.first < y.first;
    }
  };
#endif

  //
  // Member data.
  //
private:

  std::vector<Value>* _pairs;
  std::vector<Value> _buffer;
  std::array<Value*, Radix> _insertIterators;
  int _digits;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Allocate member data.
  /*!
   \param pairs The vector of integer/value pairs.
   \param digits The number of digits in the _Integer type that are actually
   used. For some applications, some of the most significant bits are not
   used. In these cases, specify the number of bits to accelerate sorting. */
  RciSort(std::vector<Value>* pairs,
          int digits = std::numeric_limits<_Integer>::digits);

  //! Sort the vector.
  void
  sort()
  {
    if (_digits != 0) {
      _sort(0, _pairs->size(), _digits - RadixBits);
    }
  }

private:

  void
  _sort(std::size_t begin, std::size_t end, int shift);

  //@}
};

} // namespace lorg
}

#define __lorg_sort_tcc__
#include "stlib/lorg/sort.tcc"
#undef __lorg_sort_tcc__

#endif
