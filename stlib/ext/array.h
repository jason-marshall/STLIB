// -*- C++ -*-

#if !defined(__ext_array_h__)
#define __ext_array_h__

#include "stlib/ext/arrayStd.h"

namespace stlib
{
namespace ext
{
//-------------------------------------------------------------------------
/*! \defgroup extArrayMake Convenience Constructor Functions.

  These functions are useful for constructing std::array's.
  \verbatim
  typedef std::array<double, 3> Point;
  Point x;
  ...
  // Convert to an array with a different value type.
  std::array<int, 3> a = ext::convert_array<int>(x);
  // Make an array filled with a specified value.
  x = ext::filled_array<Point>(0.);
  // Copy from a different kind of array.
  double y[3] = {2, 3, 5};
  x = ext::copy_array(y);
  // Make an array by specifying the elements.
  std::array<double, 1> x1;
  x1 = ext::make_array(2);
  std::array<double, 2> x2;
  x2 = ext::make_array(2, 3);
  std::array<double, 3> x3;
  x3 = ext::make_array(2, 3, 5);
  std::array<double, 4> x4;
  x4 = ext::make_array(2, 3, 5, 7); \endverbatim
*/
//@{


//! Convert to an array with a different value type.
/*! 
  This class performs the trivial conversion more efficiently than a 
  function interface.
*/
template<typename _T>
struct
ConvertArray {
  //! Convert from an array with a different value type.
  template<typename _T2, std::size_t N>
  static
  std::array<_T, N>
  convert(std::array<_T2, N> const& x) {
    std::array<_T, N> result;
    for (std::size_t i = 0; i != N; ++i) {
      result[i] = static_cast<_T>(x[i]);
    }
    return result;
  }

  //! "Convert" from an array with the same value type.
  template<std::size_t N>
  static
  std::array<_T, N> const&
  convert(std::array<_T, N> const& x) {
    return x;
  }
};


//! Convert to an array with different value type.
template<typename _Target, typename _Source, std::size_t N>
inline
std::array<_Target, N>
convert_array(const std::array<_Source, N>& x)
{
  std::array<_Target, N> result;
  for (std::size_t i = 0; i != N; ++i) {
    result[i] = static_cast<_Target>(x[i]);
  }
  return result;
}

//! Return an array filled with the specified value.
template<typename _Array>
inline
_Array
filled_array(const typename _Array::value_type& value)
{
  _Array x;
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] = value;
  }
  return x;
}

//! Copy from the input iterator to make an array.
template<typename _Array, typename _InputIterator>
inline
_Array
copy_array(_InputIterator input)
{
  _Array x;
  for (std::size_t i = 0; i != x.size(); ++i) {
    x[i] = *input++;
  }
  return x;
}


//@}

} // namespace ext
} // namespace stlib

#endif
