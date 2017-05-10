// -*- C++ -*-

#if !defined(__sfc_Traits_h__)
#define __sfc_Traits_h__

/*!
  \file
  \brief Traits that are shared amongst the data structures.
*/

#include "stlib/sfc/MortonOrder.h"

#include "stlib/geom/kernel/BBox.h"

namespace stlib
{
namespace sfc
{


//! Traits that are shared amongst the data structures.
template<std::size_t _Dimension = 3, typename _Float = float,
         typename _Code = std::uint64_t,
         template<std::size_t, typename> class _Order = MortonOrder>
struct Traits {
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The floating-point number type.
  typedef _Float Float;
  //! The unsigned integer type is used for coordinates and codes.
  typedef _Code Code;
  //! The spatial ordering.
  typedef _Order<Dimension, Code> Order;

  //! A Cartesian point.
  typedef std::array<Float, Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<Float, Dimension> BBox;
  //! The guard value for the codes is greater than any valid code.
  /*! It is also at least as large as the next code for any valid code at
    any level. */
  BOOST_STATIC_CONSTEXPR Code GuardCode = Code(-1);

  static_assert(Dimension > 0, "Dimension must be positive.");
  static_assert(std::is_integral<Code>::value,
                "The code type must be integral.");
  static_assert(std::is_unsigned<Code>::value,
                "The code type must be unsigned.");
  static_assert(std::is_floating_point<Float>::value,
                "Float must be a floating-point type.");
};


} // namespace sfc
} // namespace stlib

#endif
