// -*- C++ -*-

#if !defined(__sfc_DilateBits_h__)
#define __sfc_DilateBits_h__

/*!
  \file
  \brief Dilate the bits in an unsigned integer so they may be interleaved.
*/

#include <boost/mpl/if.hpp>

#include <array>
#include <limits>

#include <cstdint>

namespace stlib
{
namespace sfc
{

//! Functor for dilating bits.
template<std::size_t _Dimension>
class DilateBits
{
  // Constants.
private:

  //! The number of bits to expand.
  BOOST_STATIC_CONSTEXPR std::size_t _ExpandBits = 8;

  // Types.
private:

  //! Choose an integer type with sufficient bits to dilate 8 bits at a time.
  typedef typename boost::mpl::if_c<(_Dimension * _ExpandBits > 32), std::uint64_t,
          typename boost::mpl::if_c<(_Dimension * _ExpandBits > 16), std::uint32_t,
          std::uint16_t>::type>::type
          _DilatedInteger;

  // Member data.
private:

  std::array < _DilatedInteger, 1 << _ExpandBits > _expanded;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor and assignment operator. */
  //@{
public:

  //! Build the table for dilating bits.
  DilateBits();

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Dilate the bits in an integer.
  /*! Do this so that coordinates may be merged with a bitwise or to obtain
    a Morton code. For example, in 3-D, 1111 is transformed to 001001001001. */
  template<typename _Code>
  _Code
  operator()(_Code n, int nBits) const;

  //! Dilate the bits in an integer.
  /*! Specialization for 8-bit integers to avoid a compiler warning about the
    shift count exceeding the width of the type. */
  std::uint8_t
  operator()(const std::uint8_t n, int /*nBits*/) const
  {
    return _expanded[n];
  }

  //@}
};


//! Functor for dilating bits.
/*! Specialization for 1-D. */
template<>
class DilateBits<1>
{
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Dilate the bits in an integer.
  /*! This is trivial in 1-D. We just return the first argument. We do not
   check that bits past nBits are zero.*/
  template<typename _Code>
  _Code
  operator()(_Code n, int /*nBits*/) const
  {
    return n;
  }

  //@}
};

} // namespace sfc
}

#define __sfc_DilateBits_tcc__
#include "stlib/sfc/DilateBits.tcc"
#undef __sfc_DilateBits_tcc__

#endif
