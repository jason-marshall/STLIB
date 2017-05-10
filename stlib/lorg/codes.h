// -*- C++ -*-

/*!
  \file lorg/codes.h
  \brief Convert discrete coordinates to spatial indices (codes).
*/

#if !defined(__lorg_codes_h__)
#define __lorg_codes_h__

#include "stlib/lorg/coordinates.h"

namespace stlib
{
namespace lorg
{

//! Convert between discrete coordinates and Morton codes.
/*!
  \param _Integer The unsigned integer type to use for coordinates.
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.
*/
template<typename _Integer, typename _Float, std::size_t _Dimension>
class Morton :
  public DiscreteCoordinates<_Integer, _Float, _Dimension>
{
  //
  // Constants.
  //
public:

  //! The number of bits to expand.
  BOOST_STATIC_CONSTEXPR std::size_t ExpandBits = 8;

  //
  // Types.
  //
private:

  typedef DiscreteCoordinates<_Integer, _Float, _Dimension> Base;

  //
  // Member data.
  //
private:

  std::array < _Integer, 1 << ExpandBits > _expanded;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the positions.
  Morton(const std::vector<typename Base::Point>& positions);

private:

  void
  buildExpanded();

  //@}
  //--------------------------------------------------------------------------
  //! \name Conversion.
  //@{
public:

  //! Return the Morton code for the Cartesian coordinates.
  _Integer
  code(const typename Base::Point& coords) const
  {
    return code(Base::discretize(coords));
  }

private:

  //! Return the Morton code for the discrete coordinates.
  _Integer
  code(const typename Base::DiscretePoint& coords) const
  {
    _Integer result = expand(coords[0]);
    for (std::size_t i = 0; i != _Dimension; ++i) {
      result |= expand(coords[i]) << i;
    }
    return result;
  }

  // Expand the bits in an integer so that the coordinates may be merged
  // with a bitwise or to obtain the Morton code. In 3-D, 1111 is transformed
  // to 001001001001.
  _Integer
  expand(std::size_t n) const;

  //@}
};



//! Convert between discrete coordinates and Morton codes.
/*!
  Specialization for 1-D.
*/
template<typename _Integer, typename _Float>
class Morton<_Integer, _Float, 1> :
  public DiscreteCoordinates<_Integer, _Float, 1>
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = 1;

  //
  // Types.
  //
private:

  typedef DiscreteCoordinates<_Integer, _Float, Dimension> Base;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the positions.
  Morton(const std::vector<typename Base::Point>& positions) :
    Base(positions)
  {
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Conversion.
  //@{
public:

  //! Return the Morton code for the Cartesian coordinates.
  _Integer
  code(const typename Base::Point& coords) const
  {
    return code(Base::discretize(coords));
  }

private:

  //! Return the Morton code for the discrete coordinates.
  _Integer
  code(const typename Base::DiscretePoint& coords) const
  {
    return coords[0];
  }

  //@}
};


} // namespace lorg
}

#define __lorg_codes_tcc__
#include "stlib/lorg/codes.tcc"
#undef __lorg_codes_tcc__

#endif
