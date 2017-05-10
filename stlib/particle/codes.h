// -*- C++ -*-

/*!
  \file particle/codes.h
  \brief Convert discrete coordinates to spatial indices (codes).
*/

#if !defined(__particle_codes_h__)
#define __particle_codes_h__

#include "stlib/particle/coordinates.h"

#include <cassert>

namespace stlib
{
namespace particle
{

//! Convert between discrete coordinates and Morton codes.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.
  \param _Periodic Whether the domain is periodic.
*/
template<typename _Float, std::size_t _Dimension, bool _Periodic>
class Morton :
  public DiscreteCoordinates<_Float, _Dimension, _Periodic>
{
  //
  // Constants.
  //
public:

  //! The number of bits to expand.
  BOOST_STATIC_CONSTEXPR std::size_t ExpandBits = 8;
  //! The number of bits for each dimension.
  BOOST_STATIC_CONSTEXPR std::size_t SeparateBits = 1 + 6 / _Dimension;

  //
  // Types.
  //
private:

  typedef DiscreteCoordinates<_Float, _Dimension, _Periodic> Base;

public:

  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;
  //! A discrete point with integer coordinates.
  typedef typename Base::DiscretePoint DiscretePoint;

  //
  // Member data.
  //
private:

  std::array < Code, 1 << ExpandBits > _expanded;
  // The values type is the tuple of coordinates.
  std::array < std::array<unsigned char, _Dimension>,
      1 << _Dimension* SeparateBits > _separated;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the domain and the cell length.
  Morton(const geom::BBox<_Float, _Dimension>& domain, _Float cellLength);

  //! Default constructor invalidates the data members.
  /*! You must call initialize() before using this data structure. */
  Morton();

  //! Initialize from a domain and the cell length.
  /*! For best performance with plain domains, all points should lie in the
    interior of the box. */
  void
  initialize(geom::BBox<_Float, _Dimension> domain, _Float cellLength);

  //! Set the number of levels.
  void
  setLevels(std::size_t numLevels);

private:

  void
  buildExpanded();

  void
  buildSeparated();

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the maximum valid value for any coordinate.
  typename Base::DiscreteCoordinate
  maxCoordinate() const
  {
    return (typename Base::DiscreteCoordinate(1) << Base::numLevels()) - 1;
  }

  //! Return the maximum valid code.
  Code
  maxCode() const
  {
    return (Code(1) << _Dimension * Base::numLevels()) - 1;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Conversion.
  //@{
public:

  //! Return the Morton code for the Cartesian coordinates.
  Code
  code(const typename Base::Point& coords) const
  {
    return code(Base::discretize(coords));
  }

  //! Return the Morton code for the discrete coordinates.
  Code
  code(const DiscretePoint& coords) const
  {
    Code result = expand(coords[0]);
    for (std::size_t i = 0; i != _Dimension; ++i) {
      result |= expand(coords[i]) << i;
    }
    return result;
  }

  //! Return the discrete coordinates for the Morton code.
  DiscretePoint
  coordinates(Code code) const;

private:

  // Expand the bits in an integer so that the coordinates may be merged
  // with a bitwise or to obtain the Morton code. In 3-D, 1111 is transformed
  // to 001001001001.
  Code
  expand(Code n) const;

  //@}
};



//! Convert between discrete coordinates and Morton codes.
/*!
  Specialization for 1-D.
*/
template<typename _Float, bool _Periodic>
class Morton<_Float, 1, _Periodic> :
  public DiscreteCoordinates<_Float, 1, _Periodic>
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

  typedef DiscreteCoordinates<_Float, Dimension, _Periodic> Base;

public:

  //! The unsigned integer type for holding a code.
  typedef IntegerTypes::Code Code;
  //! A discrete point with integer coordinates.
  typedef typename Base::DiscretePoint DiscretePoint;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  //@{
public:

  //! Construct from the number of levels of refinement.
  /*! Note that in 1-D the number of levels of refinement doesn't matter. */
  Morton(const geom::BBox<_Float, Dimension>& domain, _Float cellLength) :
    Base(domain, cellLength)
  {
  }

  //! Default constructor invalidates the data members.
  /*! You must call initialize() before using this data structure. */
  Morton() :
    Base()
  {
  }

  //! Initialize from a domain and the cell length.
  /*! For best performance, all points should lie in the interior of the
    box. */
  void
  initialize(geom::BBox<_Float, Dimension> domain, _Float cellLength)
  {
    Base::initialize(domain, cellLength);
  }

  //! Set the number of levels.
  void
  setLevels(std::size_t numLevels)
  {
    Base::setLevels(numLevels);
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the maximum valid value for any coordinate.
  typename Base::DiscreteCoordinate
  maxCoordinate() const
  {
    return (typename Base::DiscreteCoordinate(1) << Base::numLevels()) - 1;
  }

  //! Return the maximum valid code.
  Code
  maxCode() const
  {
    return (Code(1) << Dimension * Base::numLevels()) - 1;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Conversion.
  //@{
public:

  //! Return the Morton code for the Cartesian coordinates.
  Code
  code(const typename Base::Point& coords) const
  {
    return code(Base::discretize(coords));
  }

  //! Return the Morton code for the discrete coordinates.
  Code
  code(const DiscretePoint& coords) const
  {
    return coords[0];
  }

  //! Return the discrete coordinates for the Morton code.
  DiscretePoint
  coordinates(const Code code) const
  {
    DiscretePoint coords = {{IntegerTypes::DiscreteCoordinate(code)}};
    return coords;
  }

  //@}
};


} // namespace particle
}

#define __particle_codes_tcc__
#include "stlib/particle/codes.tcc"
#undef __particle_codes_tcc__

#endif
