// -*- C++ -*-

/*!
  \file geom/orq/SpatialIndexMortonUniform.h
  \brief N-D spatial index that calculates an interlaced Morton code.
*/

#if !defined(__geom_orq_SpatialIndexMortonUniform_h__)
#define __geom_orq_SpatialIndexMortonUniform_h__

#include "stlib/geom/kernel/BBox.h"

#include <functional>

namespace stlib
{
namespace geom
{

//! N-D spatial index that calculates an interlaced Morton code.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.

  \par Overview.
  This functor converts a Cartesian location to a spatial index. The
  <a href="http://en.wikipedia.org/wiki/Z-order_curve">Morton order</a>
  defines the space-filling curve.

  \par Setup.
  First we define the domain to be digitized. One specifies a
  bounding box in the constructor and this domain is first covered with
  an equilateral block. The block is then recursively divided until
  the length of a block is no greater than the specified maximum length.
  One then has 2<sup>NS</sup> blocks where \e N is the space dimension
  and \e S is the number of subdivisions.

  \par Spatial index.
  A Cartesian location is converted to a spatial index by first
  converting the Cartesian coordinates to block indices. If the
  Cartesian point lies outside of the domain of the blocks, it
  will be assigned to the nearest block. The block indices are
  then converted to the spatial index by interleaving the bits.
  This results in a single index in the range [0...2<sup>NS</sup>-1].

  \par Moving points.
  If you use the spatial index for set of moving points, it is
  probably best to first put a bounding box around the points and
  then expand the box by how far you expect them to move.
  If the domain changes significantly, it is best to construct a
  new spatial index functor.
*/
template<typename _Float, std::size_t _Dimension>
class SpatialIndexMortonUniform :
  public std::unary_function<std::array<_Float, _Dimension>,
  std::size_t>
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;

  //
  // Types.
  //
private:
  typedef std::unary_function<std::array<_Float, _Dimension>, std::size_t>
  Base;

public:

  //! The floating-point number type.
  typedef _Float Float;
  //! The argument type is a cartesian point.
  typedef typename Base::argument_type argument_type;
  //! The result type a \c std::size_t.
  typedef typename Base::result_type result_type;
  //! A Cartesian point.
  typedef std::array<Float, Dimension> Point;

  //
  // Member data.
  //
private:

  //! The number of levels.
  std::size_t _levels;
  //! The number of Morton blocks (in each dimension).
  std::size_t _extent;
  //! The lower corner of the Cartesian domain.
  Point _lowerCorner;
  //! The inverse length of the Cartesian domain (the same in each dimension).
  Float _inverseLength;

private:

  //! Default constructor is not implemented.
  SpatialIndexMortonUniform();

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Construct from a domain and the max length of a Morton block.
  SpatialIndexMortonUniform(const BBox<Float, Dimension>& domain,
                            Float maximumLength);

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Calculate the interleaved code.
  result_type
  operator()(argument_type p) const;

  //@}
};

} // namespace geom
}

#define __geom_orq_SpatialIndexMortonUniform_ipp__
#include "stlib/geom/orq/SpatialIndexMortonUniform.ipp"
#undef __geom_orq_SpatialIndexMortonUniform_ipp__

#endif
