// -*- C++ -*-

/*!
  \file geom/spatialIndexing/SpecialEuclideanCodeUniformGrid.h
  \brief CONTINUE
*/

#if !defined(__geom_spatialIndexing_SpecialEuclideanCodeUniformGrid_h__)
#define __geom_spatialIndexing_SpecialEuclideanCodeUniformGrid_h__

#include "stlib/geom/kernel/BBox.h"

#include "stlib/numerical/equality.h"

#include <boost/math/quaternion.hpp>

namespace stlib
{
namespace geom
{

//! Encode and decode transformations in the special Euclidean group.
template<std::size_t _D, typename _Key = std::size_t>
class SpecialEuclideanCodeUniformGrid;


//! Encode and decode transformations in the 3-D special Euclidean group.
/*!
  \par
  The <a href="http://en.wikipedia.org/wiki/Euclidean_group">Euclidean
  group</a> <em>E(n)</em> is the symmetry group of <em>n</em>-dimensional
  Euclidean space. The special Euclidean group <em>SE(n)</em> is the subset
  of these transformations that preserve orientation. This set of
  transformations is commonly called the rigid body motions.

  \par
  Any transformation in <em>SE(3)</em> can be represented with a rotation,
  which is a member of the special orthogonal group <em>SO(3)</em>, followed
  by a translation. The purpose of this class is to convert between
  transformations and spatial indices. The spacial indices are packed
  into a single integer whose type is specified as a template parameter.

  \par
  Converting a translation into a spatial index is trivial. Given a bounding
  box that covers the desired domain, one simply defines a uniform grid that
  covers the bounding box. The constructor for this class takes the bounding
  box and the grid spacing as arguments. To convert a translation,
  one calculates the indices of the grid cell that contains the point at the
  head of the translation vector. Decoding returns the center of the
  cell.

  \par
  Spatial indices for rotations are trickier due to the topology of
  <em>SO(3)</em>, which is homeomorphic to a ball of radius &pi;
  in 3-D in which
  antipodal points are identified. (For a point in the ball, the vector
  from the origin is the axis of rotation and the distance from the origin
  is the angle of rotation.) Thus we consider a representation that is more
  amenable to spatial indexing. Note that the coordinates of the <em>x</em>
  and <em>y</em> axes under the rotation uniquely identify it. (We don't
  need to specify the coordinates of the <em>z</em> axis because
  \f$z = x \times y\f$. In the constructor one specifies the number of
  bits <em>n</em> to use for each of the six rotation coordinates.
  The hypercube \f$[-1..1]^6\f$ is covered by a uniform grid that has
  2<sup>n</sup> cells in each dimension. (Specifically, the centers of
  the lower and upper corner cells are at the lower and upper corners
  of the domain.) Thus a rotation is represented with a list of
  six spatial indices. This is not the most compact representation, but
  it is efficient to encode and decode.

  \par
  Instead of storing six numbers for the rotated x and y axes, one could store
  four numbers and 2 bits. That is, instead of storing the z coordinate of
  each, one could store the sign bit because the magnitude of the z coordinate
  is determined by the x and y coordinates. However, the error in the
  z coordinate may be significantly larger than in the other two because
  of compounded errors. Thus, it is preferable to store all six numbers.
*/
template<typename _Key>
class SpecialEuclideanCodeUniformGrid<3, _Key>
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t D = 3;

  //
  // Public types.
  //
public:

  //! The integer type that holds the key.
  typedef _Key Key;
  //! A Cartesian coordinate.
  typedef std::array<double, D> Point;
  //! A bounding box in Cartesian space.
  typedef geom::BBox<double, D> BBox;
  //! A quaternion.
  typedef boost::math::quaternion<double> Quaternion;

  //
  // Member data.
  //
protected:

  //! The lower corner of the Cartesian domain.
  Point _lower;
  double _spacing;
  std::array<std::size_t, D> _bitsPerTranslation;
  std::size_t _bitsPerRotation;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
   We use the synthesized copy construtor, assignment operator, and
   destructor. */
  //@{
public:

  //! Construct from the Cartesian domain, the grid spacing, and the bits per rotation coordinate.
  SpecialEuclideanCodeUniformGrid(const BBox& domain, double spacing,
                                  std::size_t bitsPerRotationCoordinate);

  //@}
  //--------------------------------------------------------------------------
  //! \name Encode/decode.
  //@{
public:

  //! Encode the transformation.
  /*!
    \param q The unit quaternion that defines the rotation.
    \param translation The translation vector.
  */
  Key
  encode(const Quaternion& q, const Point& translation) const;

  //! Decode the transformation.
  /*!
    \param key The encoded transformation.
    \param q The unit quaternion that defines the rotation.
    \param translation The translation vector.

    \note The unit quaternions q and -q define the same rotation. While
    the real component of the calculated quaternion is set to be non-negative.
    this is not sufficient to resolve the ambiguities. Thus while the
    encoded and decoded quaternions may not be close, they represent
    approximately the same rotation.
  */
  void
  decode(Key key, Quaternion* q, Point* translation) const;

  //! Encode the transformation.
  /*!
    \param x The transformed x axis.
    \param y The transformed y axis.
    \param translation The translation vector.
  */
  Key
  encode(const Point& x, const Point& y, const Point& translation) const;

  //! Decode the transformation.
  /*!
    \param key The encoded transformation.
    \param x The transformed x axis.
    \param y The transformed y axis.
    \param translation The translation vector.
  */
  void
  decode(Key key, Point* x, Point* y, Point* translation) const;

  //@}
};


} // namespace geom
}

#define __geom_spatialIndexing_SpecialEuclideanCodeUniformGrid_ipp__
#include "stlib/geom/spatialIndexing/SpecialEuclideanCodeUniformGrid.ipp"
#undef __geom_spatialIndexing_SpecialEuclideanCodeUniformGrid_ipp__

#endif
