// -*- C++ -*-

/*!
  \file
  \brief Distance calculations for axis-oriented bounding boxes.
*/

#if !defined(__geom_BBoxDistanceSimd_h__)
#define __geom_BBoxDistanceSimd_h__

#include "stlib/geom/kernel/BBox.h"

#include "stlib/simd/allocator.h"
#include "stlib/simd/constants.h"
#include "stlib/simd/functions.h"

namespace stlib
{
namespace geom
{


template<typename _Float, std::size_t _D>
class BBoxDistance
{
  //
  // Constants and types.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;

  //! The floating-point number type.
  typedef _Float Float;
  //! The representation for a Cartesian point.
  typedef std::array<Float, Dimension> Point;
  //! An axis-aligned bounding box.
  typedef geom::BBox<Float, Dimension> BBox;
  //! An aligned vector of floating-point values.
  typedef std::vector<Float, simd::allocator<Float> > AlignedVector;
  //! We use an array of aligned vectors to store coordinates.
  typedef std::array<AlignedVector, Dimension> AlignedCoordinates;

private:

  //! The SIMD vector type.
  typedef typename simd::Vector<Float>::Type Vector;
  //! The SIMD vector size.
  BOOST_STATIC_CONSTEXPR std::size_t VectorSize = simd::Vector<Float>::Size;

  //
  // Data
  //
private:

  //! Duplicated coordinates for the lower corner.
  std::array<Vector, Dimension> _lower;
  //! Duplicated coordinates for the upper corner.
  std::array<Vector, Dimension> _upper;

  //--------------------------------------------------------------------------
  //! \name Constructors etc.
  // @{
public:

  //! Construct from a bounding box.
  BBoxDistance(const BBox& box);

  // @}
  //--------------------------------------------------------------------------
  //! \name Distance.
  // @{
public:

  //! Return lower bounds on the distance between any point in this box and the objects with the specified tight bounding box data.
  /*!
    \param lowerData Packed hybrid format for the bounding box lower corners.
    \param lowerData Packed hybrid format for the bounding box upper corners.
    \param lowerBounds The output lower bounds on the squared distance.
    This vector will be resized if necessary.

    The format for the bounding box data in 3-D is:
    lowerX, lowerY, lowerZ, upperX, upperY, upperZ,
    where each block of coordinates is of the SIMD vector length.

    The lower and upper corners should be padded with NaN's. For these 
    locations, the recorded lower bound will be 0. */
  void
  lowerBound2(AlignedVector const& lowerData,
              AlignedVector const& upperData,
              AlignedVector* lowerBounds) const;

  //! Calculate upper bounds on the distance between any point in this box and the objects with the specified extreme points data.
  /*!
    \param extremePointData Each element is in packed hybrid format for the
    extreme point locations.
    \param upperBounds The output upper bounds on the squared distance.
    This vector will be resized if necessary.

    The point data should be padded with NaN's. For these locations, the
    recorded upper bound will be infinity. */
  template<std::size_t _PtsPerBox>
  void
  upperBound2(std::array<AlignedVector, _PtsPerBox> const& extremePointData,
              AlignedVector* upperBounds) const;

  //! Return an upper bound on the squared distance between any point in this box and the specified points.
  /*!
    \param pointData The vector is in packed hybrid format. It should be padded
    with NaN's. The padded locations will not affect the upper bound. */
  Float
  upperBound2(AlignedVector const& pointData)
    const;

  //! For each triangle, record if it is relevant.
  /*! It is relevant if the lower bound on the distance is not greater
    than the upper bound.

    \param lower Lower corners for the object bounding boxes.
    \param upper Upper corners for the object bounding boxes.
    \param points Points on the objects. There may be any positive number of
    points per object.
    \param relevantObjects Record whether each object is relevant (with a 0 
    or 1). */
  void
  lowerLessEqualUpper2(AlignedCoordinates const& lower,
                       AlignedCoordinates const& upper,
                       AlignedCoordinates const& points,
                       std::vector<unsigned char>* relevantObjects) const;

  //! For each triangle, record if it is relevant.
  /*! It is relevant if the lower bound on the distance is not greater
    than the upper bound. */
  void
  lowerLessEqualUpper2(_Float upperBound_,
                       AlignedCoordinates const& lower,
                       AlignedCoordinates const& upper,
                       std::vector<unsigned char>* relevantObjects) const;

private:

  //! The maximum distance from a point in this bounding box to each of the supplied points.
  Vector
  _maxDist2(const Float* soaData) const;

  Vector
  _maxDist2(AlignedCoordinates const& points,
            std::size_t const blockIndex) const;

  Vector
  _lowerBound2(AlignedCoordinates const& lower,
               AlignedCoordinates const& upper,
               std::size_t const blockIndex) const;

  Float
  _upperBound2(AlignedCoordinates const& points) const;

  // @}
};


} // namespace geom
}

#define __geom_BBoxDistanceSimd_ipp__
#include "stlib/geom/kernel/BBoxDistanceSimd.ipp"
#undef __geom_BBoxDistanceSimd_ipp__

#endif
