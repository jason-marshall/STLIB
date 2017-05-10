// -*- C++ -*-

#if !defined(__levelSet_GridUniform_h__)
#define __levelSet_GridUniform_h__

#include "stlib/container/SimpleMultiArray.h"
#include "stlib/container/SimpleMultiIndexRangeIterator.h"
#include "stlib/geom/kernel/BBox.h"

#include <iostream>

namespace stlib
{
namespace levelSet
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

//! A grid is a multi-array of patches.
/*!
  \param _T The value type.
  \param _D The dimension
*/
template<typename _T, std::size_t _D>
class GridUniform :
  public container::SimpleMultiArray<_T, _D>
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _D;

  //
  // Types.
  //
private:

  typedef container::SimpleMultiArray<_T, _D> Base;
  typedef typename Base::Index Index;

public:

  //! A Cartesian point.
  typedef std::array<_T, Dimension> Point;
  //! A bounding box.
  typedef geom::BBox<_T, Dimension> BBox;

  //
  // Member data.
  //
public:

  //! The Cartesian coordinates of the lower corner of the grid.
  const Point lowerCorner;
  //! The grid spacing.
  const _T spacing;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The copy constructor and assignment operator are disabled. We use the
    default destructor.
  */
  // @{
public:

  //! Construct from the Cartesian domain and the suggested grid spacing.
  /*!
    The grid spacing will be no greater than the suggested grid spacing and
    is the same in all dimensions. The domain will be expanded in the upper
    limits to exactly accomodate the grid.
  */
  GridUniform(const BBox& domain,
              const typename Base::value_type targetSpacing) :
    Base(calculateExtents(domain, targetSpacing),
         std::numeric_limits<_T>::quiet_NaN()),
    lowerCorner(domain.lower),
    spacing(ext::max((domain.upper - domain.lower) /
                     ext::convert_array<_T>(Base::_extents - Index(1))))
  {
    assert(spacing <= targetSpacing);
  }

  //! Construct from the extents, lower corner, and grid spacing.
  GridUniform(const typename Base::IndexList& extents,
              const Point& lowerCorner, const _T spacing) :
    Base(extents, std::numeric_limits<_T>::quiet_NaN()),
    lowerCorner(lowerCorner),
    spacing(spacing)
  {
  }

private:

  static
  typename Base::IndexList
  calculateExtents(const BBox& domain,
                   const typename Base::value_type targetSpacing)
  {
    typename Base::IndexList extents;
    for (std::size_t i = 0; i != extents.size(); ++i) {
      // Include a fudge factor for the length.
      const _T length = (domain.upper[i] - domain.lower[i]) *
                        (1 + std::numeric_limits<_T>::epsilon());
      // length = dx * (extents - 1)
      // extents = length / dx + 1
      extents[i] = std::size_t(std::ceil((length / targetSpacing + 1)));
      // Ensure that there are at least two grid points.
      assert(extents[i] >= 2);
    }
    return extents;
  }

  //! The copy constructor is not implemented.
  GridUniform(const GridUniform&);

  //! The assignment operator is not implemented.
  GridUniform&
  operator=(const GridUniform&);

  // @}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  // @{
public:

  //! Return the Cartesian position of the specified vertex.
  Point
  indexToLocation(const typename Base::IndexList& index) const
  {
    return ext::convert_array<_T>(index) * spacing;
  }

  //! Return the Cartesian domain spanned by the grid.
  BBox
  domain() const
  {
    BBox domain = {lowerCorner,
                   lowerCorner +
                   ext::convert_array<_T>(Base::extents() - Index(1)) * spacing
                  };
    return domain;
  }

  // @}
};


//! Write the grid in VTK XML format.
/*! \relates GridUniform */
template<typename _T>
void
writeVtkXml(const GridUniform<_T, 3>& grid, std::ostream& out);


//! Write the grid in VTK XML format.
/*! \relates GridUniform */
template<typename _T>
void
writeVtkXml(const GridUniform<_T, 2>& grid, std::ostream& out);


//! Print information about the grid.
/*! \relates GridUniform */
template<typename _T, std::size_t _D>
void
printInfo(const GridUniform<_T, _D>& grid, std::ostream& out);


} // namespace levelSet
}

#define __levelSet_GridUniform_ipp__
#include "stlib/levelSet/GridUniform.ipp"
#undef __levelSet_GridUniform_ipp__

#endif
