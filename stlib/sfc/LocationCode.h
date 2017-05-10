// -*- C++ -*-

#if !defined(__sfc_LocationCode_h__)
#define __sfc_LocationCode_h__

/*!
  \file
  \brief Class for working with location codes.
*/

#include "stlib/sfc/DiscreteCoordinates.h"

namespace stlib
{
namespace sfc
{

//! Class for working with location codes.
template<typename _Traits>
class LocationCode : public DiscreteCoordinates<_Traits>
{
  //
  // Types.
  //
private:

  //! The base class implements common functionality for codes and blocks.
  typedef DiscreteCoordinates<_Traits> Base;
  //! The class for computing location codes from index coordinates.
  typedef typename _Traits::Order Order;

public:

  //! The unsigned integer type is used for coordinates and codes.
  typedef typename _Traits::Code Code;
  //! The floating-point number type.
  typedef typename _Traits::Float Float;
  //! A Cartesian point.
  typedef typename _Traits::Point Point;
  //! A bounding box.
  typedef typename _Traits::BBox BBox;

  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;

  //
  // Member data.
  //
private:

  //! The class for computing location codes from index coordinates.
  Order _order;
  //! The maximum valid code.
  Code _maxValid;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  LocationCode();

  //! Construct from the domain and the number of levels of refinement.
  /*! The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  LocationCode(const Point& lowerCorner, const Point& lengths,
               std::size_t numLevels = Base::MaxLevels);

  //! Construct from a tight bounding box and a minimum allowed cell length.
  /*! The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  LocationCode(const BBox& tbb, Float minCellLength = 0);

  //@}
  //--------------------------------------------------------------------------
  //! \name Accessors.
  //@{
public:

  //! Return the number of bits that are used in (valid) codes.
  /*! \note The guard code uses all available bits in Code. */
  int
  numBits() const
  {
    return Dimension * Base::numLevels();
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulate location codes.
  //@{
public:

  //! Return the code for the cell that contains the point.
  Code
  code(const Point& p) const
  {
    return _order.code(Base::coordinates(p), Base::numLevels());
  }

  //! Return the next code.
  Code
  next(const Code code) const
  {
    return code + 1;
  }

  //! Return the next parent of the code.
  Code
  nextParent(const Code code) const
  {
    return ((code >> Dimension) + 1) << Dimension;
  }

  //! Extract the location bits from the code.
  /*! Simply return the code. This is for compatibility with BlockCode. */
  Code
  location(const Code code) const
  {
    return code;
  }

  //! Return true if the code is valid.
  /*! This functionality is only used for testing purposes. */
  bool
  isValid(const Code code) const
  {
    // Check that bits more significant than those used are not set.
    return code <= _maxValid;
  }

  //! Return true if the other is equal.
  bool
  operator==(const LocationCode& other) const
  {
    return Base::operator==(other) && _maxValid == other._maxValid;
  }

  //@}
};


} // namespace sfc
} // namespace stlib

#define __sfc_LocationCode_tcc__
#include "stlib/sfc/LocationCode.tcc"
#undef __sfc_LocationCode_tcc__

#endif
