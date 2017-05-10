// -*- C++ -*-

#if !defined(__sfc_BlockCode_h__)
#define __sfc_BlockCode_h__

/*!
  \file
  \brief Class for working with location blocks (location code plus level).
*/

#include "stlib/sfc/DiscreteCoordinates.h"

#include "stlib/lorg/order.h"

namespace stlib
{
namespace sfc
{

//! Class for working with location blocks (location code plus level).
template<typename _Traits>
class BlockCode : public DiscreteCoordinates<_Traits>
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
  //! The number of bits that are used to encode the level.
  std::size_t _levelBits;
  //! The mask that is used to extract the level.
  Code _levelMask;
  //! The masks that are used to extract location information at a specified level.
  std::array<Code, Base::MaxLevels + 1> _locationMasks;
  //! The bits that are used to move to the next cell.
  std::array<Code, Base::MaxLevels + 1> _increments;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  BlockCode();

  //! Construct from the domain and the number of levels of refinement.
  /*! The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  BlockCode(const Point& lowerCorner, const Point& lengths,
            std::size_t numLevels = Base::MaxLevels);

  //! Construct from a tight bounding box and a minimum allowed cell length.
  /*! The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  BlockCode(const BBox& tbb, Float minCellLength = 0);

  //! Set the number of levels of refinement.
  void
  setNumLevels(const std::size_t numLevels);

private:

  //! Set the location masks and increment bits.
  void
  _initialize();

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
    return Dimension * Base::numLevels() + _levelBits;
  }

  //! Return the number of bits that are used to encode the level.
  std::size_t
  levelBits() const
  {
    return _levelBits;
  }

  //@}
  //--------------------------------------------------------------------------
  //! \name Manipulate block codes.
  //@{
public:

  //! Return the code for the block (at the finest level) that contains the point.
  Code
  code(const Point& p) const;

  //! Return true if the code is valid.
  /*! This functionality is only used for testing purposes. */
  bool
  isValid(Code code) const;

  //! Extract the level from the code.
  std::size_t
  level(const Code code) const
  {
    return code & _levelMask;
  }

  //! Extract the location bits from the code. (Mask out the level.)
  Code
  location(const Code code) const
  {
    return code & ~_levelMask;
  }

  //! Return the parent block. (Lower it by one level.)
  Code
  parent(Code code) const;

  //! Convert to a different level. (This may change the location.)
  Code
  atLevel(Code code, std::size_t n) const;

  //! Return the next code at the same level.
  Code
  next(const Code code) const
  {
    // Increment at the location indicated by the level.
    return code + _increments[level(code)];
  }

  //! Return the next parent.
  Code
  nextParent(const Code code) const
  {
    return next(parent(code));
  }

  //! Return the location bits for the next parent.
  Code
  locationNextParent(const Code code) const
  {
    return location(next(parent(code)));
  }

  //! Return true if the other is equal.
  bool
  operator==(const BlockCode& other) const;

  //@}
};


//! Generate codes and sort the objects.
/*!
  \relates BlockCode

  \param blockCode Data structure for manipulating codes.
  \param objects The sequence of objects will be sorted by their codes.
  \param objectCodes Output the sorted sequence of objects codes.
*/
template<typename _Traits, typename _Object>
void
sort(BlockCode<_Traits> const& blockCode,
     std::vector<_Object>* objects,
     std::vector<typename _Traits::Code>* objectCodes);


//! Coarsen to obtain cell codes for AdaptiveCells.
/*!
  \relates BlockCode

  \param blockCode Data structure for manipulating codes.
  \param objectCodes The sorted sequence of object codes.
  \param cellCodes The output sequence of cell codes. This will be terminated
  with the guard code.
  \param maxObjectsPerCell The maximum allowed objects per cell.
*/
template<typename _Traits>
void
coarsen(BlockCode<_Traits> const& blockCode,
        std::vector<typename _Traits::Code> const& objectCodes,
        std::vector<typename _Traits::Code>* cellCodes,
        std::size_t maxObjectsPerCell);


} // namespace sfc
} // namespace stlib

#define __sfc_BlockCode_tcc__
#include "stlib/sfc/BlockCode.tcc"
#undef __sfc_BlockCode_tcc__

#endif
