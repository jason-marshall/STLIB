// -*- C++ -*-

#if !defined(__sfc_DiscreteCoordinates_h__)
#define __sfc_DiscreteCoordinates_h__

/**
  \file
  \brief Base class for location codes and block codes (location plus level).
*/

#include "stlib/sfc/Traits.h"
#include "stlib/numerical/integer/bits.h"

namespace stlib
{
namespace sfc
{

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;

/// Result holds the maximum levels of refinement for a block code.
/**
  We need one bit so that we can represent a code that is greater than all
  valid codes. We need _Dimension * Result bits for the bits of
  the location code. We need ceil(log_2(Result + 1)) bits to record the level.
  Since there is no simple formula for the maximum level of refinement,
  we specify the value for each dimension and each integer type that we
  intend to use with template specialization.
*/
template<std::size_t _Dimension, typename _Code>
struct BlockMaxLevels;


/// Base class for location codes and block codes (location plus level).
template<typename _Traits>
class DiscreteCoordinates
{
  //
  // Types.
  //
public:

  /// The unsigned integer type is used for coordinates and codes.
  typedef typename _Traits::Code Code;
  /// The floating-point number type.
  typedef typename _Traits::Float Float;
  /// A Cartesian point.
  typedef typename _Traits::Point Point;
  /// A bounding box.
  typedef typename _Traits::BBox BBox;

  //
  // Constants.
  //
public:

  /// The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Traits::Dimension;
  /// The number of children for a parent.
  BOOST_STATIC_CONSTEXPR std::size_t NumChildren = std::size_t(1) << Dimension;
  /// The maximum possible number of levels given the dimension and integer type.
  BOOST_STATIC_CONSTEXPR std::size_t MaxLevels =
    BlockMaxLevels<Dimension, Code>::Result;

  //
  // Types.
  //
protected:

  /// A point with discrete coordinates.
  typedef std::array<Code, Dimension> DiscretePoint;

  //
  // Member data.
  //
protected:

  /// The lower corner of the Cartesian domain.
  Point _lowerCorner;
  /// The lengths of the Cartesian domain.
  Point _lengths;
  /// The number of levels of refinement.
  std::size_t _numLevels;
  /// The scaling factors in transforming to cells (at the finest level).
  /** These are the extents divided by the lengths. */
  Point _scaling;

  //--------------------------------------------------------------------------
  /** \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  /// The default constructor results in invalid member data.
  DiscreteCoordinates();

  /// Construct from the domain and the number of levels of refinement.
  /** The lengths must be positive, but need not be the same. However, for
   most applications, the lengths will be the same as this maximizes the
   compactness of the cells. */
  DiscreteCoordinates(Point const& lowerCorner, Point const& lengths,
                      std::size_t numLevels);

  /// Construct from a tight bounding box and a minimum allowed cell length.
  /** The domain will be expanded so that truncation errors will not result
    in points that lie outside of the domain of the trie, and then it
    will be extended so that the lengths in each dimension are the same.
    The cell length at the maximum level of refinement will not be less
    than the specifed minimum length. */
  DiscreteCoordinates(BBox const& tbb, Float minCellLength);

  /// Set the number of levels of refinement.
  void
  setNumLevels(std::size_t numLevels);

  //@}
  //--------------------------------------------------------------------------
  /// \name Accessors.
  //@{
public:

  /// Return the lower corner of the Cartesian domain.
  Point const&
  lowerCorner() const
  {
    return _lowerCorner;
  }

  /// The lengths of the Cartesian domain.
  Point const&
  lengths() const
  {
    return _lengths;
  }

  /// Return the number of levels of refinement.
  std::size_t
  numLevels() const
  {
    return _numLevels;
  }

  /// Return true if the other is equal.
  bool
  operator==(DiscreteCoordinates const& other) const;

protected:

  /// Calculate index coordinates for the Cartesian point.
  DiscretePoint
  coordinates(Point const& p) const;

  //@}
};


/// Write the lower corner, lengths, and number of levels.
template<typename _Traits>
inline
std::ostream&
operator<<(std::ostream& out, DiscreteCoordinates<_Traits> const& x)
{
  return out << "lower corner = " << x.lowerCorner() << ", "
         << "lengths = " << x.lengths() << ", "
         << "num levels = " << x.numLevels() << '\n';
}


} // namespace sfc
} // namespace stlib

#define __sfc_DiscreteCoordinates_tcc__
#include "stlib/sfc/DiscreteCoordinates.tcc"
#undef __sfc_DiscreteCoordinates_tcc__

#endif
