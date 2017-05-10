// -*- C++ -*-

/*!
  \file geom/orq/MortonCoordinates.h
  \brief N-D Morton coordinates.
*/

#if !defined(__geom_orq_MortonCoordinates_h__)
#define __geom_orq_MortonCoordinates_h__

#include "stlib/geom/kernel/BBox.h"

#include <functional>
#include <utility>

namespace stlib
{
namespace geom
{

// CONTINUE: If I used a representation that had expanded bits (separated by
// one less than the dimension) then I could rapidly compute the code with
// bitwise or's.

// CONTINUE: Rename to DiscreteCoordinates.
//! N-D Morton coordinates.
/*!
  \param _Float The floating-point number type may be either \c float
  or \c double.
  \param _Dimension The Dimension of the space.

  \par
  This functor converts a Cartesian location to Morton coordinates.
  Thus, it simply discretizes the coordinates.
*/
template<typename _Float, std::size_t _Dimension>
class MortonCoordinates :
  public std::unary_function<std::array<_Float, _Dimension>,
  std::array<std::size_t, _Dimension> >
{
  //
  // Constants.
  //
public:

  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The number of levels of refinement.
  /*! Determine the number of levels that we can have with the interleaved
    code. Leave one of the bits so that there is a number past the largest
    valid code. This makes iteration easier. */
  BOOST_STATIC_CONSTEXPR std::size_t Levels =
    (std::numeric_limits<std::size_t>::digits - 1) / Dimension;
  //! The number of Morton blocks (in each dimension).
  BOOST_STATIC_CONSTEXPR std::size_t Extent = std::size_t(1) << Levels;

  //
  // Types.
  //
private:
  typedef std::unary_function<std::array<_Float, _Dimension>,
          std::array<std::size_t, _Dimension> >
          Base;

public:

  //! The floating-point number type.
  typedef _Float Float;
  //! The argument type is a cartesian point.
  typedef typename Base::argument_type argument_type;
  //! The result type is discretized coordinates.
  typedef typename Base::result_type result_type;
  //! A Cartesian point.
  typedef argument_type Point;

  //
  // Member data.
  //
private:

  //! The lower corner of the Cartesian domain.
  Point _lowerCorner;
  //! The length of the Cartesian domain (the same in each dimension).
  Float _length;
  //! The inverse length.
  Float _inverseLength;

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    Use the synthesized copy constructor, assignment operator, and destructor.
  */
  //@{
public:

  //! Default constructor invalidates the member data.
  MortonCoordinates() :
    _lowerCorner(ext::filled_array<Point>(std::numeric_limits<Float>::
                                          quiet_NaN())),
    _inverseLength(std::numeric_limits<Float>::quiet_NaN())
  {
  }

  //! Construct from a domain.
  /*! All points must lie inside the domain. Be sure to account for round-off
    error. That is, don't supply an exact bounding box. */
  MortonCoordinates(const BBox<Float, Dimension>& domain);

  //@}
  //--------------------------------------------------------------------------
  //! \name Functor.
  //@{
public:

  //! Calculate the Morton coordinates.
  result_type
  operator()(argument_type p) const;

  //! Return the index of the highest level whose Morton box length exceeds the specified length.
  std::size_t
  level(Float length) const;

  //@}
};

//! Convert the Morton coordinates to a code at the specified level.
template<std::size_t _Dimension>
inline
std::size_t
mortonCode(const std::array<std::size_t, _Dimension>& coords,
           const std::size_t level)
{
  const std::size_t NumLevels = MortonCoordinates<double, _Dimension>::Levels;
  // Interlace the coordinates to obtain the code.
  std::size_t code = 0;
  std::size_t mask = std::size_t(1) << (NumLevels - 1);
  for (std::size_t i = 0; i != level; ++i) {
    for (std::size_t j = 0; j != _Dimension; ++j) {
      code <<= 1;
      code |= (mask & coords[_Dimension - 1 - j]) >> (NumLevels - 1 - i);
    }
    mask >>= 1;
  }
  code <<= (NumLevels - level) * _Dimension;
  return code;
}

// CONTINUE: Hilbert index.

//! Convert the Morton coordinates to a code at the maximum level.
template<std::size_t _Dimension>
inline
std::size_t
mortonCode(const std::array<std::size_t, _Dimension>& coords)
{
  return mortonCode(coords, MortonCoordinates<double, _Dimension>::Levels);
}

//! Calculate Morton codes for the records and sort both.
/*! */
template<typename _Record, typename _Float, std::size_t _Dimension,
         typename _Location>
inline
void
indexAndSort(std::vector<_Record>* records,
             std::vector<std::size_t>* codes,
             const MortonCoordinates<_Float, _Dimension>& mortonCoordinates,
             _Location location)
{
  // Make a vector of code/index pairs.
  std::vector<std::pair<std::size_t, std::size_t> > pairs(records->size());
  // Calculate the codes and indices.
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    pairs[i].first = mortonCode(mortonCoordinates(location((*records)[i])));
    pairs[i].second = i;
  }
  // CONTINUE: An in-place MSD radix sort would be the best way to sort.
  // Taking an "or" over all values would tell you the digit at which
  // to start.
  // Technically, it sorts using the composite number formed by the pair.
  // Since the first is most significant, this is fine.
  std::sort(pairs.begin(), pairs.end());

  // Set values in the vector of codes.
  codes->resize(pairs.size());
  for (std::size_t i = 0; i != codes->size(); ++i) {
    (*codes)[i] = pairs[i].first;
  }

  // Order the records.
  {
    std::vector<_Record> tmp(records->size());
    for (std::size_t i = 0; i != tmp.size(); ++i) {
      tmp[i] = (*records)[pairs[i].second];
    }
    records->swap(tmp);
  }
}


} // namespace geom
}

#define __geom_orq_MortonCoordinates_ipp__
#include "stlib/geom/orq/MortonCoordinates.ipp"
#undef __geom_orq_MortonCoordinates_ipp__

#endif
