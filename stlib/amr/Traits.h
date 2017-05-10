// -*- C++ -*-

/*!
  \file amr/Traits.h
  \brief The traits for an orthtree.
*/

#if !defined(__amr_Traits_h__)
#define __amr_Traits_h__

#include "stlib/amr/SpatialIndexMorton.h"

#include "stlib/container/MultiIndexTypes.h"

namespace stlib
{
namespace amr
{

//! The traits for an orthtree.
/*!
  \param _Dimension The dimension of the space.
  \param _MaximumLevel The maximum level in the tree.
  \param _SpatialIndex The spatial index data structure.  This determines
  the level and position of the node.  It also holds a key for storing the
  node in the \c std::map data structure.
  \param FloatT The real number type.
*/
template<std::size_t _Dimension, std::size_t _MaximumLevel,
         template<std::size_t, std::size_t> class _SpatialIndex =
         SpatialIndexMorton,
         typename FloatT = double>
struct Traits {
  //! The space dimension.
  BOOST_STATIC_CONSTEXPR std::size_t Dimension = _Dimension;
  //! The maximum level.
  BOOST_STATIC_CONSTEXPR std::size_t MaximumLevel = _MaximumLevel;
  //! A single index.
  typedef typename container::MultiIndexTypes<Dimension>::Index Index;
  //! A multi-index in a multi-array.
  typedef typename container::MultiIndexTypes<Dimension>::IndexList IndexList;
  //! A list if sizes.
  typedef typename container::MultiIndexTypes<Dimension>::SizeList SizeList;
  //! The spatial index.
  typedef _SpatialIndex<Dimension, MaximumLevel> SpatialIndex;
  //! The number type.
  typedef FloatT Number;
  //! A Cartesian point.
  typedef std::array<Number, Dimension> Point;
  //! The number of orthants = 2<sup>Dimension</sup>.
  BOOST_STATIC_CONSTEXPR std::size_t NumberOfOrthants =
    SpatialIndex::NumberOfOrthants;
};

} // namespace amr
}

#endif
