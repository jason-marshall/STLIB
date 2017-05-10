// -*- C++ -*-

#if !defined(__sfc_LocationCode_tcc__)
#error This file is an implementation detail of LocationCode.
#endif

namespace stlib
{
namespace sfc
{


//--------------------------------------------------------------------------
// Constructors etc.


template<typename _Traits>
inline
LocationCode<_Traits>::
LocationCode() :
  Base(),
  _order(),
  _maxValid(0)
{
}


template<typename _Traits>
inline
LocationCode<_Traits>::
LocationCode(const Point& lowerCorner, const Point& lengths,
             const std::size_t numLevels) :
  Base(lowerCorner, lengths, numLevels),
  _order(),
  _maxValid((Code(1) << (Dimension* numLevels)) - 1)
{
}


template<typename _Traits>
inline
LocationCode<_Traits>::
LocationCode(const BBox& tbb, const Float minCellLength) :
  // Determine the lower corner, lengths, and number of levels with the base
  // constructor.
  Base(tbb, minCellLength),
  _order(),
  _maxValid(0)
{
  // Call the other constructor.
  *this = LocationCode(Base::_lowerCorner, Base::_lengths, Base::_numLevels);
}


} // namespace sfc
}
