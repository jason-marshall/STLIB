// -*- C++ -*-

#if !defined(__sfc_DiscreteCoordinates_tcc__)
#error This file is an implementation detail of DiscreteCoordinates.
#endif

namespace stlib
{
namespace sfc
{

template<>
struct BlockMaxLevels<1, std::uint8_t> {
  //! 1 + 4 + ceil(log_2(5)) == 8
  BOOST_STATIC_CONSTEXPR std::size_t Result = 4;
};

template<>
struct BlockMaxLevels<1, std::uint16_t> {
  //! 1 + 11 + ceil(log_2(12)) == 16
  BOOST_STATIC_CONSTEXPR std::size_t Result = 11;
};

template<>
struct BlockMaxLevels<1, std::uint32_t> {
  //! 1 + 26 + ceil(log_2(27)) == 32
  BOOST_STATIC_CONSTEXPR std::size_t Result = 26;
};

template<>
struct BlockMaxLevels<1, std::uint64_t> {
  //! 1 + 57 + ceil(log_2(58)) == 64
  BOOST_STATIC_CONSTEXPR std::size_t Result = 57;
};


template<>
struct BlockMaxLevels<2, std::uint8_t> {
  //! 1 + 2*2 + ceil(log_2(3)) == 7
  BOOST_STATIC_CONSTEXPR std::size_t Result = 2;
};

template<>
struct BlockMaxLevels<2, std::uint16_t> {
  //! 1 + 2*6 + ceil(log_2(7)) == 16
  BOOST_STATIC_CONSTEXPR std::size_t Result = 6;
};

template<>
struct BlockMaxLevels<2, std::uint32_t> {
  //! 1 + 2*13 + ceil(log_2(14)) == 31
  BOOST_STATIC_CONSTEXPR std::size_t Result = 13;
};

template<>
struct BlockMaxLevels<2, std::uint64_t> {
  //! 1 + 2*29 + ceil(log_2(30)) == 64
  BOOST_STATIC_CONSTEXPR std::size_t Result = 29;
};


template<>
struct BlockMaxLevels<3, std::uint8_t> {
  //! 1 + 3*1 + ceil(log_2(3)) == 6
  BOOST_STATIC_CONSTEXPR std::size_t Result = 1;
};

template<>
struct BlockMaxLevels<3, std::uint16_t> {
  //! 1 + 3*4 + ceil(log_2(5)) == 16
  BOOST_STATIC_CONSTEXPR std::size_t Result = 4;
};

template<>
struct BlockMaxLevels<3, std::uint32_t> {
  //! 1 + 3*9 + ceil(log_2(10)) == 32
  BOOST_STATIC_CONSTEXPR std::size_t Result = 9;
};

template<>
struct BlockMaxLevels<3, std::uint64_t> {
  //! 1 + 3*19 + ceil(log_2(20)) == 63
  BOOST_STATIC_CONSTEXPR std::size_t Result = 19;
};


//--------------------------------------------------------------------------
// Constructors etc.


template<typename _Traits>
inline
DiscreteCoordinates<_Traits>::
DiscreteCoordinates() :
  _lowerCorner(ext::filled_array<Point>(std::numeric_limits<Float>::
                                        quiet_NaN())),
  _lengths(ext::filled_array<Point>(std::numeric_limits<Float>::
                                    quiet_NaN())),
  _numLevels(std::size_t(-1)),
  _scaling(ext::filled_array<Point>(std::numeric_limits<Float>::
                                    quiet_NaN()))
{
}


template<typename _Traits>
inline
DiscreteCoordinates<_Traits>::
DiscreteCoordinates(const Point& lowerCorner, const Point& lengths,
                    const std::size_t numLevels) :
  _lowerCorner(lowerCorner),
  _lengths(lengths),
  _numLevels(),
  _scaling()
{
  // Check for problems with the domain.
  assert(ext::max(_lengths) > 0);
  setNumLevels(numLevels);
}


template<typename _Traits>
inline
DiscreteCoordinates<_Traits>::
DiscreteCoordinates(const BBox& tbb, Float minCellLength)
{
  // Let the compiler handle the initializer list.
  // The minimum cell length may not be negative.
  assert(minCellLength >= 0);
  // Start with the maximum length of the box. Then increase it to handle
  // truncation errors.
  Float length = ext::max(tbb.upper - tbb.lower) *
    (1 + std::sqrt(std::numeric_limits<Float>::epsilon()));
  if (length == 0) {
    length = minCellLength;
  }
  if (length == 0) {
    length = 1;
  }
  const Point lengths = ext::filled_array<Point>(length);
  // To get the lower corner, we subtract half the lengths from the midpoint.
  const Point lowerCorner = Float(0.5) * (tbb.upper + tbb.lower - lengths);
  std::size_t numLevels = 0;
  for (minCellLength *= 2; minCellLength <= length && numLevels < MaxLevels;
       minCellLength *= 2, ++numLevels) {
  }
  // Call the other constructor.
  *this = DiscreteCoordinates(lowerCorner, lengths, numLevels);
}


template<typename _Traits>
inline
void
DiscreteCoordinates<_Traits>::
setNumLevels(const std::size_t numLevels)
{
  // Check that the number of levels does not exceed what the integer can hold.
  assert(numLevels <= MaxLevels);
  // Check for problems with the floating-point precision.
  Float const extents = Code(1) << _numLevels;
  if (extents - 1 == extents) {
    throw std::runtime_error("Error in stlib::sfc::setNumLevels(): The "
                             "floating-point precision is insufficient for "
                             "the number of levels.");
  }
  // Set the number of levels and scaling.
  _numLevels = numLevels;
  _scaling = ext::filled_array<Point>(Code(1) << numLevels);
  _scaling /= _lengths;
}


template<typename _Traits>
inline
bool
DiscreteCoordinates<_Traits>::
operator==(const DiscreteCoordinates& other) const
{
  return _lowerCorner == other._lowerCorner &&
         _lengths == other._lengths &&
         _numLevels == other._numLevels &&
         _scaling == other._scaling;
}


template<typename _Traits>
inline
typename DiscreteCoordinates<_Traits>::DiscretePoint
DiscreteCoordinates<_Traits>::
coordinates(const Point& p) const
{
#ifdef STLIB_DEBUG
  const Float Extent = Code(1) << _numLevels;
#endif
  DiscretePoint result;
  for (std::size_t i = 0; i != Dimension; ++i) {
    // Scale to the array of cells with floating-point arithmetic and then
    // cast to an integer.
#ifdef STLIB_DEBUG
    Float x = (p[i] - _lowerCorner[i]) * _scaling[i];
    assert(0 <= x && x < Extent);
    result[i] = x;
#else
    result[i] = (p[i] - _lowerCorner[i]) * _scaling[i];
#endif
  }
  return result;
}


} // namespace sfc
} // namespace stlib
