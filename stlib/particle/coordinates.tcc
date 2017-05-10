// -*- C++ -*-

#if !defined(__particle_coordinates_tcc__)
#error This file is an implementation detail of coordinates.
#endif

namespace stlib
{
namespace particle
{

//--------------------------------------------------------------------------
// Constructors etc.

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
DiscreteCoordinates() :
  _lowerCorner(ext::filled_array<Point>(std::numeric_limits<_Float>::
                                        quiet_NaN())),
  _lengths(ext::filled_array<Point>(std::numeric_limits<_Float>::
                                    quiet_NaN())),
  _numLevels(0),
  _cellExtents(ext::filled_array<DiscretePoint>(0)),
  _cellExtentsFloat(ext::filled_array<Point>(0)),
  _scaling(ext::filled_array<Point>(std::numeric_limits<_Float>::
                                    quiet_NaN()))
{
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
setLevels(const std::size_t numLevels)
{
  assert(numLevels <
         std::size_t(std::numeric_limits<DiscreteCoordinate>::digits));
  // 2^numLevels.
  const DiscreteCoordinate oldExtent = DiscreteCoordinate(1) << _numLevels;
  const DiscreteCoordinate newExtent = DiscreteCoordinate(1) << numLevels;
  // Check for problems with floating-point precision.
  assert(_Float(newExtent - 1) != _Float(newExtent));
  for (std::size_t i = 0; i != _Dimension; ++i) {
    // Expand the domain to cover all of the cells (used or not).
    _lengths[i] = cellLengths()[i] * oldExtent;
    _cellExtents[i] = newExtent;
    _cellExtentsFloat[i] = _cellExtents[i];
    _scaling[i] = _cellExtentsFloat[i] / _lengths[i];
  }
  _numLevels = numLevels;
}

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
_initialize(geom::BBox<_Float, _Dimension> domain,
            const _Float cellLength_, std::false_type /*Periodic*/)
{
  // The maximum length, before expanding.
  const _Float maxLength = ext::max(domain.upper - domain.lower);
  // Check for problems with the domain.
  assert(maxLength > 0);
  // Check the cell length.
  assert(cellLength_ > 0);

  // Determine an appropriate number of levels.
  _numLevels = 0;
  _Float len = maxLength;
  // Determine the number of levels from the requested cell size.
  while (cellLength_ < len) {
    len *= 0.5;
    ++_numLevels;
  }
  // The number of levels may not exceed the available bits.
  assert(_numLevels <
         std::size_t(std::numeric_limits<DiscreteCoordinate>::digits));

  // 2^numLevels.
  std::fill(_cellExtents.begin(), _cellExtents.end(),
            DiscreteCoordinate(1) << _numLevels);
  _cellExtentsFloat = ext::convert_array<_Float>(_cellExtents);
  // Check for problems with floating-point precision.
  assert(_cellExtentsFloat[0] - 1 != _cellExtentsFloat[0]);

  // The lengths of the box with equal sides that contains the domain.
  _lengths = cellLength_ * _cellExtentsFloat;
  _scaling = _cellExtentsFloat / _lengths;

  // Expand the domain to get the lower corner.
  _lowerCorner = domain.lower;
  for (std::size_t i = 0; i != _lowerCorner.size(); ++i) {
    _lowerCorner[i] -=
      0.5 * (_lengths[i] - (domain.upper[i] - domain.lower[i]));
  }
}


template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
void
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
_initialize(geom::BBox<_Float, _Dimension> domain,
            const _Float cellLength_, std::true_type /*Periodic*/)
{
  // The box that defines the domain.
  _lowerCorner = domain.lower;
  _lengths = domain.upper - domain.lower;

#ifndef NDEBUG
  {
    const _Float minLength = ext::min(_lengths);
    // Check for problems with the domain.
    assert(minLength > 0);
    // Check the suggested cell length.
    assert(0 < cellLength_ && cellLength_ <= minLength);
  }
#endif

  // Use the maximum length to determine an appropriate number of levels.
  _numLevels = 0;
  // The target cell extent.
  const DiscreteCoordinate target = DiscreteCoordinate(ext::max(_lengths) /
                                    cellLength_);
  while (DiscreteCoordinate(1) << _numLevels < target) {
    ++_numLevels;
  }

  // The number of levels may not exceed the available bits.
  assert(_numLevels <
         std::size_t(std::numeric_limits<DiscreteCoordinate>::digits));

  for (std::size_t i = 0; i != _Dimension; ++i) {
    // Choose the largest cell extent such that the cell length is at least
    // as large as specified.
    _cellExtents[i] = DiscreteCoordinate(std::floor(_lengths[i] /
                                         cellLength_));
    assert(0 < _cellExtents[i] &&
           _cellExtents[i] <= DiscreteCoordinate(1) << _numLevels);
    _cellExtentsFloat[i] = _cellExtents[i];
    // Check for problems with floating-point precision.
    assert(_cellExtentsFloat[i] - 1 != _cellExtentsFloat[i]);
    _scaling[i] = _cellExtentsFloat[i] / _lengths[i];
  }
  // The cell length must be at least as large as specified.
  assert(ext::min(cellLengths()) >= cellLength_);
}


//--------------------------------------------------------------------------
// Functor.

template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
typename DiscreteCoordinates<_Float, _Dimension, _Periodic>::DiscretePoint
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
discretize(const Point& p) const
{
  // Convert to integer block coordinates.
  _Float x;
  DiscretePoint coords;
  for (std::size_t i = 0; i != coords.size(); ++i) {
    // Scale to the unit box and then to the array of cells.
    x = (p[i] - _lowerCorner[i]) * _scaling[i];
    // Truncate to lie within the array of cells. Note that this adds
    // computational cost, but the alternative is to require all points
    // be within the domain, i.e., shift the problem to the caller.
    if (x < 0) {
      x = 0;
    }
    if (x >= _cellExtentsFloat[i]) {
      x = _cellExtentsFloat[i] - 1;
    }
    // Cast to an integer.
    coords[i] = DiscreteCoordinate(x);
  }
  return coords;
}

// Return the index of the highest level whose cell lengths match or exceed
// the specified length.
template<typename _Float, std::size_t _Dimension, bool _Periodic>
inline
std::size_t
DiscreteCoordinates<_Float, _Dimension, _Periodic>::
level(const _Float length) const
{
  _Float boxLength = ext::min(_lengths);
  std::size_t level = 0;
  for (; level < _numLevels; ++level) {
    boxLength *= 0.5;
    if (boxLength < length) {
      break;
    }
  }
  return level;
}

} // namespace particle
}
