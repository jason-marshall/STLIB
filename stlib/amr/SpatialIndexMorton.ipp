// -*- C++ -*-

#if !defined(__amr_SpatialIndexMorton_ipp__)
#error This file is an implementation detail of the class SpatialIndexMorton.
#endif

namespace stlib
{
namespace amr
{

//--------------------------------------------------------------------------
// Constructors etc.

// Default constructor.  Zero level and coordinates.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
SpatialIndexMorton<_Dimension, _MaximumLevel>::
SpatialIndexMorton() :
  _code(0),
  _coordinates(ext::filled_array<CoordinateList>(0)),
  _level(0)
{
  static_assert(Dimension > 0, "Dimension must be positive.");
}

// Construct from the level and the coordinates.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
SpatialIndexMorton<_Dimension, _MaximumLevel>::
SpatialIndexMorton(const Level level, const CoordinateList& coordinates) :
  _code(),
  _coordinates(coordinates),
  _level(level)
{
  static_assert(Dimension > 0, "Dimension must be positive.");
  updateCode();
}

//--------------------------------------------------------------------------
// Manipulators.

// Set the level and coordinates.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
set(const Level level, const CoordinateList& coordinates)
{
  _level = level;
  _coordinates = coordinates;
  updateCode();
}

// Transform to the invalid index.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
invalidate()
{
  // Set the invalid bit.
  _code = 1 << Dimension * MaximumLevel;
  // Setting the coordinates and level is not really necessary, but this
  // way all invalid indices are identical.
  std::fill(_coordinates.begin(), _coordinates.end(), 0);
  _level = 0;
}

// Transform to the parent node index.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToParent()
{
#ifdef STLIB_DEBUG
  assert(_level != 0);
#endif
  --_level;
  _coordinates >>= 1;
  updateCode();
}

// Transform to the parent node index.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToAncestor(const std::size_t steps)
{
#ifdef STLIB_DEBUG
  assert(_level >= steps);
#endif
  _level -= steps;
  _coordinates >>= steps;
  updateCode();
}

// Transform to the specified child node index.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToChild(std::size_t n)
{
#ifdef STLIB_DEBUG
  assert(n < NumberOfOrthants);
#endif
  ++_level;
  _coordinates <<= 1;
  for (std::size_t i = 0; i != Dimension; ++i) {
    _coordinates[i] += n % 2;
    n >>= 1;
  }
  updateCode();
}

// Transform to the specified child node index the specified number of times.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToChild(const std::size_t n, const std::size_t steps)
{
#ifdef STLIB_DEBUG
  assert(n < NumberOfOrthants);
#endif
  _level += steps;
  for (std::size_t s = 0; s != steps; ++s) {
    _coordinates <<= 1;
    std::size_t m = n;
    for (std::size_t i = 0; i != Dimension; ++i) {
      _coordinates[i] += m % 2;
      m >>= 1;
    }
  }
  updateCode();
}

// Transform to the specified neighbor.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToNeighbor(const std::size_t n)
{
#ifdef STLIB_DEBUG
  assert(n < 2 * Dimension);
  assert(hasNeighbor(*this, n));
#endif
  // The coordinate is n / 2.
  // The direction in that coordinate is n % 2.
  // Change coordinate by +-1.
  _coordinates[n / 2] += 2 * (n % 2) - 1;
  updateCode();
}

// Transform to the next node at this level.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
transformToNext()
{
  // Shift to get rid of the zeros in the least significant bits.
  _code >>= (MaximumLevel - _level) * Dimension;
  // Move to the next node.
  _code += 1;
  unlaceBits(_code, _level, &_coordinates);
  // Shift to remove a possible overflow that could occur if this was
  // the last node.
  _code <<= (std::numeric_limits<Code>::digits - _level * Dimension);
  // Shift to restore the zero padding.
  _code >>= (std::numeric_limits<Code>::digits - MaximumLevel * Dimension);
}

// Update the code from the coordinates and the level.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
updateCode()
{
  // Make a copy of the coordinates.  We will right-shift them to strip
  // off the binary digits.
  CoordinateList c(_coordinates);
  // First left-shift to get the real coordinates.
  c <<= (MaximumLevel - _level);
  // Interlace the coordinates.
  _code = interlaceBits<Code>(c, MaximumLevel);
}

// Update the coordinates from the code and the level.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
updateCoordinates()
{
  // Shift to get rid of the zeros in the least significant bits.
  _code >>= (MaximumLevel - _level) * Dimension;
  unlaceBits(_code, MaximumLevel, &_coordinates);
  // Shift to restore the zero padding.
  _code <<= (MaximumLevel - _level) * Dimension;
}

//--------------------------------------------------------------------------
// File I/O.

// Print the level and the coordinates.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
print(std::ostream& out) const
{
  out << double(_level) << " ";
  for (std::size_t i = 0; i != Dimension; ++i) {
    numerical::printBits(out, _coordinates[i]);
    out << " ";
  }
  numerical::printBits(out, _code);
}

//--------------------------------------------------------------------------
// Message stream I/O.

// Write to the message stream.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
write(MessageOutputStream& out) const
{
  out << (unsigned char)(_code >> Dimension * MaximumLevel);
  out << _level;
  out.write(&_coordinates[0], Dimension);
}

// Write to the message stream.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
write(MessageOutputStreamChecked& out) const
{
  out << (unsigned char)(_code >> Dimension * MaximumLevel);
  out << _level;
  out.write(&_coordinates[0], Dimension);
}

// Read from the message stream.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
SpatialIndexMorton<_Dimension, _MaximumLevel>::
read(MessageInputStream& in)
{
  unsigned char invalid;
  in >> invalid;
  in >> _level;
  in.read(&_coordinates[0], Dimension);
  if (invalid) {
    invalidate();
  }
  else {
    updateCode();
  }
}

//---------------------------------------------------------------------------
// Topology and geometry.

// Return true if the key is a local lower corner.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
isLowerCorner(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x)
{
  std::size_t sum = 0;
  for (std::size_t d = 0; d != _Dimension; ++d) {
    sum += x.getCoordinates()[d] % 2;
  }
  return sum == 0;
}

// Return true if the key has a parent.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasParent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x)
{
  return x.getLevel() != 0;
}

// Return true if the key has children parent.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasChildren(const SpatialIndexMorton<_Dimension, _MaximumLevel>& x)
{
  return x.getLevel() != _MaximumLevel;
}

// Compute the distance between the two nodes.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
void
computeSeparations(const SpatialIndexMorton<_Dimension, _MaximumLevel>& index1,
                   const SpatialIndexMorton<_Dimension, _MaximumLevel>& index2,
                   std::array<int, _Dimension>* separations)
{
  typedef SpatialIndexMorton<_Dimension, _MaximumLevel> SpatialIndexMorton;
  typedef typename SpatialIndexMorton::CoordinateList CoordinateList;

  CoordinateList location1, location2;
  computeLocation(index1, &location1);
  computeLocation(index2, &location2);
  // CONTINUE: Does int have enough digits?
  const int length1 = computeLength(index1);
  const int length2 = computeLength(index2);
  for (std::size_t d = 0; d != _Dimension; ++d) {
    (*separations)[d] = (location1[d] < location2[d] ?
                         (location2[d] - location1[d]) - length1 :
                         (location1[d] - location2[d]) - length2);
  }
}

// Return true if the nodes are adjacent.
// Two nodes in N-D are adjacent if they share a (N-1)-D boundary.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
areAdjacent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& a,
            const SpatialIndexMorton<_Dimension, _MaximumLevel>& b)
{
  std::array<int, _Dimension> separations;
  computeSeparations(a, b, &separations);
  std::size_t countZero = 0, countNegative = 0;
  for (std::size_t i = 0; i != _Dimension; ++i) {
    countZero += separations[i] == 0;
    countNegative += separations[i] < 0;
  }
  return countZero == 1 && countNegative == _Dimension - 1;
}

// Return true if the second node is a descendent of the first.
// Note: isDescendent(x, x) is true.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
isDescendent(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
             SpatialIndexMorton<_Dimension, _MaximumLevel> descendent)
{
  if (descendent.getLevel() < node.getLevel()) {
    return false;
  }
  descendent.transformToAncestor(descendent.getLevel() - node.getLevel());
  return node == descendent;
}

// Return true if the node has a neighbor in the specified direction.
// direction / 2 gives the coordinate.  direction % 2 gives the direction in
// that coordinate (negative or positive).
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasNeighbor(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
            const std::size_t direction)
{
  typedef typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Coordinate
  Coordinate;

  // Negative direction.
  if (direction % 2 == 0) {
    // The lower side of the node.
    return node.getCoordinates()[direction / 2] != 0;
  }
  // Positive direction.
  else {
    // The upper side of the node.
    Coordinate upper = computeLocation(node, direction / 2)
                       + computeLength(node);
    // The coordinate may have more than _MaximumLevel digits.  Keep only
    // _MaximumLevel digits.
    upper <<= std::numeric_limits<Coordinate>::digits - _MaximumLevel;
    return upper != 0;
  }
}

// Return true if there is a next node on same level.
// The maximum code is
// (2^{D l} - 1) << ((M - l)D)
// where \e D is the dimension, \e l is the level, and \e M is the maximum
// level. This function just checks if the code is equal to the maximum value.
template<std::size_t _Dimension, std::size_t _MaximumLevel>
inline
bool
hasNext(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node)
{
  typedef typename SpatialIndexMorton<_Dimension, _MaximumLevel>::Code Code;
  Code maxCode = 1;
  maxCode <<= _Dimension * node.getLevel();
  maxCode -= 1;
  maxCode <<= (_MaximumLevel - node.getLevel()) * _Dimension;
  return node.getCode() != maxCode;
}

// Get the adjacent neighbors of the node.
template<std::size_t _Dimension, std::size_t _MaximumLevel,
         typename _OutputIterator>
inline
void
getAdjacentNeighbors(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
                     _OutputIterator adjacent)
{
  SpatialIndexMorton<_Dimension, _MaximumLevel> neighbor;
  // For each signed direction.
  for (std::size_t direction = 0; direction != 2 * _Dimension; ++direction) {
    // If there is an adjacent neighbor.
    if (hasNeighbor(node, direction)) {
      // Get the adjacent neighbor.
      neighbor = node;
      neighbor.transformToNeighbor(direction);
      // Add it to the list of adjacent neighbors.
      *adjacent++ = neighbor;
    }
  }
}

// Get the adjacent neighbors at the next highest level.
template<std::size_t _Dimension, std::size_t _MaximumLevel,
         typename _OutputIterator>
inline
void
getAdjacentNeighborsHigherLevel
(const SpatialIndexMorton<_Dimension, _MaximumLevel>& node,
 _OutputIterator adjacent)
{
  typedef SpatialIndexMorton<_Dimension, _MaximumLevel> SpatialIndexMorton;

  SpatialIndexMorton neighbor;
  std::size_t s, d, directionBit;

  // For each signed direction.
  for (std::size_t direction = 0; direction != 2 * _Dimension; ++direction) {
    // If there is an adjacent neighbor.
    if (hasNeighbor(node, direction)) {
      // Get the adjacent neighbor.
      neighbor = node;
      neighbor.transformToNeighbor(direction);
      // The sign of the direction;
      s = direction % 2;
      // The unsigned direction.
      d = direction / 2;
      // For each child of the neighbor.
      for (std::size_t child = 0;
           child != SpatialIndexMorton::NumberOfOrthants; ++child) {
        directionBit = (child >> d) % 2;
        // If the d_th direction bit is opposite, this is an adjacent node.
        if (directionBit != s) {
          // Get the adjacent child on the higher level.
          neighbor.transformToChild(child);
          // Add it to the list of adjacent neighbors.
          *adjacent++ = neighbor;
          // Return to the adjacent neighbor on the same level.
          neighbor.transformToParent();
        }
      }
    }
  }
}

} // namespace amr
}
