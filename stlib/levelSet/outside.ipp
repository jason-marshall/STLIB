/* -*- C++ -*- */

#if !defined(__levelSet_outside_ipp__)
#error This file is an implementation detail of outside.
#endif

namespace stlib
{
namespace levelSet
{

// The leftmost bit in a byte is the most significant.
// With little-endian ordering, the first byte in a multi-byte integer
// is the least significant. (Little end first.)
// The bit ordering for a 32-bit unsigned integer is the following.
// 07 06 05 04 03 02 01 00 15 14 13 12 10 09 08 ...


// Precondition: All of the sign bits must be initialized to 1.
template<typename _T>
inline
void
setSign(const Grid<_T, 3, 8>& grid,
        const std::array<std::size_t, 3>& byteExtents,
        std::vector<std::size_t>* sign)
{
#ifdef STLIB_DEBUG
  for (std::size_t i = 0; i != sign->size(); ++i) {
    assert((*sign)[i] == std::numeric_limits<std::size_t>::max());
  }
#endif
  const std::size_t D = 3;
  const std::size_t N = 8;

  typedef Grid<_T, D, N> Grid;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef typename Grid::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Strides for indexing the multi-array of bytes.
  // Below we will use the fact that byteStrides[0] is 1 to avoid a
  // multiplication.
  const IndexList byteStrides = {{
      1, byteExtents[0],
      byteExtents[0]* byteExtents[1]
    }
  };

  // Note: This does not violate strict aliasing rules. It's is OK to
  // use a character type to alias any other type.
  unsigned char* signBytes = reinterpret_cast<unsigned char*>(&(*sign)[0]);
  // Loop over the patches.
  const Iterator pEnd = Iterator::end(grid.extents());
  for (Iterator p = Iterator::begin(grid.extents()); p != pEnd; ++p) {
    // Grid index.
    const IndexList& gi = *p;
    const VertexPatch& patch = grid(gi);
    if (! patch.isRefined()) {
      // Unrefined patches that have all negative distances.
      if (patch.fillValue < 0) {
        // Loop over the second and third dimensions.
        for (std::size_t j = 0; j != N; ++j) {
          for (std::size_t k = 0; k != N; ++k) {
            // Set 8 bits at a time.
            signBytes[gi[0] + (gi[1] * N + j) * byteStrides[1] +
                      (gi[2] * N + k) * byteStrides[2]] = 0;
          }
        }
      }
    }
    // Refined patch.
    else {
      // Loop over the second and third dimensions.
      for (std::size_t j = 0; j != N; ++j) {
        for (std::size_t k = 0; k != N; ++k) {
          // Pack the sign information into an unsigned char.
          unsigned char s = 0;
          for (std::size_t i = N; i-- != 0;) {
            s <<= 1;
            s |= patch(i, j, k) > 0;
          }
          // Set 8 bits at a time.
          signBytes[gi[0] + (gi[1] * N + j) * byteStrides[1] +
                    (gi[2] * N + k) * byteStrides[2]] = s;
        }
      }
    }
  }
}


// Set the boundaries in the y and z directions to be outside. (Sweeping
// will take care of the x boundaries.) On these boundaries we set the
// outside bits to the same values as the sign bits. Thus, the object may
// intersect the boundary, but internal cavities are not permitted to
// touch the boundary.
inline
void
setBoundaryCondition(const std::array<std::size_t, 3>& extents,
                     const std::vector<std::size_t>& sign,
                     std::vector<std::size_t>* outside)
{
#ifdef STLIB_DEBUG
  assert(extents[1] >= 2 && extents[2] >= 2);
#endif

  // -z face.
  memcpy(&(*outside)[0], &sign[0],
         extents[0] * extents[1] * sizeof(std::size_t));
  // +z face.
  std::size_t offset = extents[0] * extents[1] * (extents[2] - 1);
  memcpy(&(*outside)[offset], &sign[offset],
         extents[0] * extents[1] * sizeof(std::size_t));

  const std::size_t size = extents[0] * sizeof(std::size_t);
  for (std::size_t i = 1; i != extents[2] - 1; ++i) {
    // -y row.
    offset = i * extents[0] * extents[1];
    memcpy(&(*outside)[offset], &sign[offset], size);
    // +y row.
    offset += extents[0] * (extents[1] - 1);
    memcpy(&(*outside)[offset], &sign[offset], size);
  }
}


// Sweep in the x direction within a single row.
inline
void
sweepXLocal(const std::size_t length, const std::size_t* sign,
            std::size_t* outside)
{
  const int MaxShift = std::numeric_limits<std::size_t>::digits - 1;
  const std::size_t HiBit = std::size_t(1) << MaxShift;
  // CONTINUE: Determine a good choice experimentally.
  const std::size_t NumIterations = 8;

#ifdef STLIB_DEBUG
  assert(length != 0);
#endif

  // Check the simple case that the row is composed of a single element.
  if (length == 1) {
    for (std::size_t n = 0; n != NumIterations; ++n) {
      outside[0] |= ((outside[0] >> 1) | (outside[0] << 1)) & sign[0];
    }
    return;
  }

  std::size_t shifted;
  for (std::size_t n = 0; n != NumIterations; ++n) {
    // First element.
    shifted = (outside[0] >> 1) | (outside[0] << 1) |
              (1 & outside[1]) << MaxShift;
    outside[0] |= shifted & sign[0];
    // Internal elements.
    for (std::size_t i = 1; i != length - 1; ++i) {
      // Shift the bits left and right.
      shifted = (outside[i] >> 1) | (outside[i] << 1) |
                (HiBit & outside[i - 1]) >> MaxShift |
                (1 & outside[i + 1]) << MaxShift;
      outside[i] |= shifted & sign[i];
    }
    // Last element.
    shifted = (outside[length - 1] >> 1) | (outside[length - 1] << 1) |
              (HiBit & outside[length - 2]) >> MaxShift;
    outside[length - 1] |= shifted & sign[length - 1];
  }
}


// Sweep in the x direction for each of the rows.
inline
void
sweepXLocal(const std::array<std::size_t, 3> extents,
            const std::vector<std::size_t>& sign,
            std::vector<std::size_t>* outside)
{
  // Loop over the rows.
  const std::size_t end = ext::product(extents);
  for (std::size_t i = 0; i != end; i += extents[0]) {
    sweepXLocal(extents[0], &sign[i], &(*outside)[i]);
  }
}


// length - The number of std::size_t integers in a row.
// source - The outside bits in the source row.
// sign - The sign bits in the target row.
// target - The outside bits in the target row.
inline
void
sweepAdjacentRow(const std::size_t length,
                 const std::size_t* source,
                 const std::size_t* sign, std::size_t* target)
{
  const int MaxShift = std::numeric_limits<std::size_t>::digits - 1;
  const std::size_t HiBit = std::size_t(1) << MaxShift;

  std::size_t s, combined;

  // The first.
  s = source[0];
  // Combine the source bits from center, left and right.
  combined = s | (s >> 1) | (s << 1);
  if (length > 1) {
    combined |= (1 & source[1]) << MaxShift;
  }
  // Propagate the outside information.
  target[0] |= combined & sign[0];

  // Loop over all except the first and last.
  for (std::size_t i = 1; i < length - 1; ++i) {
    s = source[i];
    // Combine the source bits from center, left and right.
    combined = s | (s >> 1) | (s << 1) |
               (HiBit & source[i - 1]) >> MaxShift |
               (1 & source[i + 1]) << MaxShift;
    // Propagate the outside information.
    target[i] |= combined & sign[i];
  }

  // The last.
  s = source[length - 1];
  // Combine the source bits from center, left and right.
  combined = s | (s >> 1) | (s << 1);
  if (length > 1) {
    combined |= (HiBit & source[length - 2]) >> MaxShift;
  }
  // Propagate the outside information.
  target[length - 1] |= combined & sign[length - 1];
}


inline
void
sweepAdjacentRow(const std::array<std::size_t, 3> extents,
                 const std::vector<std::size_t>& sign,
                 std::vector<std::size_t>* outside,
                 const std::size_t j, const std::size_t k,
                 const int dj, const int dk)
{
#ifdef STLIB_DEBUG
  // Check for a valid source row.
  assert(j < extents[1] && k < extents[2]);
  // Valid offset to the target row.
  assert(-1 <= dj && dj <= 1 &&
         -1 <= dk && dk <= 1 &&
         !(dj == 0 && dk == 0));
#endif
  // The index offsets for the source and target rows.
  const std::size_t source = j * extents[0] + k * extents[0] * extents[1];
  const std::size_t target = (j + dj) * extents[0] +
                             (k + dk) * extents[0] * extents[1];
  // Call the function above with pointers to the beginnings of the rows.
  sweepAdjacentRow(extents[0], &(*outside)[source], &sign[target],
                   &(*outside)[target]);
}


inline
void
sweepY(const std::array<std::size_t, 3>& extents,
       const std::vector<std::size_t>& sign,
       std::vector<std::size_t>* outside)
{
  // Loop over the z slices.
  for (std::size_t k = 0; k != extents[2]; ++k) {
    // Positive direction.
    for (std::size_t j = 0; j != extents[1] - 1; ++j) {
      sweepAdjacentRow(extents, sign, outside, j, k, 1, 0);
    }
    // Negative direction.
    for (std::size_t j = extents[1] - 1; j != 0; --j) {
      sweepAdjacentRow(extents, sign, outside, j, k, -1, 0);
    }
  }
}


inline
void
sweepZ(const std::array<std::size_t, 3>& extents,
       const std::vector<std::size_t>& sign,
       std::vector<std::size_t>* outside)
{
  // Loop over the z slices in the positive direction.
  for (std::size_t k = 0; k != extents[2] - 1; ++k) {
    // First row.
    sweepAdjacentRow(extents, sign, outside, 0, k, 0, 1);
    sweepAdjacentRow(extents, sign, outside, 0, k, 1, 1);
    // Interior rows.
    for (std::size_t j = 1; j != extents[1] - 1; ++j) {
      // CONTINUE: Write a function that updates three rows at a time.
      sweepAdjacentRow(extents, sign, outside, j, k, -1, 1);
      sweepAdjacentRow(extents, sign, outside, j, k, 0, 1);
      sweepAdjacentRow(extents, sign, outside, j, k, 1, 1);
    }
    // Last row.
    sweepAdjacentRow(extents, sign, outside, extents[1] - 1, k, -1, 1);
    sweepAdjacentRow(extents, sign, outside, extents[1] - 1, k, 0, 1);
  }

  // Loop over the z slices in the negative direction.
  for (std::size_t k = extents[2] - 1; k != 0; --k) {
    // First row.
    sweepAdjacentRow(extents, sign, outside, 0, k, 0, -1);
    sweepAdjacentRow(extents, sign, outside, 0, k, 1, -1);
    // Interior rows.
    for (std::size_t j = 1; j != extents[1] - 1; ++j) {
      sweepAdjacentRow(extents, sign, outside, j, k, -1, -1);
      sweepAdjacentRow(extents, sign, outside, j, k, 0, -1);
      sweepAdjacentRow(extents, sign, outside, j, k, 1, -1);
    }
    // Last row.
    sweepAdjacentRow(extents, sign, outside, extents[1] - 1, k, -1, -1);
    sweepAdjacentRow(extents, sign, outside, extents[1] - 1, k, 0, -1);
  }
}


inline
std::size_t
countBits(const std::vector<std::size_t>& outside)
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != outside.size(); ++i) {
    count += numerical::popCount(outside[i]);
  }
  return count;
}


template<typename _T>
inline
void
markOutside(Grid<_T, 3, 8>* grid,
            const std::array<std::size_t, 3>& byteExtents,
            const std::vector<std::size_t>& outside)
{
  const std::size_t D = 3;
  const std::size_t N = 8;

  typedef Grid<_T, D, N> Grid;
  typedef typename Grid::VertexPatch VertexPatch;
  typedef typename Grid::IndexList IndexList;
  typedef container::SimpleMultiIndexRangeIterator<D> Iterator;

  // Strides for indexing the multi-array of bytes.
  // Below we will use the fact that byteStrides[0] is 1 to avoid a
  // multiplication.
  const IndexList byteStrides = {{
      1, byteExtents[0],
      byteExtents[0]* byteExtents[1]
    }
  };

  // Note: This does not violate strict aliasing rules. It's is OK to
  // use a character type to alias any other type.
  const unsigned char* outsideBytes =
    reinterpret_cast<const unsigned char*>(&outside[0]);
  // Loop over the patches.
  const Iterator pEnd = Iterator::end(grid->extents());
  for (Iterator p = Iterator::begin(grid->extents()); p != pEnd; ++p) {
    // Grid index.
    const IndexList& gi = *p;
    VertexPatch& patch = (*grid)(gi);
    if (! patch.isRefined()) {
      // Check the first row in the patch. All of the bits should be either
      // 0 or 1.
      if (outsideBytes[gi[0] + gi[1] * N * byteStrides[1] +
                       gi[2] * N * byteStrides[2]]) {
        patch.fillValue = - std::numeric_limits<_T>::infinity();
      }
    }
    // Refined patch.
    else {
      // Single index for the patch elements. Incremented in the innermost
      // loop.
      std::size_t n = 0;
      // Loop over the second and third dimensions.
      for (std::size_t k = 0; k != N; ++k) {
        for (std::size_t j = 0; j != N; ++j) {
          // Unpack the outside information.
          const unsigned char out =
            outsideBytes[gi[0] + (gi[1] * N + j) * byteStrides[1] +
                         (gi[2] * N + k) * byteStrides[2]];
          unsigned char mask = 1;
          for (std::size_t i = 0; i != N; ++i, ++n) {
            if (mask & out) {
              patch[n] = - std::numeric_limits<_T>::infinity();
            }
            mask <<= 1;
          }
        }
      }
    }
  }
}


// We require the grid patch extent to match the number of bits in a byte.
template<typename _T>
inline
void
markOutsideAsNegativeInf(Grid<_T, 3, 8>* grid)
{
  const std::size_t D = 3;
  const std::size_t N = 8;

  typedef Grid<_T, D, N> Grid;
  typedef typename Grid::IndexList IndexList;

  // Determine the grid extents for the bit arrays. Because we access the
  // bits using std::size_t, round up to a multiple of its size (in bytes)
  // in the first dimension.
  const IndexList byteExtents = {
    {
      (grid->extents()[0] + (sizeof(std::size_t) - 1)) / sizeof(std::size_t)*
      sizeof(std::size_t),
      N* grid->extents()[1],
      N* grid->extents()[2]
    }
  };
  const std::size_t numBytes = ext::product(byteExtents);
  // The integer type is std::size_t.
  const IndexList integerExtents = {{
      byteExtents[0] / sizeof(std::size_t),
      byteExtents[1], byteExtents[2]
    }
  };

  // The sign array.
  // Start by marking the sign of the distance as positive. We do this because
  // the bit arrays may be larger than the grid in the x direction. Any
  // extra is definitely positive (outside).
  std::vector<std::size_t> sign(numBytes / sizeof(std::size_t),
                                std::numeric_limits<std::size_t>::max());

  // First mark the points (both unrefined patches and refined grid points)
  // that have negative distances.
  setSign(*grid, byteExtents, &sign);

  // The outside array. Start with nothing marked as outside.
  std::vector<std::size_t> outside(sign.size(), 0);

  // Set the boundary condition.
  setBoundaryCondition(integerExtents, sign, &outside);

  // Loop until the number of outside bits does not change.
  std::size_t oldCount = 1;
  std::size_t count = 0;
  while (count != oldCount) {
    // Sweep in the z directions, including the diagonal directions.
    sweepZ(integerExtents, sign, &outside);
    // Sweep in the y directions.
    sweepY(integerExtents, sign, &outside);
    // Sweep in the x directions.
    sweepXLocal(integerExtents, sign, &outside);
    // Count the number of outside grid points.
    oldCount = count;
    count = countBits(outside);
  }

  // Mark the outside points as negative infinity.
  markOutside(grid, byteExtents, outside);
}


} // namespace levelSet
}
