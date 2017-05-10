// -*- C++ -*-

#if !defined(__levelSet_Grid_ipp__)
#error This file is an implementation detail of Grid.
#endif

namespace stlib
{
namespace levelSet
{


// Construct from the Cartesian domain and the suggested grid patch spacing.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
Grid<_T, _D, N, _R>::
Grid(const BBox& domain, const _R targetSpacing) :
  GeometryBase(domain, targetSpacing),
  ArrayBase(GeometryBase::gridExtents),
  // Don't allocate memory for the patch arrays until the grid is refined.
  _data()
{
}


// Construct from the Cartesian domain and the grid extents.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
Grid<_T, _D, N, _R>::
Grid(const BBox& domain, const IndexList& extents) :
  GeometryBase(domain, extents),
  ArrayBase(extents),
  // Don't allocate memory for the patch arrays until the grid is refined.
  _data()
{
}


// Return the value at the specified grid point.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
_T
Grid<_T, _D, N, _R>::
operator()(const IndexList& patch,
           const IndexList& index) const
{
  if (ArrayBase::operator()(patch).isRefined()) {
    return ArrayBase::operator()(patch)(index);
  }
  return ArrayBase::operator()(patch).fillValue;
}


template<typename _T, std::size_t N, typename _R>
inline
void
_getVoxelPatch(const Grid<_T, 1, N, _R>& grid,
               typename Grid<_T, 1, N, _R>::IndexList index,
               typename Grid<_T, 1, N, _R>::VoxelPatch* patch)
{
  typedef Grid<_T, 1, N, _R> Grid;
  typedef typename Grid::VertexPatch VertexPatch;

  const VertexPatch& p = grid(index);
  assert(p.isRefined());
  for (std::size_t i = 0; i != N; ++i) {
    (*patch)[i] = p[i];
  }
  (*patch)[N] = (*patch)[N - 1];
  ++index[0];
  if (index[0] != grid.extents()[0] && grid(index).isRefined()) {
    (*patch)[N] = grid(index)[0];
  }
}


template<typename _T, std::size_t N, typename _R>
inline
void
_getVoxelPatch(const Grid<_T, 2, N, _R>& grid,
               typename Grid<_T, 2, N, _R>::IndexList index,
               typename Grid<_T, 2, N, _R>::VoxelPatch* patch)
{
  const std::size_t Dimension = 2;
  typedef Grid<_T, Dimension, N, _R> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::VertexPatch VertexPatch;

  IndexList i, j;
  {
    const VertexPatch& p = grid(index);
    assert(p.isRefined());
    // Body.
    for (i[0] = 0; i[0] != N; ++i[0]) {
      for (i[1] = 0; i[1] != N; ++i[1]) {
        (*patch)(i) = p(i);
      }
    }
    // Sides.
    for (std::size_t a = 0; a != Dimension; ++a) {
      const std::size_t b = (a + 1) % Dimension;
      i[a] = N;
      j[a] = N - 1;
      for (i[b] = 0; i[b] != N; ++i[b]) {
        j[b] = i[b];
        (*patch)(i) = p(j);
      }
    }
    // Corner.
    (*patch)(N, N) = p(N - 1, N - 1);
  }

  // Sides.
  for (std::size_t a = 0; a != Dimension; ++a) {
    ++index[a];
    if (index[a] != grid.extents()[a] && grid(index).isRefined()) {
      const VertexPatch& p = grid(index);
      const std::size_t b = (a + 1) % Dimension;
      i[a] = N;
      j[a] = 0;
      for (i[b] = 0; i[b] != N; ++i[b]) {
        j[b] = i[b];
        (*patch)(i) = p(j);
      }
    }
    --index[a];
  }

  // Corner.
  ++index[0];
  ++index[1];
  if (index[0] != grid.extents()[0] && index[1] != grid.extents()[1] &&
      grid(index).isRefined()) {
    const VertexPatch& p = grid(index);
    (*patch)(N, N) = p(0, 0);
  }
}


template<typename _T, std::size_t N, typename _R>
inline
void
_getVoxelPatch(const Grid<_T, 3, N, _R>& grid,
               typename Grid<_T, 3, N, _R>::IndexList index,
               typename Grid<_T, 3, N, _R>::VoxelPatch* patch)
{
  const std::size_t Dimension = 3;
  typedef Grid<_T, Dimension, N, _R> Grid;
  typedef typename Grid::IndexList IndexList;
  typedef typename Grid::VertexPatch VertexPatch;

  IndexList i, j;
  {
    const VertexPatch& p = grid(index);
    assert(p.isRefined());
    // Body.
    for (i[0] = 0; i[0] != N; ++i[0]) {
      for (i[1] = 0; i[1] != N; ++i[1]) {
        for (i[2] = 0; i[2] != N; ++i[2]) {
          (*patch)(i) = p(i);
        }
      }
    }
    // Sides.
    for (std::size_t a = 0; a != Dimension; ++a) {
      const std::size_t b = (a + 1) % Dimension;
      const std::size_t c = (a + 2) % Dimension;
      i[a] = N;
      j[a] = N - 1;
      for (i[b] = 0; i[b] != N; ++i[b]) {
        for (i[c] = 0; i[c] != N; ++i[c]) {
          j[b] = i[b];
          j[c] = i[c];
          (*patch)(i) = p(j);
        }
      }
    }
    // Edges.
    for (std::size_t a = 0; a != Dimension; ++a) {
      const std::size_t b = (a + 1) % Dimension;
      const std::size_t c = (a + 2) % Dimension;
      i[b] = i[c] = N;
      j[b] = j[c] = N - 1;
      for (i[a] = 0; i[a] != N; ++i[a]) {
        j[a] = i[a];
        (*patch)(i) = p(j);
      }
    }
    // Corner.
    (*patch)(N, N, N) = p(N - 1, N - 1, N - 1);
  }

  // Sides.
  for (std::size_t a = 0; a != Dimension; ++a) {
    ++index[a];
    if (index[a] != grid.extents()[a] && grid(index).isRefined()) {
      const VertexPatch& p = grid(index);
      const std::size_t b = (a + 1) % Dimension;
      const std::size_t c = (a + 2) % Dimension;
      i[a] = N;
      j[a] = 0;
      for (i[b] = 0; i[b] != N; ++i[b]) {
        for (i[c] = 0; i[c] != N; ++i[c]) {
          j[b] = i[b];
          j[c] = i[c];
          (*patch)(i) = p(j);
        }
      }
    }
    --index[a];
  }
  // Edges.
  for (std::size_t a = 0; a != Dimension; ++a) {
    const std::size_t b = (a + 1) % Dimension;
    const std::size_t c = (a + 2) % Dimension;
    ++index[b];
    ++index[c];
    if (index[b] != grid.extents()[b] && index[c] != grid.extents()[c] &&
        grid(index).isRefined()) {
      const VertexPatch& p = grid(index);
      i[b] = i[c] = N;
      j[b] = j[c] = 0;
      for (i[a] = 0; i[a] != N; ++i[a]) {
        j[a] = i[a];
        (*patch)(i) = p(j);
      }
    }
    --index[b];
    --index[c];
  }
  // Corner.
  ++index[0];
  ++index[1];
  ++index[2];
  if (index[0] != grid.extents()[0] && index[1] != grid.extents()[1] &&
      index[2] != grid.extents()[2] && grid(index).isRefined()) {
    const VertexPatch& p = grid(index);
    (*patch)(N, N, N) = p(0, 0, 0);
  }
}


// Get the specified voxel patch.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
Grid<_T, _D, N, _R>::
getVoxelPatch(const IndexList& i, VoxelPatch* patch) const
{
  // A level of indirection for the space dimension.
  _getVoxelPatch(*this, i, patch);
}


// Get the number of refined patches.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
std::size_t
Grid<_T, _D, N, _R>::
numRefined() const
{
  std::size_t count = 0;
  for (std::size_t i = 0; i != ArrayBase::size(); ++i) {
    count += ArrayBase::operator[](i).isRefined();
  }
  return count;
}


// Report the set of adjacent neighbors in the specified direction.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
template<typename _OutputIterator>
inline
void
Grid<_T, _D, N, _R>::
adjacentNeighbors(const IndexList& patch,
                  const IndexList& index,
                  const std::size_t direction,
                  _OutputIterator neighbors) const
{
#ifdef STLIB_DEBUG
  assert(direction < 2 * _D);
#endif
  typedef IndexList IndexList;
  typedef typename ArrayBase::Range Range;
  const IndexList NullIndex =
    ext::filled_array<IndexList>(std::numeric_limits<std::size_t>::max());
  const std::size_t coordinate = direction / 2;
  IndexList p = patch;
  IndexList i = index;
  // Refined.
  if (ArrayBase::operator()(patch).isRefined()) {
    // Negative direction.
    if (direction % 2 == 0) {
      // On lower face.
      if (index[coordinate] == 0) {
        if (patch[coordinate] != 0) {
          --p[coordinate];
          if (ArrayBase::operator()(p).isRefined()) {
            i[coordinate] = N - 1;
            *neighbors++ = std::make_pair(p, i);
          }
          else {
            *neighbors++ = std::make_pair(p, NullIndex);
          }
        }
      }
      else {
        --i[coordinate];
        *neighbors++ = std::make_pair(p, i);
      }
    }
    // Positive direction.
    else {
      // On upper face.
      if (index[coordinate] == N - 1) {
        if (patch[coordinate] != ArrayBase::extents()[coordinate] - 1) {
          ++p[coordinate];
          if (ArrayBase::operator()(p).isRefined()) {
            i[coordinate] = 0;
            *neighbors++ = std::make_pair(p, i);
          }
          else {
            *neighbors++ = std::make_pair(p, NullIndex);
          }
        }
      }
      else {
        ++i[coordinate];
        *neighbors++ = std::make_pair(p, i);
      }
    }
  }
  // Unrefined.
  else {
    // Negative direction.
    if (direction % 2 == 0) {
      if (patch[coordinate] != 0) {
        --p[coordinate];
        // Neighbor refined.
        if (ArrayBase::operator()(p).isRefined()) {
          IndexList extents = ext::filled_array<IndexList>(N);
          extents[coordinate] = 1;
          IndexList bases = ext::filled_array<IndexList>(0);
          bases[coordinate] = N - 1;
          Range range = {extents, bases};
          GeometryBase::report(p, range, neighbors);
        }
        // Neighbor unrefined.
        else {
          *neighbors++ = std::make_pair(p, NullIndex);
        }
      }
    }
    // Positive direction.
    else {
      if (patch[coordinate] != ArrayBase::extents()[coordinate] - 1) {
        ++p[coordinate];
        // Neighbor refined.
        if (ArrayBase::operator()(p).isRefined()) {
          IndexList extents = ext::filled_array<IndexList>(N);
          extents[coordinate] = 1;
          IndexList bases = ext::filled_array<IndexList>(0);
          Range range = {extents, bases};
          GeometryBase::report(p, range, neighbors);
        }
        // Neighbor unrefined.
        else {
          *neighbors++ = std::make_pair(p, NullIndex);
        }
      }
    }
  }
}


// Report the set of all adjacent neighbors.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
template<typename _OutputIterator>
inline
void
Grid<_T, _D, N, _R>::
adjacentNeighbors(const DualIndices& pair, _OutputIterator neighbors) const
{
  for (std::size_t direction = 0; direction != 2 * _D; ++direction) {
    adjacentNeighbors(pair.first, pair.second, direction, neighbors);
  }
}


// Report the set of all neighbors.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
template<typename _OutputIterator>
inline
void
Grid<_T, _D, N, _R>::
allNeighbors(const DualIndices& pair, _OutputIterator neighbors) const
{
  typedef IndexList IndexList;
  typedef typename ArrayBase::Range Range;
  typedef container::SimpleMultiIndexRangeIterator<_D> Iterator;

  const IndexList& patch = pair.first;
  const IndexList& index = pair.second;

  // Note that in this function we will work with global indices.
  // g = p * N + i.
  IndexList lower, upper;
  DualIndices di;
  // Refined.
  if (ArrayBase::operator()(patch).isRefined()) {
    // Define a range that contains the neighbors of the point.
    for (std::size_t d = 0; d != _D; ++d) {
      lower[d] = patch[d] * N + index[d];
      if (lower[d] != 0) {
        --lower[d];
      }
      if (patch[d] == ArrayBase::extents()[d] - 1 && index[d] == N - 1) {
        upper[d] = ArrayBase::extents()[d] * N;
      }
      else {
        upper[d] = patch[d] * N + index[d] + 2;
      }
    }
    // Loop over the range. Note that unrefined patches may be reported
    // multiple times.
    const Range range = {upper - lower, lower};
    const Iterator end = Iterator::end(range);
    for (Iterator i = Iterator::begin(range); i != end; ++i) {
      di.first = *i / N;
      di.second = *i % N;
      // Skip the grid point itself.
      if (di == pair) {
        continue;
      }
      *neighbors++ = di;
    }
  }
  // Unrefined.
  else {
    // Define a range that contains the neighbors of the patch.
    for (std::size_t d = 0; d != _D; ++d) {
      if (patch[d] == 0) {
        lower[d] = 0;
      }
      else {
        lower[d] = patch[d] * N - 1;
      }
      if (patch[d] == ArrayBase::extents()[d] - 1) {
        upper[d] = ArrayBase::extents()[d] * N;
      }
      else {
        upper[d] = (patch[d] + 1) * N + 1;
      }
    }
    // Loop over the range and skip the patch itself.
    const Range range = {upper - lower, lower};
    const Iterator end = Iterator::end(range);
    for (Iterator i = Iterator::begin(range); i != end; ++i) {
      di.first = *i / N;
      if (di.first == patch) {
        continue;
      }
      di.second = *i % N;
      *neighbors++ = di;
    }
  }
}


// Return true if the grid is valid.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
bool
Grid<_T, _D, N, _R>::
isValid() const
{
  if (! GeometryBase::isValid()) {
    return false;
  }
  // Check each patch.
  for (std::size_t i = 0; i != ArrayBase::size(); ++i) {
    if (! ArrayBase::operator[](i).isValid()) {
      return false;
    }
  }
  // Check the memory.
  typename std::vector<_T>::const_pointer p = &_data[0];
  for (std::size_t i = 0; i != ArrayBase::size(); ++i) {
    if (ArrayBase::operator[](i).isRefined()) {
      if (ArrayBase::operator[](i).data() != p) {
        return false;
      }
      p += NumVerticesPerPatch;
    }
  }
  if (p != &_data[0] + _data.size()) {
    return false;
  }
  return true;
}


// Return a reference to the value at the specified grid point.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
_T&
Grid<_T, _D, N, _R>::
operator()(const IndexList& patch,
           const IndexList& index)
{
  if (ArrayBase::operator()(patch).isRefined()) {
    return ArrayBase::operator()(patch)(index);
  }
#if 0
  // For no good reason, this form generates a compilation error when using
  // CUDA 4.0.
  return ArrayBase::operator()(patch).fillValue;
#else
  VertexPatch& p = ArrayBase::operator()(patch);
  return p.fillValue;
#endif
}


// Make all of the grids unrefined.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
Grid<_T, _D, N, _R>::
clear()
{
  for (std::size_t i = 0; i != ArrayBase::size(); ++i) {
    ArrayBase::operator[](i).clear();
  }
  _data.clear();
}


// Refine the specified grids.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
Grid<_T, _D, N, _R>::
refine(const std::vector<std::size_t>& indices)
{
  // First clear the grid.
  clear();
  // Then refine the specified patches.
  _data.resize(indices.size() * NumVerticesPerPatch);
  // CONTINUE: Fix for integer value types.
  std::fill(_data.begin(), _data.end(),
            std::numeric_limits<_T>::quiet_NaN());
  std::size_t dataIndex = 0;
  for (std::size_t i = 0; i != indices.size(); ++i) {
    const std::size_t index = indices[i];
    ArrayBase::operator[](index).refine(&_data[dataIndex]);
    dataIndex += NumVerticesPerPatch;
  }
  // Check that we used all of the data.
  assert(dataIndex == _data.size());
}


// Refine the patches that have one or more dependencies.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
Grid<_T, _D, N, _R>::
refine(const container::StaticArrayOfArrays<unsigned>& dependencies)
{
  std::vector<std::size_t> patchIndices;
  for (std::size_t i = 0; i != dependencies.getNumberOfArrays(); ++i) {
    if (! dependencies.empty(i)) {
      patchIndices.push_back(i);
    }
  }
  // Refine the appropriate patches and set the rest to have an unknown
  // distance.
  refine(patchIndices);
}


template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
Grid<_T, _D, N, _R>::
coarsen()
{
  // Pack the array.
  std::size_t target = 0;
  // For each refined patch.
  for (std::size_t i = 0; i != ArrayBase::size(); ++i) {
    VertexPatch& patch = ArrayBase::operator[](i);
    if (! patch.isRefined()) {
      continue;
    }
    if (patch.shouldBeCoarsened()) {
      patch.coarsen();
    }
    else {
      // If the patch data needs to be moved.
      if (patch.data() != &_data[target]) {
        std::copy(patch.begin(), patch.end(), &_data[target]);
      }
      patch.refine(&_data[target]);
      target += NumVerticesPerPatch;
    }
  }

  // Resize.
  _data.resize(target);
}


// Add a constant to all vertices and fill values.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
Grid<_T, _D, N, _R>&
operator+=(Grid<_T, _D, N, _R>& grid, const _T x)
{
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (grid[i].isRefined()) {
      grid[i] += x;
    }
    else {
      grid[i].fillValue += x;
    }
  }
  return grid;
}


// Subtract a constant from all vertices and fill values.
template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
Grid<_T, _D, N, _R>&
operator-=(Grid<_T, _D, N, _R>& grid, const _T x)
{
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (grid[i].isRefined()) {
      grid[i] -= x;
    }
    else {
      grid[i].fillValue -= x;
    }
  }
  return grid;
}


template<typename _T, std::size_t N, typename _R>
inline
void
writeVtkXml(const Grid<_T, 3, N, _R>& grid, std::ostream& out)
{
  const std::size_t Dimension = 3;
  typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;
  typedef typename Grid<_T, Dimension, N, _R>::VertexPatch VertexPatch;

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  out << "<ImageData WholeExtent=\""
      << "0 " << grid.extents()[0] * N - 1 << ' '
      << "0 " << grid.extents()[1] * N - 1 << ' '
      << "0 " << grid.extents()[2] * N - 1
      << "\" Origin=\"" << grid.lowerCorner
      << "\" Spacing=\""
      << grid.spacing << ' '
      << grid.spacing << ' '
      << grid.spacing
      << "\">\n";

  // For each refined patch.
  const Iterator end = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
    out << "<Piece Extent=\""
        << (*i)[0] * N << ' '
        << ((*i)[0] + 1) * N - 1 << ' '
        << (*i)[1] * N << ' '
        << ((*i)[1] + 1) * N - 1 << ' '
        << (*i)[2] * N << ' '
        << ((*i)[2] + 1) * N - 1
        << "\">\n";

    out << "<PointData Scalars=\"Distance\">\n"
        << "<DataArray type=\"Float64\" Name=\"Distance\" "
        << "NumberOfComponents=\"1\" format=\"ascii\">\n";
    // Loop over the vertices in the patch.
    const VertexPatch& patch = grid(*i);
    if (patch.isRefined()) {
      std::copy(patch.begin(), patch.end(),
                std::ostream_iterator<_T>(out, " "));
    }
    else {
      for (std::size_t i = 0; i != patch.size(); ++i) {
        out << patch.fillValue << ' ';
      }
    }
    out << "</DataArray>\n";
    out << "</PointData>\n";

    out << "<CellData></CellData>\n";
    out << "</Piece>\n";
  }
  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


template<typename _T, std::size_t N, typename _R>
inline
void
writeVtkXml(const Grid<_T, 2, N, _R>& grid, std::ostream& out)
{
  typedef container::SimpleMultiIndexRangeIterator<2> Iterator;
  typedef Grid<_T, 2, N, _R> Grid;
  typedef typename Grid::VertexPatch Patch;

  // Construct a uniform grid.
  GridUniform<_T, 2> uniform(grid.extents() * N, grid.lowerCorner,
                             grid.spacing);

  //
  // Copy the elements.
  //
  // For each patch.
  const Iterator iEnd = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != iEnd; ++i) {
    const Patch& patch = grid(*i);
    // For each vertex.
    if (patch.isRefined()) {
      const Iterator jEnd = Iterator::end(patch.extents());
      for (Iterator j = Iterator::begin(patch.extents()); j != jEnd;
           ++j) {
        uniform(*i * N + *j) = patch(*j);
      }
    }
    else {
      const Iterator jEnd = Iterator::end(patch.extents());
      for (Iterator j = Iterator::begin(patch.extents()); j != jEnd;
           ++j) {
        uniform(*i * N + *j) = patch.fillValue;
      }
    }
  }

  // Write the uniform grid.
  writeVtkXml(uniform, out);
}


// CONTINUE Writing in multiple pieces does not work. The result doesn't
// display correctly in Paraview, even when all the pieces are output.
template<typename _T, std::size_t N, typename _R>
inline
void
writeVtkXmlOld(const Grid<_T, 2, N, _R>& grid, std::ostream& out)
{
  const std::size_t Dimension = 2;
  typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;
  typedef typename Grid<_T, Dimension, N, _R>::VertexPatch VertexPatch;

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  out << "<ImageData WholeExtent=\""
      << "0 " << grid.extents()[0] * N - 1 << ' '
      << "0 " << grid.extents()[1] * N - 1 << ' '
      << "0 0"
      << "\" Origin=\"" << grid.lowerCorner << " 0"
      << "\" Spacing=\""
      << grid.spacing << ' '
      << grid.spacing << ' '
      << grid.spacing
      << "\">\n";

  // For each refined patch.
  const Iterator end = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
    if (! grid(*i).isRefined()) {
      continue;
    }
    out << "<Piece Extent=\""
        << (*i)[0] * N << ' '
        << ((*i)[0] + 1) * N - 1 << ' '
        << (*i)[1] * N << ' '
        << ((*i)[1] + 1) * N - 1 << ' '
        << " 0 0\">\n";

    out << "<PointData Scalars=\"Distance\">\n"
        << "<DataArray type=\"Float64\" Name=\"Distance\" "
        << "NumberOfComponents=\"1\" format=\"ascii\">\n";
    // Loop over the vertices in the patch.
    const VertexPatch& patch = grid(*i);
    std::copy(patch.begin(), patch.end(),
              std::ostream_iterator<_T>(out, " "));
    out << "</DataArray>\n";
    out << "</PointData>\n";

    out << "<CellData></CellData>\n";
    out << "</Piece>\n";
  }
  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


template<typename _T, std::size_t N, typename _R>
inline
void
writeVtkXml(const Grid<_T, 2, N, _R>& grid,
            const std::array<std::size_t, 2>& patchIndices,
            std::ostream& out)
{
  const std::size_t Dimension = 2;
  typedef typename Grid<_T, Dimension, N, _R>::VertexPatch VertexPatch;

  out << "<?xml version=\"1.0\"?>\n"
      << "<VTKFile type=\"ImageData\">\n";

  out << "<ImageData WholeExtent=\""
      << "0 " << grid.extents()[0] * N - 1 << ' '
      << "0 " << grid.extents()[1] * N - 1 << ' '
      << "0 0"
      << "\" Origin=\"" << grid.lowerCorner << " 0"
      << "\" Spacing=\""
      << grid.spacing << ' '
      << grid.spacing << ' '
      << grid.spacing
      << "\">\n";

  out << "<Piece Extent=\""
      << patchIndices[0] * N << ' '
      << (patchIndices[0] + 1) * N - 1 << ' '
      << patchIndices[1] * N << ' '
      << (patchIndices[1] + 1) * N - 1 << ' '
      << " 0 0\">\n";

  out << "<PointData Scalars=\"Distance\">\n"
      << "<DataArray type=\"Float64\" Name=\"Distance\" "
      << "NumberOfComponents=\"1\" format=\"ascii\">\n";
  // Loop over the vertices in the patch.
  const VertexPatch& patch = grid(patchIndices);
  if (patch.isRefined()) {
    std::copy(patch.begin(), patch.end(),
              std::ostream_iterator<_T>(out, " "));
  }
  else {
    for (std::size_t i = 0; i != patch.size(); ++i) {
      out << patch.fillValue << ' ';
    }
  }
  out << "</DataArray>\n";
  out << "</PointData>\n";

  out << "<CellData></CellData>\n";
  out << "</Piece>\n";

  out << "</ImageData>\n";
  out << "</VTKFile>\n";
}


// CONTINUE Writing in multiple pieces does not work. The result doesn't
// display correctly in Paraview, even when all the pieces are output.
template<typename _T, std::size_t N, typename _R>
inline
void
writeParaview(const Grid<_T, 2, N, _R>& grid, const std::string& name)
{
  const std::size_t Dimension = 2;
  typedef container::SimpleMultiIndexRangeIterator<Dimension> Iterator;

  // Open the ParaView file.
  std::string paraviewName = name;
  paraviewName += ".pvd";
  std::ofstream paraviewFile(paraviewName.c_str());
  paraviewFile << "<?xml version=\"1.0\"?>\n"
               << "<VTKFile type=\"Collection\">\n"
               << "<Collection>\n";
  // For each refined patch.
  const Iterator end = Iterator::end(grid.extents());
  for (Iterator i = Iterator::begin(grid.extents()); i != end; ++i) {
    std::string vtkName = name;
    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(6) << grid.arrayIndex(*i);
    vtkName += oss.str();
    vtkName += ".vti";

    paraviewFile << "<DataSet part=\"1\" file=\"" << vtkName << "\"/>\n";

    std::ofstream vtkFile(vtkName.c_str());
    writeVtkXml(grid, *i, vtkFile);
  }
  paraviewFile << "</Collection>\n";
  paraviewFile << "</VTKFile>\n";
}


template<typename _T, std::size_t _D, std::size_t N, typename _R>
inline
void
printInfo(const Grid<_T, _D, N, _R>& grid, std::ostream& out)
{
  const _T Inf = std::numeric_limits<_T>::infinity();

  std::size_t size = 0;
  std::size_t nonNegative = 0;
  std::size_t negative = 0;
  std::size_t unknown = 0;
  std::size_t positiveFar = 0;
  std::size_t negativeFar = 0;
  _T x;
  // For each refined patch.
  for (std::size_t i = 0; i != grid.size(); ++i) {
    if (! grid[i].isRefined()) {
      continue;
    }
    // For each grid point in the patch.
    for (std::size_t j = 0; j != grid[i].size(); ++j) {
      x = grid[i][j];
      ++size;
      if (x >= 0) {
        ++nonNegative;
      }
      else if (x < 0) {
        ++negative;
      }
      else {
        ++unknown;
      }
      if (x == Inf) {
        ++positiveFar;
      }
      else if (x == -Inf) {
        ++negativeFar;
      }
    }
  }
  printInfo(static_cast<const GridGeometry<_D, N, _R>&>(grid), out);
  out << "Number of patches = " << grid.size() << '\n'
      << "Number of refined patches = " << grid.numRefined() << '\n'
      << "Number of grid points = " << size << '\n'
      << "known/unknown = " << size - unknown << " / " << unknown << '\n'
      << "non-negative/negative = " << nonNegative << " / " << negative
      << '\n'
      << "positive far/negative far = " << positiveFar << " / " << negativeFar
      << '\n';
}


} // namespace levelSet
}
