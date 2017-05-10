// -*- C++ -*-

#if !defined(__numerical_interp_extrap_ipp__)
#error This file is an implementation detail of interp_extrap.
#endif

namespace stlib
{
namespace numerical
{

// Routine for 1-D linear interpolation/extrapolation.
// OUTPUT:
// value - The tuple of the interpolated fields.
// isValueKnown - If the value can be determined.  This will be false
// iff the fields are known at none of the four points.
// INPUT:
// position - The position in index coordinates where the field will be
//            interpolated.  This lies between the second and third
//            grid points.
// field - The value of the field at the four grid points.
// isKnown - If the value of the field is known at these grid points.
// lowerIndex - The index of the first grid point.
//
// I use FieldTuple instead of templating on the tuple dimension.  This way
// the argument can be a scalar or a tuple.
template<typename FieldTuple, typename T>
inline
void
intExt(FieldTuple* value,
       bool* isValueKnown,
       const std::array<FieldTuple, 4>& field,
       const std::array<bool, 4>& isKnown,
       const T position,
       const std::ptrdiff_t lowerIndex)
{
  assert(lowerIndex + 1 <= position && position <= lowerIndex + 2);

  *isValueKnown = true;
  if (isKnown[1] && isKnown[2]) {
    // Linear interpolation.
    *value = field[1] + (field[2] - field[1]) *
             (position - (lowerIndex + 1));
  }
  else if (isKnown[1]) {
    // Constant extrapolation.
    *value = field[1];
  }
  else if (isKnown[2]) {
    // Constant extrapolation.
    *value = field[2];
  }
  else if (isKnown[0] && isKnown[3]) {
    // Choose the closer one.
    if (position - lowerIndex < 1.5) {
      // Constant extrapolation.
      *value = field[0];
    }
    else {
      // Constant extrapolation.
      *value = field[3];
    }
  }
  else if (isKnown[0]) {
    // Constant extrapolation.
    *value = field[0];
  }
  else if (isKnown[3]) {
    // Constant extrapolation.
    *value = field[3];
  }
  else {
    *isValueKnown = false;
  }
}

template<std::size_t M, typename T>
inline
void
makeFieldsTuple(std::array<std::array<T, M>, 4>* fieldsTuple,
                const container::MultiArrayConstRef < T, 1 + 1 > fields,
                container::MultiIndexTypes<1>::IndexList i)
{
  typedef container::MultiIndexTypes < 1 + 1 >::Index Index;
  container::MultiIndexTypes < 1 + 1 >::IndexList mi;
  for (Index x = 0; x != 4; ++x) {
    mi[1] = i[0] + x;
    for (mi[0] = 0; mi[0] != M; ++mi[0]) {
      (*fieldsTuple)[x][mi[0]] = fields(mi);
    }
  }
}


// 1-D interpolation/extrapolation at a single point.  There are M scalar
// field arrays.
// OUTPUT:
//   values - The values of the interpolated fields.
// INPUT:
//   position - The Cartesian position at which to interpolate.
//   field - An array of values to be interpolated.
//   distance - The array of distances.
//   grid - Holds domain and grid information.
//   defaultValues - If no neigboring grid points are known, values is set
//     to defaultValues.
// TEMPLATE PARAMETERS:
//   M is the number of fields.
//   T in the number type.
// NOTE:
// I use Arrays so that this function can be called from a function
// that is templated on the dimension, N.
template<std::size_t M, typename T>
inline
void
intExt(std::array<T, M>* values,
       const std::array<T, 1>& position,
       const std::array<T, M>& defaultValues,
       const geom::RegularGrid<1, T>& grid,
       const container::MultiArrayConstRef<T, 1> distance,
       const container::MultiArrayConstRef < T, 1 + 1 > fields)
{
  // The single index type.
  typedef container::MultiIndexTypes<1>::Index Index;
  // Multi-index.
  typedef container::MultiIndexTypes<1>::IndexList IndexList;

  //
  // The position is surrounded by 4 grid points.
  // First calculate the index of the left point.
  //
  std::array<T, 1> continuousIndex = position;
  grid.convertLocationToIndex(&continuousIndex);
  const Index i = static_cast<Index>(std::floor(continuousIndex[0])) - 1;

  //
  // Only do the interpolation/extrapolation if the 4 grid points are
  // within this grid.
  //
  if (!(0 <= i && i <= Index(grid.getExtents()[0]) - 4)) {
    return;
  }

  std::array<std::array<T, M>, 4> fieldsTuple;
  IndexList mi;
  mi[0] = i;
  makeFieldsTuple(&fieldsTuple, fields, mi);

  std::array<bool, 4> isKnownTuple;
  for (std::size_t x = 0; x != 4; ++x) {
    mi[0] = i + x;
    isKnownTuple[x] = (0 <= distance(mi));
  }

  //
  // Interpolate/extrapolate.
  //
  bool isValueKnown;
  intExt(values, &isValueKnown,
         fieldsTuple, isKnownTuple, continuousIndex[0], i);

  if (! isValueKnown) {
    *values = defaultValues;
  }
}



template<std::size_t M, typename T>
inline
void
makeFieldsTuple(std::array<std::array<T, M>, 4>* fieldsTuple,
                const container::MultiArrayConstRef < T, 2 + 1 > fields,
                container::MultiIndexTypes<2>::IndexList i)
{
  typedef container::MultiIndexTypes < 2 + 1 >::Index Index;
  container::MultiIndexTypes < 2 + 1 >::IndexList mi;
  mi[2] = i[1];
  for (Index x = 0; x != 4; ++x) {
    mi[1] = i[0] + x;
    for (mi[0] = 0; mi[0] != M; ++mi[0]) {
      (*fieldsTuple)[x][mi[0]] = fields(mi);
    }
  }
}

// 2-D
// OUTPUT:
//   values - The values of the interpolated fields.
// INPUT:
//   position - The Cartesian position at which to interpolate.
//   field - An array of values to be interpolated.
//   distance - The array of distances.
//   grid - Holds domain and grid information.
//   defaultValues - If no neigboring grid points are known, values is set
//     to defaultValues.
// TEMPLATE PARAMETERS
//   M is the number of fields.
//   T in the number type.
template<std::size_t M, typename T>
inline
void
intExt(std::array<T, M>* values,
       const std::array<T, 2>& position,
       const std::array<T, M>& defaultValues,
       const geom::RegularGrid<2, T>& grid,
       const container::MultiArrayConstRef<T, 2> distance,
       const container::MultiArrayConstRef < T, 2 + 1 > fields)
{
  // The single index type.
  typedef container::MultiIndexTypes<2>::Index Index;
  // Multi-index.
  typedef container::MultiIndexTypes<2>::IndexList IndexList;

  //
  // The position lies in a square defined by 16 grid points.
  // First calculate the indices of the lower corner.
  //
  std::array<T, 2> continuousIndex = position;
  grid.convertLocationToIndex(&continuousIndex);
  const Index i = static_cast<Index>(std::floor(continuousIndex[0])) - 1;
  const Index j = static_cast<Index>(std::floor(continuousIndex[1])) - 1;

  //
  // Only do the interpolation/extrapolation if the 16 grid points are
  // within this grid.
  //
  if (!(0 <= i && i <= Index(grid.getExtents()[0]) - 4 &&
        0 <= j && j <= Index(grid.getExtents()[1]) - 4)) {
    return;
  }

  std::array<std::array<T, M>, 4> fieldsTuple;
  std::array<bool, 4> isKnownTuple;

  //
  // Interpolate in the x direction.
  //
  std::array<std::array<T, M>, 4> xInterpFields;
  std::array<bool, 4> xInterpIsKnown;
  Index x, y;
  IndexList mi;

  for (y = 0; y != 4; ++y) {
    mi[0] = i;
    mi[1] = j + y;
    makeFieldsTuple(&fieldsTuple, fields, mi);
    for (x = 0; x != 4; ++x) {
      mi[0] = i + x;
      isKnownTuple[x] = (0 <= distance(mi));
    }
    intExt(&xInterpFields[y], &xInterpIsKnown[y],
           fieldsTuple, isKnownTuple, continuousIndex[0], i);
  }

  //
  // Interpolate in the y direction.
  //
  bool isValueKnown;
  intExt(values, &isValueKnown,
         xInterpFields, xInterpIsKnown, continuousIndex[1], j);

  if (! isValueKnown) {
    *values = defaultValues;
  }
}

template<std::size_t M, typename T>
inline
void
makeFieldsTuple(std::array<std::array<T, M>, 4>* fieldsTuple,
                const container::MultiArrayConstRef < T, 3 + 1 > fields,
                container::MultiIndexTypes<3>::IndexList i)
{
  typedef container::MultiIndexTypes < 3 + 1 >::Index Index;
  container::MultiIndexTypes < 3 + 1 >::IndexList mi;
  mi[2] = i[1];
  mi[3] = i[2];
  for (Index x = 0; x != 4; ++x) {
    mi[1] = i[0] + x;
    for (mi[0] = 0; mi[0] != M; ++mi[0]) {
      (*fieldsTuple)[x][mi[0]] = fields(mi);
    }
  }
}

// 3-D
// OUTPUT:
//   values - The values of the interpolated fields.
// INPUT:
//   position - The Cartesian position at which to interpolate.
//   field - An array of values to be interpolated.
//   distance - The array of distances.
//   grid - Holds domain and grid information.
//   defaultValues - If no neigboring grid points are known, values is set
//     to defaultValues.
// TEMPLATE PARAMETERS
//   M is the number of fields.
//   T in the number type.
template<std::size_t M, typename T>
inline
void
intExt(std::array<T, M>* values,
       const std::array<T, 3>& position,
       const std::array<T, M>& defaultValues,
       const geom::RegularGrid<3, T>& grid,
       const container::MultiArrayConstRef<T, 3> distance,
       const container::MultiArrayConstRef < T, 3 + 1 > fields)
{
  // The single index type.
  typedef container::MultiIndexTypes<3>::Index Index;
  // Multi-index.
  typedef container::MultiIndexTypes<3>::IndexList IndexList;

  //
  // The position lies in a cuboid defined by 64 grid points.
  // First calculate the indices of the lower corner of this octet.
  //
  std::array<T, 3> continuousIndex = position;
  grid.convertLocationToIndex(&continuousIndex);
  const Index i = static_cast<Index>(std::floor(continuousIndex[0])) - 1;
  const Index j = static_cast<Index>(std::floor(continuousIndex[1])) - 1;
  const Index k = static_cast<Index>(std::floor(continuousIndex[2])) - 1;

  //
  // Only do the interpolation/extrapolation if the 64 grid points are
  // within this grid.
  //
  if (!(0 <= i && i <= Index(grid.getExtents()[0] - 4) &&
        0 <= j && j <= Index(grid.getExtents()[1] - 4) &&
        0 <= k && k <= Index(grid.getExtents()[2] - 4))) {
    return;
  }

  std::array<std::array<T, M>, 4> fieldsTuple;
  std::array<bool, 4> isKnownTuple;

  //
  // Interpolate in the x direction.
  //
  std::array<T, M> xInterpFields[4][4];
  bool xInterpIsKnown[4][4];
  Index x, y, z;
  IndexList mi;

  for (z = 0; z != 4; ++z) {
    for (y = 0; y != 4; ++y) {
      mi[0] = i;
      mi[1] = j + y;
      mi[2] = k + z;
      makeFieldsTuple(&fieldsTuple, fields, mi);
      for (x = 0; x != 4; ++x) {
        mi[0] = i + x;
        isKnownTuple[x] = (0 <= distance(mi));
      }
      intExt(&xInterpFields[y][z], &xInterpIsKnown[y][z],
             fieldsTuple, isKnownTuple, continuousIndex[0], i);
    }
  }

  //
  // Interpolate in the y direction.
  //
  std::array<std::array<T, M>, 4> yInterpFields;
  std::array<bool, 4> yInterpIsKnown;

  for (z = 0; z != 4; ++z) {
    for (y = 0; y != 4; ++y) {
      fieldsTuple[y] = xInterpFields[y][z];
      isKnownTuple[y] = xInterpIsKnown[y][z];
    }
    intExt(&yInterpFields[z], &yInterpIsKnown[z],
           fieldsTuple, isKnownTuple, continuousIndex[1], j);
  }

  //
  // Interpolate in the z direction.
  //
  bool isValueKnown;
  intExt(values, &isValueKnown,
         yInterpFields, yInterpIsKnown, continuousIndex[2], k);

  if (! isValueKnown) {
    *values = defaultValues;
  }
}


// Interplotation/extrapolation for an array of points.
template<std::size_t N, std::size_t M, typename T>
inline
void
gridInterpExtrap(std::vector<std::array<T, M> >* values,
                 const std::vector<std::array<T, N> >& positions,
                 const std::array<T, M>& defaultValues,
                 const geom::RegularGrid<N, T>& grid,
                 const container::MultiArrayConstRef<T, N> distance,
                 const container::MultiArrayConstRef < T, N + 1 > fields)
{
  assert(values->size() == positions.size());

  for (std::size_t i = 0; i != positions.size(); ++i) {
    intExt(&(*values)[i], positions[i], defaultValues, grid, distance, fields);
  }
}

template<std::size_t N, std::size_t M, typename T>
inline
void
gridInterpExtrap(std::vector<std::array<T, M> >* values,
                 const std::vector<std::array<T, N> >& positions,
                 const std::array<T, M>& defaultValues,
                 const std::array<std::size_t, N>& extents,
                 const std::array<T, N>& lowerCorner,
                 const std::array<T, N>& upperCorner,
                 const T* distance,
                 const T* fields)
{
  //
  // Wrap the arguments in the proper classes.
  //
  const geom::BBox<T, N> domain = {lowerCorner, upperCorner};
  const geom::RegularGrid<N, T> grid(extents, domain);
  const container::MultiArrayConstRef<T, N> _distance(distance, extents);
  std::array < std::size_t, N + 1 > fieldExtents;
  fieldExtents[0] = M;
  std::copy(extents.begin(), extents.end(), fieldExtents.begin() + 1);
  const container::MultiArrayConstRef < T, N + 1 > _fields(fields, fieldExtents);

  // Do the interpolation/extrapolation.
  gridInterpExtrap(values, positions, defaultValues, grid, _distance,
                   _fields);
}

} // namespace numerical
}
