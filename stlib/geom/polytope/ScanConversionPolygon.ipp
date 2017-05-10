// -*- C++ -*-

#if !defined(__geom_ScanConversionPolygon_ipp__)
#error This file is an implementation detail of the class ScanConversionPolygon.
#endif

namespace stlib
{
namespace geom
{


//
// Constructors and destructor.
//


template<typename _Index, typename T>
inline
ScanConversionPolygon<_Index, T>::
ScanConversionPolygon(const std::size_t size) :
  _vertices()
{
  _vertices.reserve(size);
}


template<typename _Index, typename T>
inline
ScanConversionPolygon<_Index, T>::
ScanConversionPolygon(const ScanConversionPolygon& other) :
  _vertices(other._vertices) {}


template<typename _Index, typename T>
inline
ScanConversionPolygon<_Index, T>&
ScanConversionPolygon<_Index, T>::
operator=(const ScanConversionPolygon& other)
{
  // Avoid assignment to self
  if (&other != this) {
    _vertices = other._vertices;
  }
  // Return *this so assignments can chain
  return *this;
}


//
// Mathematical operations
//


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
orderVertices()
{
  // If the polygon is not degenerate.
  if (_vertices.size() >= 3) {

    // Find the vertex with minimum y coordinate and put it in the
    // first position.
    Point temp;
    Iterator min = _vertices.begin();
    Iterator iter = _vertices.begin() + 1;
    for (; iter != _vertices.end(); ++iter) {
      if ((*iter)[1] < (*min)[1]) {
        min = iter;
      }
    }
    if (min != _vertices.begin()) {
      temp = *_vertices.begin();
      *_vertices.begin() = *min;
      *min = temp;
    }

    // Calculate the pseudo-angle from the bottom point.
    std::vector<Number> angle(_vertices.size());
    angle.clear();
    const Point bottom = *_vertices.begin();
    for (iter = _vertices.begin(); iter != _vertices.end(); ++iter) {
      temp = *iter - bottom;
      angle.push_back(computePseudoAngle(temp));
    }

    // Sort the vertices by angle.
    Index i, j;
    Number angleTemp;
    for (i = Index(_vertices.size()) - 1; i >= 0; --i) {
      for (j = 1; j <= i; ++j) {
        if (angle[j - 1] > angle[j]) {
          temp = _vertices[j - 1];
          _vertices[j - 1] = _vertices[j];
          _vertices[j] = temp;
          angleTemp = angle[j - 1];
          angle[j - 1] = angle[j];
          angle[j] = angleTemp;
        }
      }
    }
  }
}


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
removeDuplicates()
{
  // The square of the floating point precision.
  static Number eps =
    std::pow(10 * std::numeric_limits<Number>::epsilon(), 2);

  // If there is more than one vertex.
  if (_vertices.size() > 1) {
    Iterator prev = _vertices.begin();
    Iterator iter = _vertices.begin() + 1;
    while (iter != _vertices.end()) {
      if (ext::squaredDistance(*iter, *prev) < eps) {
        _vertices.erase(iter);
        iter = prev + 1;
      }
      else {
        ++prev;
        ++iter;
      }
    }
  }
}


template<typename _Index, typename T>
inline
std::size_t
ScanConversionPolygon<_Index, T>::
computeBottomAndTop(Number* bottom, Number* top) const
{
  // Sanity check: The polygon should not be degenerate.
  assert(_vertices.size() >= 3);

  std::size_t minIndex = 0;
  *bottom = *top = _vertices[0][1];
  Number y;
  for (std::size_t i = 1; i < _vertices.size(); ++i) {
    if ((y = _vertices[i][1]) < *bottom) {
      minIndex = i;
      *bottom = y;
    }
    else if (y > *top) {
      *top = y;
    }
  }
  return minIndex;
}


template<typename _Index, typename T>
template<typename IndexOutputIterator, std::size_t N>
inline
void
ScanConversionPolygon<_Index, T>::
scanConvertTriangle(IndexOutputIterator coordinates,
                    const std::array<std::size_t, N>& extents,
                    std::array<Index, N> multiIndex) const
{
  assert(_vertices.size() == 3);

  //
  // Determine the bottom, left and right vertices.
  //
  Index bottomIndex, rightIndex, leftIndex;
  if (_vertices[0][1] < _vertices[1][1]) {
    if (_vertices[0][1] < _vertices[2][1]) {
      bottomIndex = 0;
      rightIndex = 1;
      leftIndex = 2;
    }
    else {
      bottomIndex = 2;
      rightIndex = 0;
      leftIndex = 1;
    }
  }
  else {
    if (_vertices[1][1] < _vertices[2][1]) {
      bottomIndex = 1;
      rightIndex = 2;
      leftIndex = 0;
    }
    else {
      bottomIndex = 2;
      rightIndex = 0;
      leftIndex = 1;
    }
  }
  const Point& bottomVertex = _vertices[bottomIndex];
  const Point& rightVertex = _vertices[rightIndex];
  const Point& leftVertex = _vertices[leftIndex];

  //
  // The scan conversion proceeds in two stages.
  // Do the first stage.
  //

  // Get the starting row.
  Index row = bottomVertex[1] > 0 ? Index(bottomVertex[1]) + 1 : 0;
  // Get the ending row.
  Index topRow = std::min(Index(std::floor(std::min(rightVertex[1],
                                leftVertex[1]))),
                          Index(extents[1]) - 1);

  Number leftDxDy, leftIntersection;
  Number rightDxDy, rightIntersection;

  // Find the intersection of the left edge with the first row.
  Number dy = leftVertex[1] - bottomVertex[1];
  if (dy > 1e-5) {
    leftDxDy = (leftVertex[0] - bottomVertex[0]) / dy;
    leftIntersection = (bottomVertex[0]
                        + (row - bottomVertex[1]) * leftDxDy);
  }
  else {
    leftDxDy = 0;
    leftIntersection = std::min(bottomVertex[0], leftVertex[0]);
  }

  // Find the intersection of the right edge with the first row.
  dy = rightVertex[1] - bottomVertex[1];
  if (dy > 1e-5) {
    rightDxDy = (rightVertex[0] - bottomVertex[0]) / dy;
    rightIntersection = (bottomVertex[0]
                         + (row - bottomVertex[1]) * rightDxDy);
  }
  else {
    rightDxDy = 0;
    rightIntersection = std::min(bottomVertex[0], rightVertex[0]);
  }

  // Loop until all rows in the first stage have been scanned.
  while (row <= topRow) {
    // Scan convert the row.
    const Index end =
      std::min(rightIntersection > 0 ? Index(rightIntersection) : Index(-1),
               Index(extents[0]) - 1);
    for (Index col = leftIntersection > 0 ? Index(leftIntersection) + 1 :
                     Index(0); col <= end; ++col) {
      multiIndex[0] = col;
      multiIndex[1] = row;
      *coordinates++ = multiIndex;
    }

    // Increment the row.
    ++row;
    // Adjust the left and right intersections.
    leftIntersection += leftDxDy;
    rightIntersection += rightDxDy;
  }

  //
  // Do the second stage of the scan conversion.
  //

  // Get the ending row.
  topRow = std::min(Index(std::floor(std::max(rightVertex[1], leftVertex[1]))),
                    Index(extents[1]) - 1);

  // If this row passes through the triangle.
  if (row <= topRow) {

    if (leftVertex[1] < rightVertex[1]) {
      // Find the intersection of the left edge with this row.
      Number dy = rightVertex[1] - leftVertex[1];
      if (dy > 1e-5) {
        leftDxDy = (rightVertex[0] - leftVertex[0]) / dy;
        leftIntersection = (leftVertex[0]
                            + (row - leftVertex[1]) * leftDxDy);
      }
      else {
        leftDxDy = 0;
        leftIntersection = std::min(leftVertex[0], rightVertex[0]);
      }
    }
    else {
      // Find the intersection of the right edge with this row.
      Number dy = leftVertex[1] - rightVertex[1];
      if (dy > 1e-5) {
        rightDxDy = (leftVertex[0] - rightVertex[0]) / dy;
        rightIntersection = (rightVertex[0]
                             + (row - rightVertex[1]) * rightDxDy);
      }
      else {
        rightDxDy = 0;
        rightIntersection = std::min(rightVertex[0], leftVertex[0]);
      }
    }
    // Loop until all rows in the second stage have been scanned.
    while (row <= topRow) {
      // Scan convert the row.
      const Index end =
        std::min(rightIntersection > 0 ? Index(rightIntersection) : Index(-1),
                 Index(extents[0]) - 1);
      for (Index col = leftIntersection > 0 ? Index(leftIntersection) + 1 :
                       Index(0); col <= end; ++col) {
        multiIndex[0] = col;
        multiIndex[1] = row;
        *coordinates++ = multiIndex;
      }

      // Increment the row.
      ++row;
      // Adjust the left and right intersections.
      if (row <= topRow) {
        leftIntersection += leftDxDy;
        rightIntersection += rightDxDy;
      }
    }
  }
}


template<typename _Index, typename T>
template<typename IndexOutputIterator, std::size_t N>
inline
void
ScanConversionPolygon<_Index, T>::
scanConvert(IndexOutputIterator coordinates,
            const std::array<std::size_t, N>& extents,
            std::array<Index, N> multiIndex) const
{
  // If the polygon is degenerate, do nothing.
  if (_vertices.size() < 3) {
    return;
  }

  // Special case of a triangle.
  if (_vertices.size() == 3) {
    scanConvertTriangle(coordinates, extents, multiIndex);
    return;
  }

  Number bottomY, topY;
  // Get bottom vertex.
  Index bottom = computeBottomAndTop(&bottomY, &topY);

  // Get the starting row.
  Index row = bottomY > 0 ? Index(bottomY) + 1 : 0;
  // Get the ending row.
  Index topRow;
  if (topY < 0) {
    topRow = -1;
  }
  else {
    topRow = std::min(Index(topY), Index(extents[1]) - 1);
  }
  // The indices that track the left and right Line segments.
  CyclicIndex<Index>
  leftBottom(Index(_vertices.size())),
             leftTop(Index(_vertices.size())),
             rightBottom(Index(_vertices.size())),
             rightTop(Index(_vertices.size()));
  leftBottom.set(bottom);
  rightBottom.set(bottom);
  leftTop.set(bottom);
  --leftTop;
  rightTop.set(bottom);
  ++rightTop;
  bool newLeftEdge = true;
  bool newRightEdge = true;

  Number leftDxDy = 0,
         leftIntersection = 0,
         rightDxDy = 0,
         rightIntersection = 0;

  while (row <= topRow) { // loop until all rows have been scanned.

    // CONTINUE
#if 0
    std::cerr << "row = " << row << " topRow = " << topRow << "\n";
    if (row == 0 && topRow == 0) {
      put(std::cerr);
    }
#endif

    // Get the left edge for this row
    // Loop until we get an edge that crosses the row.  Skip horizontal edges.
    while (_vertices[leftTop()][1] < row ||
           _vertices[leftTop()][1] <= _vertices[leftBottom()][1]) {
      --leftTop;
      --leftBottom;
      newLeftEdge = true;
    }

    //std::cerr << "Get the right edge for this row.\n";
    // Get the right edge for this row
    while (_vertices[rightTop()][1] < row ||
           _vertices[rightTop()][1] <= _vertices[rightBottom()][1]) {
      ++rightTop;
      ++rightBottom;
      newRightEdge = true;
    }

    // Find the intersection of the left edge with this row
    if (newLeftEdge) {
      Point p = _vertices[leftBottom()];
      Point q = _vertices[leftTop()];
      Number dy = q[1] - p[1];
      if (dy > 1e-5) {
        leftDxDy = (q[0] - p[0]) / dy;
        leftIntersection = p[0] + (row - p[1]) * leftDxDy;
      }
      else {
        leftDxDy = 0;
        leftIntersection = std::min(p[0], q[0]);
      }
      newLeftEdge = false;
    }
    else {
      leftIntersection += leftDxDy;
    }

    // Find the intersection of the right edge with this row
    if (newRightEdge) {
      Point p = _vertices[rightBottom()];
      Point q = _vertices[rightTop()];
      Number dy = q[1] - p[1];
      if (dy > 1.0e-5) {
        rightDxDy = (q[0] - p[0]) / dy;
        rightIntersection = p[0] + (row - p[1]) * rightDxDy;
      }
      else {
        rightDxDy = 0;
        rightIntersection = std::max(p[0], q[0]);
      }
      newRightEdge = false;
    }
    else {
      rightIntersection += rightDxDy;
    }

    //std::cerr << "Scan convert the row.\n";
    // Scan convert the row.
    const Index end =
      std::min(rightIntersection > 0 ? Index(rightIntersection) : Index(-1),
               Index(extents[0]) - 1);
    for (Index col = leftIntersection > 0 ? Index(leftIntersection) + 1 :
                     Index(0); col <= end; ++col) {
      multiIndex[0] = col;
      multiIndex[1] = row;
      *coordinates++ = multiIndex;
    }

    // Increment the row.
    ++row;
  }
}


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
clip(const Line_2<Number>& line)
{
  // Initially indicate that there are no points above or below the Line.
  Index above = -1, below = -1;

  // Calculate the distance of the vertices from the line.
  std::vector<Number> dist;
  dist.reserve(_vertices.size());
  for (std::size_t i = 0; i < _vertices.size(); i++) {
    dist[i] = line.computeSignedDistance(_vertices[i]);
  }

  // Try to find a vertex above and below the Line.
  for (std::size_t i = 0; i < _vertices.size()
       && (above == -1 || below == -1); i++) {
    if (above == -1 && dist[i] > 0) {
      above = i;
    }
    if (below == -1 && dist[i] < 0) {
      below = i;
    }
  }

  // If there are no points below the line do nothing.
  if (below == -1) {
    return;
  }

  // If there are no points above the line the polygon is annihilated
  if (above == -1) {
    _vertices.clear();
    return;
  }

  // There are points above and below the line.  Find the points of
  // transition and clip the polygon.
  CyclicIndex<Index> left(Index(_vertices.size()));
  CyclicIndex<Index> right(Index(_vertices.size()));
  left.set(above);
  right.set(above);

  // Find the transition on one side.
  for (++left; dist[left()] > 0; ++left)
    ;
  Index leftBelow = left();
  --left;
  Index leftAbove = left();

  // Find the point of intersection.
  assert(dist[leftAbove] > 0 && dist[leftBelow] <= 0);
  Point leftInt;
  line.computeIntersection(_vertices[leftBelow], _vertices[leftAbove],
                           &leftInt);

  // Find the transition on the other side.
  for (--right; dist[right()] > 0; --right)
    ;
  Index rightBelow = right();
  ++right;
  Index rightAbove = right();

  // Find the point of intersection.
  assert(dist[rightAbove] > 0.0 && dist[rightBelow] <= 0.0);
  Point rightInt;
  line.computeIntersection(_vertices[rightBelow], _vertices[rightAbove],
                           &rightInt);

  //
  // Make the new polygon.
  //

  // Copy the old vertices.
  Container oldVertices(_vertices);
  // Erase the vertices of this polygon.
  _vertices.clear();
  // Add the vertices above the line.
  for (right.set(rightAbove); right() != leftBelow; ++right) {
    _vertices.push_back(oldVertices[right()]);
  }
  // Add the intersection points.
  if (oldVertices[leftAbove] != leftInt &&
      oldVertices[rightAbove] != leftInt) {
    _vertices.push_back(leftInt);
  }
  if (oldVertices[leftAbove] != rightInt &&
      oldVertices[rightAbove] != rightInt &&
      leftInt != rightInt) {
    _vertices.push_back(rightInt);
  }
}


template<typename _Index, typename T>
inline
bool
ScanConversionPolygon<_Index, T>::
isValid() const
{
  const std::size_t size = _vertices.size();
  if (size < 3) {
    return false;
  }
  for (std::size_t i = 0; i < size; i++) {
    if (_vertices[i] == _vertices[(i + 1) % size]) {
      return false;
    }
  }
  return true;
}


//
// Equality
//


template<typename _Index, typename T>
inline
bool
operator==(const ScanConversionPolygon<_Index, T>& a,
           const ScanConversionPolygon<_Index, T>& b)
{
  // Check that a and b have the same number of vertices.
  if (a.getVerticesSize() != b.getVerticesSize()) {
    return false;
  }

  // Check each vertex.
  for (std::size_t i = 0; i < a.getVerticesSize(); ++i) {
    if (a.getVertex(i) != b.getVertex(i)) {
      return false;
    }
  }
  return true;
}


//
// File I/O
//


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
get(std::istream& in)
{
  // Clear the vertices.
  _vertices.clear();
  // Get the number of vertices.
  std::size_t nv;
  in >> nv;
  // Get the vertices.
  Point p;
  for (; nv > 0; --nv) {
    in >> p;
    _vertices.push_back(p);
  }
}


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
put(std::ostream& out) const
{
  ConstIterator iter;
  for (iter = _vertices.begin(); iter != _vertices.end(); ++iter) {
    out << *iter << '\n';
  }
}


template<typename _Index, typename T>
inline
void
ScanConversionPolygon<_Index, T>::
mathematicaPrint(std::ostream& out) const
{
  out << "Line[{";
  ConstIterator iter;
  for (iter = _vertices.begin(); iter != _vertices.end(); ++iter) {
    out << "{" << (*iter)[0] << "," << (*iter)[1] << "},";
  }
  iter = _vertices.begin();
  out << "{" << (*iter)[0] << "," << (*iter)[1] << "}}],"
      << '\n';
}


} // namespace geom
}
