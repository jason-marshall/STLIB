// -*- C++ -*-

#if !defined(__geom_ScanConversionPolyhedron_ipp__)
#error This file is an implementation detail of the class ScanConversionPolyhedron.
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
ScanConversionPolyhedron<_Index, T>&
ScanConversionPolyhedron<_Index, T>::
operator=(const ScanConversionPolyhedron& other)
{
  // Avoid assignment to self
  if (&other != this) {
    _edges = other._edges;
  }
  // Return *this so assignments can chain
  return *this;
}


template<typename _Index, typename T>
inline
ScanConversionPolyhedron<_Index, T>&
ScanConversionPolyhedron<_Index, T>::
operator=(const IndexedEdgePolyhedron<Number>& x)
{
  // Clear any old edges.
  _edges.clear();

  // Add edges from the indexed edge polyhedron.
  for (std::size_t n = 0; n != x.getEdgesSize(); ++n) {
    insertEdge(x.getEdgeSource(n), x.getEdgeTarget(n));
  }

  // Return *this so assignments can chain
  return *this;
}


//
// Mathematical Member Functions
//


template<typename _Index, typename T>
inline
void
ScanConversionPolyhedron<_Index, T>::
computeBBox(BBox<Number, 3>* bb) const
{
  typename std::vector<Segment>::const_iterator iter = _edges.begin();
  // if this polyhedron has any edges.
  if (iter != _edges.end()) {
#if 0
    // For no good reason this fails with a segmentation fault on 32-bit
    // GCC 4.2.
    *bb = makeBBox((*iter).getSource(), (*iter).getTarget());
#else
    const BBox<Number, 3> box = {(*iter).getSource(), (*iter).getTarget()};
    *bb = box;
#endif
    ++iter;
    for (; iter != _edges.end(); ++iter) {
      *bb += (*iter).getSource();
      *bb += (*iter).getTarget();
    }
  }
  else {
    bb->lower = ext::filled_array<Point>(0);
    bb->upper = ext::filled_array<Point>(-1);
  }
}


// Scan convert the polyhedron.
template<typename _Index, typename T>
template<typename IndexOutputIterator>
inline
void
ScanConversionPolyhedron<_Index, T>::
scanConvert(IndexOutputIterator coordinates,
            const RegularGrid<3, Number>& grid) const
{
  static Polygon poly;

  // Find the top and bottom.
  typename std::vector<Segment>::const_iterator iter;
  Number bottom = grid.getExtents()[2] - 1;
  Number top = 0;
  for (iter = _edges.begin(); iter != _edges.end(); ++iter) {
    if ((*iter).getSource()[2] < bottom) {
      bottom = (*iter).getSource()[2];
    }
    if ((*iter).getSource()[2] > top) {
      top = (*iter).getSource()[2];
    }
    if ((*iter).getTarget()[2] < bottom) {
      bottom = (*iter).getTarget()[2];
    }
    if ((*iter).getTarget()[2] > top) {
      top = (*iter).getTarget()[2];
    }
  }

  // Scan convert each z slice.
  const Index end = std::min(Index(top), Index(grid.getExtents()[2]) - 1);
  for (Index k = (bottom > 0 ? Index(bottom) + 1 : 0); k <= end; ++k) {
    // Intersect the polyhedron with the plane to get a polygon.
    computeZIntersection(&poly, k);
    // If the polygon is not degenerate.
    if (poly.getVerticesSize() >= 3) {
      // Put the vertices of the polygon in positive order.
      poly.orderVertices();
      // Scan convert the polygon.
      poly.scanConvert(coordinates, grid.getExtents(), k);
    }
  }
}


//
// Manipulators
//


template<typename _Index, typename T>
inline
void
ScanConversionPolyhedron<_Index, T>::
insertEdge(const Point& p, const Point& q)
{
  static Segment seg;

  if (p[2] == q[2]) {
    return;
  }

  if (p[2] < q[2]) {
    seg.make(p, q);
  }
  else if (q[2] < p[2]) {
    seg.make(q, p);
  }

  _edges.push_back(seg);
}


//
// File I/O
//


// CONTINUE: Implement this with a put() member function.
template<typename _Index, typename T>
inline
std::ostream&
operator<<(std::ostream& out,
           const ScanConversionPolyhedron<_Index, T>& polyhedron)
{
  out << std::setiosflags(std::ios::fixed);
  for (std::size_t n = 0; n != polyhedron.getEdges().size(); ++n) {
    out << "Segment number " << n << "\n"
        << polyhedron.getEdges()[n] << "\n";
  }
  return out;
}


template<typename _Index, typename T>
inline
void
mathematicaPrint(std::ostream& out,
                 const ScanConversionPolyhedron<_Index, T>& polyhedron)
{
  typedef typename ScanConversionPolyhedron<_Index, T>::Segment Segment;

  out << std::setiosflags(std::ios::fixed);
  out << "{";
  const std::size_t size = polyhedron.getEdges().size();
  for (std::size_t n = 0; n != size; ++n) {
    const Segment& s = polyhedron.getEdges()[n];
    out << "Line[{{"
        << s.getSource()[0] << ","
        << s.getSource()[1] << ","
        << s.getSource()[2] << "},{"
        << s.getTarget()[0] << ","
        << s.getTarget()[1] << ","
        << s.getTarget()[2] << "}}]";
    if (n != size - 1) {
      out << "," << "\n";
    }
    else {
      out << "}" << "\n";
    }
  }
}


//! Read as a list of edges.
/*! \relates ScanConversionPolyhedron */
template<typename _Index, typename T>
inline
std::istream&
operator>>(std::istream& in, ScanConversionPolyhedron<_Index, T>& polyhedron)
{
  // Clear any old edges.
  polyhedron.clear();
  // Get the number of edges.
  std::size_t size;
  in >> size;
  // Get each edge.
  typename ScanConversionPolyhedron<_Index, T>::Point source, target;
  while (size-- != 0) {
    in >> source >> target;
    polyhedron.insertEdge(source, target);
  }
  return in;
}


//
// Private Member Functions
//


template<typename _Index, typename T>
inline
void
ScanConversionPolyhedron<_Index, T>::
computeZIntersection(Polygon* poly, const Number z) const
{
  Number x, y;
  typename Polygon::Point p2;
  poly->clear();

  typename std::vector<Segment>::const_iterator iter;
  for (iter = _edges.begin(); iter != _edges.end(); ++iter) {
    if (stlib::geom::computeZIntersection(*iter, &x, &y, z)) {
      p2[0] = x;
      p2[1] = y;
      poly->insert(p2);
    }
  }
}

} // namespace geom
}
