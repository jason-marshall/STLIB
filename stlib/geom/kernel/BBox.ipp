// -*- C++ -*-

#if !defined(__geom_BBox_ipp__)
#error This file is an implementation detail of the class BBox.
#endif

namespace stlib
{
namespace geom
{

//
// Member functions
//


template<typename Float, std::size_t D>
inline
bool
isEmpty(BBox<Float, D> const& box)
{
  // Bounding boxes in 0-D are empty by definition.
  if (D == 0) {
    return true;
  }
  for (std::size_t i = 0; i != D; ++i) {
    if (box.lower[i] > box.upper[i]) {
      return true;
    }
  }
  return false;
}


template<typename Float, std::size_t D>
inline
BBox<Float, D>&
operator+=(BBox<Float, D>& box, std::array<Float, D> const& p)
{
  for (std::size_t i = 0; i != D; ++i) {
    if (p[i] < box.lower[i]) {
      box.lower[i] = p[i];
    }
    if (p[i] > box.upper[i]) {
      box.upper[i] = p[i];
    }
  }
  return box;
}


template<typename Float, std::size_t D>
inline
BBox<Float, D>&
operator+=(BBox<Float, D>& box, BBox<Float, D> const& rhs)
{
  if (isEmpty(rhs)) {
    return box;
  }
  for (std::size_t i = 0; i != D; ++i) {
    if (rhs.lower[i] < box.lower[i]) {
      box.lower[i] = rhs.lower[i];
    }
    if (rhs.upper[i] > box.upper[i]) {
      box.upper[i] = rhs.upper[i];
    }
  }
  return box;
}


template<typename BBoxT>
inline
BBoxT
_specificBBox(std::false_type /*hasInfinity*/)
{
  using Point = typename BBoxT::Point;
  using Float = typename BBoxT::Float;
  return BBoxT{ext::filled_array<Point>
      (std::numeric_limits<Float>::max()),
      ext::filled_array<Point>
      (std::numeric_limits<Float>::min())};
}


template<typename BBoxT>
inline
BBoxT
_specificBBox(std::true_type /*hasInfinity*/)
{
  using Point = typename BBoxT::Point;
  using Float = typename BBoxT::Float;
  return BBoxT{ext::filled_array<Point>
      (std::numeric_limits<Float>::infinity()),
      -ext::filled_array<Point>
      (std::numeric_limits<Float>::infinity())};
}


template<typename BBoxT>
inline
BBoxT
specificBBox()
{
  return _specificBBox<BBoxT>(std::integral_constant<bool,
                              std::numeric_limits<typename BBoxT::Float>::
                              has_infinity>{});
}


template<typename BBoxT, typename InputIterator>
inline
BBoxT
specificBBox(InputIterator begin, InputIterator end)
{
  // If there are no objects, make the box empty.
  if (begin == end) {
    return specificBBox<BBoxT>();
  }
  // Bound the first object.
  BBoxT result = specificBBox<BBoxT>(*begin++);
  // Add the rest of the objects.
  while (begin != end) {
    result += specificBBox<BBoxT>(*begin++);
  }
  return result;
}


template<typename BBoxT, typename InputIterator, typename UnaryFunction>
inline
BBoxT
specificBBox(InputIterator begin, InputIterator end, UnaryFunction boundable)
{
  // If there are no objects, make the box empty.
  if (begin == end) {
    return specificBBox<BBoxT>();
  }
  // Bound the first object.
  BBoxT result = specificBBox<BBoxT>(boundable(*begin++));
  // Add the rest of the objects.
  while (begin != end) {
    result += specificBBox<BBoxT>(boundable(*begin++));
  }
  return result;
}


template<typename BBoxT, typename _ForwardIterator>
inline
std::vector<BBoxT>
specificBBoxForEach(_ForwardIterator begin, _ForwardIterator end)
{
  std::vector<BBoxT> result(std::distance(begin, end));
  for (std::size_t i = 0; i != result.size(); ++i) {
    result[i] = specificBBox<BBoxT>(*begin++);
  }
  return result;
}


template<typename Float, typename Float2, std::size_t D>
struct BBoxForObject<Float, BBox<Float2, D> >
{
  std::size_t static constexpr Dimension = D;
  using DefaultBBox = BBox<Float2, D>;

  static
  BBox<Float, D>
  create(BBox<Float2, D> const& x)
  {
    return BBox<Float, D>{ext::ConvertArray<Float>::convert(x.lower),
        ext::ConvertArray<Float>::convert(x.upper)};
  }
};

/// Trivially convert a bounding box to one of the same type.
/** This function differs from the one in which a bounding box is converted
    to one with a different number type in that here we return a constant
    reference to the argument. Thus, we avoid constructing a bounding box. */
template<typename Float, std::size_t D>
struct BBoxForObject<Float, BBox<Float, D> >
{
  std::size_t static constexpr Dimension = D;
  using DefaultBBox = BBox<Float, D>;

  static
  BBox<Float, D> const&
  create(BBox<Float, D> const& x)
  {
    return x;
  }
};

template<typename Float, typename Float2, std::size_t D>
struct BBoxForObject<Float, std::array<Float2, D> >
{
  std::size_t static constexpr Dimension = D;
  using DefaultBBox = BBox<Float2, D>;

  static
  BBox<Float, D>
  create(std::array<Float2, D> const& x)
  {
    return BBox<Float, D>{ext::ConvertArray<Float>::convert(x),
        ext::ConvertArray<Float>::convert(x)};
  }
};

template<typename Float, typename Float2, std::size_t D, std::size_t N>
struct BBoxForObject<Float, std::array<std::array<Float2, D>, N> >
{
  std::size_t static constexpr Dimension = D;
  using DefaultBBox = BBox<Float2, D>;

  static
  BBox<Float, D>
  create(std::array<std::array<Float2, D>, N> const& x)
  {
    static_assert(N != 0, "Error.");
    BBox<Float, D> box = {ext::ConvertArray<Float>::convert(x[0]),
                            ext::ConvertArray<Float>::convert(x[0])};
    for (std::size_t i = 1; i != N; ++i) {
      box += ext::ConvertArray<Float>::convert(x[i]);
    }
    return box;
  }
};

template<typename Float, typename _Geometric, typename Data>
struct BBoxForObject<Float, std::pair<_Geometric, Data> >
{
  std::size_t static constexpr Dimension =
    BBoxForObject<Float, _Geometric>::Dimension;
  using DefaultBBox = typename BBoxForObject<Float, _Geometric>::DefaultBBox;

  static
  BBox<Float, Dimension>
  create(std::pair<_Geometric, Data> const& x)
  {
    return specificBBox<BBox<Float, Dimension> >(x.first);
  }
};


//
// Mathematical free functions
//


template<typename Float, std::size_t D, typename _Object>
inline
bool
isInside(BBox<Float, D> const& box, _Object const& x)
{
  // Check if the bounding box around the object is contained in this
  // bounding box.
  return isInside(box, specificBBox<BBox<Float, D> >(x));
}


template<typename Float, std::size_t D>
inline
bool
isInside(BBox<Float, D> const& box, std::array<Float, D> const& p)
{
  for (std::size_t i = 0; i != D; ++i) {
    if (p[i] < box.lower[i] || box.upper[i] < p[i]) {
      return false;
    }
  }
  return true;
}


template<typename Float, std::size_t D>
inline
bool
doOverlap(BBox<Float, D> const& a, BBox<Float, D> const& b)
{
  for (std::size_t i = 0; i != D; ++i) {
    if (std::max(a.lower[i], b.lower[i]) >
        std::min(a.upper[i], b.upper[i])) {
      return false;
    }
  }
  return true;
}


// Return the squared distance between two 1-D intervals.
template<typename Float>
inline
Float
squaredDistanceBetweenIntervals(Float const lower1, Float const upper1,
                                Float const lower2, Float const upper2)
{
  // Consider the intervals [l1..u1] and [l2..u2].
  // l1 u1 l2 u2
  if (upper1 < lower2) {
    return (upper1 - lower2) * (upper1 - lower2);
  }
  // l2 u2 l1 u2
  if (upper2 < lower1) {
    return (upper2 - lower1) * (upper2 - lower1);
  }
  return 0;
}


// Return the squared distance between two bounding boxes.
template<typename Float, std::size_t D>
inline
Float
squaredDistance(BBox<Float, D> const& x, BBox<Float, D> const& y)
{
  Float d2 = 0;
  for (std::size_t i = 0; i != D; ++i) {
    d2 += squaredDistanceBetweenIntervals(x.lower[i], x.upper[i],
                                          y.lower[i], y.upper[i]);
  }
  return d2;
}


// CONTINUE: Try to get rid of floor and ceil.
template<typename Index, typename MultiIndexOutputIterator>
inline
void
scanConvert(MultiIndexOutputIterator indices, BBox<double, 3> const& box)
{
  // Make the index bounding box.
  BBox<Index, 3> ib = {{{
        Index(std::ceil(box.lower[0])),
        Index(std::ceil(box.lower[1])),
        Index(std::ceil(box.lower[2]))
      }
    },
    { {
        Index(std::floor(box.upper[0])),
        Index(std::floor(box.upper[1])),
        Index(std::floor(box.upper[2]))
      }
    }
  };

  // Scan convert the integer bounding box.
  scanConvertIndex(indices, ib);
}


template<typename MultiIndexOutputIterator, typename Index>
inline
void
scanConvertIndex(MultiIndexOutputIterator indices, BBox<Index, 3> const& box)
{

  Index const iStart = box.lower[0];
  Index const iEnd = box.upper[0];
  Index const jStart = box.lower[1];
  Index const jEnd = box.upper[1];
  Index const kStart = box.lower[2];
  Index const kEnd = box.upper[2];

  std::array<Index, 3> index;
  for (index[2] = kStart; index[2] <= kEnd; ++index[2]) {
    for (index[1] = jStart; index[1] <= jEnd; ++index[1]) {
      for (index[0] = iStart; index[0] <= iEnd; ++index[0]) {
        *indices++ = index;
      }
    }
  }
}


template<typename MultiIndexOutputIterator, typename Index>
inline
void
scanConvert(MultiIndexOutputIterator indices, BBox<double, 3> const& box,
            BBox<Index, 3> const& domain)
{
  // Make the integer bounding box.
  BBox<Index, 3> ib = {{{
        Index(std::ceil(box.lower[0])),
        Index(std::ceil(box.lower[1])),
        Index(std::ceil(box.lower[2]))
      }
    },
    { {
        Index(std::floor(box.upper[0])),
        Index(std::floor(box.upper[1])),
        Index(std::floor(box.upper[2]))
      }
    }
  };

  // Scan convert the integer bounding box on the specified domain.
  scanConvertIndex(indices, ib, domain);
}


template<typename MultiIndexOutputIterator, typename Index>
inline
void
scanConvertIndex(MultiIndexOutputIterator indices, BBox<Index, 3> const& box,
                 BBox<Index, 3> const& domain)
{
  BBox<Index, 3> inter = intersection(box, domain);
  if (! isEmpty(inter)) {
    scanConvertIndex(indices, inter);
  }
}


} // namespace geom
} // namespace stlib
