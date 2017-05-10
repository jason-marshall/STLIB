// -*- C++ -*-

#if !defined(__lorg_order_tcc__)
#error This file is an implementation detail of order.
#endif

namespace stlib
{
namespace lorg
{

template<typename _Integer>
inline
void
sort(std::vector<std::pair<_Integer, std::size_t> >* pairs, const int digits)
{
  // RciSort is faster than std::sort() using GNU 4.4.7 on an Intel
  // Xeon E5-2687W.
  RciSort<_Integer, std::size_t> rci(pairs, digits);
  rci.sort();
#if 0
  if (std::numeric_limits<_Integer>::digits > 32) {
    // Technically, it sorts using the composite number formed by the pair.
    // Since the first is most significant, this is fine.
    std::sort(pairs->begin(), pairs->end());
#if 0
    // Sorting by the first part is slower.
    LessThanFirst<std::pair<_Integer, std::size_t> > comp;
    std::sort(pairs->begin(), pairs->end(), comp);
#endif
  }
#endif
}

//! Order according to the code values.
template<typename _Integer>
inline
void
codeOrder(const std::vector<_Integer>& codes,
          std::vector<std::size_t>* ranked)
{
  // Make a vector of code/index pairs.
  std::vector<std::pair<_Integer, std::size_t> > pairs(codes.size());
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    pairs[i].first = codes[i];
    pairs[i].second = i;
  }
  // Sort the pairs by the codes.
  sort(&pairs);
  // Record the order.
  ranked->resize(codes.size());
  for (std::size_t i = 0; i != ranked->size(); ++i) {
    (*ranked)[i] = pairs[i].second;
  }
}

//! Order according to the code values.
template<typename _Integer>
inline
void
codeOrder(const std::vector<_Integer>& codes, std::vector<std::size_t>* ranked,
          std::vector<std::size_t>* mapping)
{
  codeOrder(codes, ranked);
  mapping->resize(ranked->size());
  for (std::size_t i = 0; i != mapping->size(); ++i) {
    (*mapping)[(*ranked)[i]] = i;
  }
}

inline
void
randomOrder(const std::size_t size, std::vector<std::size_t>* ranked)
{
  ranked->resize(size);
  for (std::size_t i = 0; i != ranked->size(); ++i) {
    (*ranked)[i] = i;
  }
  std::random_shuffle(ranked->begin(), ranked->end());
}

inline
void
randomOrder(const std::size_t size, std::vector<std::size_t>* ranked,
            std::vector<std::size_t>* mapping)
{
  randomOrder(size, ranked);
  mapping->resize(ranked->size());
  for (std::size_t i = 0; i != mapping->size(); ++i) {
    (*mapping)[(*ranked)[i]] = i;
  }
}

// Does not improve performance.
#if 0
template<typename _Pair>
struct
    LessThanFirst {
  bool
  operator()(const _Pair& x, const _Pair& y)
  {
    return x.first < y.first;
  }
};
#endif

template<typename _Integer, typename _Float, std::size_t _Dimension>
inline
void
mortonOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
            std::vector<std::size_t>* ranked)
{
  assert(! positions.empty());
  if (ranked->size() != positions.size()) {
    ranked->resize(positions.size());
  }

  // The functor for generating codes.
  Morton<_Integer, _Float, _Dimension> morton(positions);

  // Make a vector of code/index pairs.
  std::vector<std::pair<_Integer, std::size_t> > pairs(positions.size());
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    pairs[i].first = morton.code(positions[i]);
    pairs[i].second = i;
  }

  // Sort the pairs by the codes.
  sort(&pairs);

  // Record the order.
  for (std::size_t i = 0; i != ranked->size(); ++i) {
    (*ranked)[i] = pairs[i].second;
  }
}

template<typename _Integer, typename _Float, std::size_t _Dimension>
void
mortonOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
            std::vector<std::size_t>* ranked,
            std::vector<std::size_t>* mapping)
{
  mortonOrder<_Integer>(positions, ranked);
  mapping->resize(ranked->size());
  for (std::size_t i = 0; i != mapping->size(); ++i) {
    (*mapping)[(*ranked)[i]] = i;
  }
}

template<typename _Float, std::size_t _Dimension>
inline
void
axisOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
          std::vector<std::size_t>* ranked)
{
  assert(! positions.empty());
  if (ranked->size() != positions.size()) {
    ranked->resize(positions.size());
  }

  // Determine the dimension of greatest extent.
  // Calculate a bounding box around the positions.
  geom::BBox<_Float, _Dimension> const box =
    geom::specificBBox<geom::BBox<_Float, _Dimension> >(positions.begin(),
                                                        positions.end());
  // The extents of the box.
  const std::array<_Float, _Dimension> extents = box.upper - box.lower;
  // The dimension of greatest extent.
  std::size_t dim = 0;
  for (std::size_t i = 1; i != extents.size(); ++i) {
    if (extents[i] > extents[dim]) {
      dim = i;
    }
  }

  // Make a vector of coordinate/index pairs.
  std::vector<std::pair<_Float, std::size_t> > pairs(positions.size());
  for (std::size_t i = 0; i != pairs.size(); ++i) {
    pairs[i].first = positions[i][dim];
    pairs[i].second = i;
  }

  // Technically, it sorts using the composite number formed by the pair.
  // Since the first is most significant, this is fine.
  std::sort(pairs.begin(), pairs.end());

  // Record the order.
  for (std::size_t i = 0; i != ranked->size(); ++i) {
    (*ranked)[i] = pairs[i].second;
  }
}

template<typename _Float, std::size_t _Dimension>
void
axisOrder(const std::vector<std::array<_Float, _Dimension> >& positions,
          std::vector<std::size_t>* ranked,
          std::vector<std::size_t>* mapping)
{
  axisOrder(positions, ranked);
  mapping->resize(ranked->size());
  for (std::size_t i = 0; i != mapping->size(); ++i) {
    (*mapping)[(*ranked)[i]] = i;
  }
}

} // namespace lorg
}
