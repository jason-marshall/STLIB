// -*- C++ -*-

#if !defined(__container_IndexRange_ipp__)
#error This file is an implementation detail of the class IndexRange.
#endif

namespace stlib
{
namespace container
{

//---------------------------------------------------------------------------
// Free functions.

// Return the intersection of the two ranges.
inline
IndexRange
overlap(const IndexRange& x, const IndexRange& y)
{
#ifdef STLIB_DEBUG
  assert(x.step() == 1 && y.step() == 1);
#endif
  return overlap(x.extent(), x.base(), y.extent(), y.base());
}

// Return the intersection of the two index ranges.
inline
IndexRange
overlap(const IndexRange::size_type extent1,
        const IndexRange::Index base1,
        const IndexRange::size_type extent2,
        const IndexRange::Index base2)
{
  typedef IndexRange::Index Index;

  const Index base = std::max(base1, base2);
  const Index upper = std::min(base1 + Index(extent1),
                               base2 + Index(extent2));
  const IndexRange::size_type extent = (upper > base ? upper - base : 0);
  return IndexRange(extent, base);
}

// Return true if the index is in the index range.
inline
bool
isIn(const IndexRange& range, const IndexRange::Index index)
{
  typedef IndexRange::Index Index;
  // If the range has unit steps.
  if (range.step() == 1) {
    if (!(range.base() <= index &&
          index < range.base() + Index(range.extent()))) {
      return false;
    }
  }
  // If the range does not have unit steps.
  else {
    if (!(range.base() <= index &&
          index < range.base() +
          range.step() * Index(range.extent()) &&
          (index - Index(range.base())) % range.step() == 0)) {
      return false;
    }
  }
  return true;
}

} // namespace container
}
