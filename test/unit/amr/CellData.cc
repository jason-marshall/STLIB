// -*- C++ -*-

#include "stlib/amr/CellData.h"
#include "stlib/amr/Traits.h"

#include <iostream>

USING_STLIB_EXT_ARRAY;
using namespace stlib;

// CONTINUE REMOVE
#if 0
template<typename _T, std::size_t _Dimension>
inline
bool
isIndexIn(const container::MultiArray<_T, _Dimension>& array,
          const typename container::MultiArray<_T, _Dimension>::IndexList& index)
{
  for (std::size_t d = 0; d != _Dimension; ++d) {
    if (!(array.bases()[d] <= index[d] &&
          index[d] < array.bases()[d] + array.extents()[d])) {
      return false;
    }
  }
  return true;
}
#endif

template < std::size_t Dimension, std::size_t MaximumLevel,
           std::size_t GhostWidth >
inline
void
test()
{
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel
            << ", GhostWidth = " << GhostWidth << "\n";
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef typename Traits::SpatialIndex SpatialIndex;

  typedef amr::CellData < Traits, 1 /*Depth*/, GhostWidth > CellData;
  typedef typename CellData::FieldTuple FieldTuple;
  typedef typename CellData::Array Array;
  typedef typename CellData::ArrayView ArrayView;
  typedef typename CellData::SizeList SizeList;
  typedef typename Array::const_iterator const_iterator;
  typedef typename Array::Index Index;
  typedef container::MultiIndexRange<Dimension> Range;
  typedef container::MultiIndexRangeIterator<Dimension> MultiIndexRangeIterator;

  // CONTINUE REMOVE
#if 0
  // Default constructor.
  {
    CellData x;
    assert(x.getArray().size() == 0);
  }
#endif
  // Initialize from the extents.
  SpatialIndex spatialIndex;
  const SizeList arrayExtents = ext::filled_array<SizeList>(6);
  const FieldTuple initialValue = ext::filled_array<FieldTuple>(7.);
  CellData x(spatialIndex, arrayExtents, initialValue);
  assert(x.getArray().extents() == arrayExtents + 2 * GhostWidth);
  assert(x.getInteriorArray().extents() == arrayExtents);
  {
    for (const_iterator i = x.getArray().begin(); i != x.getArray().end();
         ++i) {
      assert(*i == initialValue);
    }
  }
  {
    ArrayView view = x.getInteriorArray();
    typename ArrayView::const_iterator end = view.end();
    for (typename ArrayView::const_iterator i = view.begin(); i != end; ++i) {
      assert(*i == initialValue);
    }
  }
  // Initialize from another CellData.
  {
    CellData y(spatialIndex, x, initialValue);
    assert(y.getArray().extents() == arrayExtents + 2 * GhostWidth);
    assert(y.getInteriorArray().extents() == arrayExtents);
    {
      for (const_iterator i = y.getArray().begin(); i != y.getArray().end();
           ++i) {
        assert(*i == initialValue);
      }
    }
  }
  // Copy constructor.
  {
    CellData y(x);
    assert(x == y);
  }
  // Assignment operator.
  {
    CellData y(spatialIndex, arrayExtents,
               ext::filled_array<FieldTuple>(23.));
    y = x;
    assert(x == y);
  }
  // Prolong.
  {
    // Initialize the interior and ghost cells.
    {
      const MultiIndexRangeIterator end =
        MultiIndexRangeIterator::end(x.getArray().range());
      for (MultiIndexRangeIterator i =
             MultiIndexRangeIterator::begin(x.getArray().range()); i != end; ++i) {
        x.getArray()(*i) = ext::filled_array<FieldTuple>(stlib::ext::sum(*i));
      }
    }
    {
      spatialIndex.transformToChild(0);
      CellData y(spatialIndex, arrayExtents,
                 ext::filled_array<FieldTuple>(0.));
      y.prolong(x);
      Range interior = y.getInteriorArray().range();
      const Array& whole = y.getArray();
      const MultiIndexRangeIterator end =
        MultiIndexRangeIterator::end(whole.range());
      for (MultiIndexRangeIterator i =
             MultiIndexRangeIterator::begin(whole.range()); i != end; ++i) {
        if (isIn(interior, *i)) {
          assert(whole(*i) == x.getArray()(*i / Index(2)));
        }
        else {
          assert(whole(*i) == ext::filled_array<FieldTuple>(0.));
        }
      }
    }

    {
      spatialIndex.transformToNeighbor(1);
      CellData y(spatialIndex, arrayExtents,
                 ext::filled_array<FieldTuple>(0.));
      y.prolong(x);
      Range interior = y.getInteriorArray().range();
      const Array& whole = y.getArray();
      const MultiIndexRangeIterator end =
        MultiIndexRangeIterator::end(whole.range());
      for (MultiIndexRangeIterator i =
             MultiIndexRangeIterator::begin(whole.range()); i != end; ++i) {
        if (isIn(interior, *i)) {
          assert(whole(*i) == x.getArray()(*i / Index(2)));
        }
        else {
          assert(whole(*i) == ext::filled_array<FieldTuple>(0.));
        }
      }
    }
  }
  // Restrict.
  {
    x.getArray().fill(ext::filled_array<FieldTuple>(7.));
    for (std::size_t n = 0; n != Traits::NumberOfOrthants; ++n) {
      spatialIndex = SpatialIndex();
      spatialIndex.transformToChild(n);
      CellData y(spatialIndex, arrayExtents,
                 ext::filled_array<FieldTuple>(0.));
      const MultiIndexRangeIterator end =
        MultiIndexRangeIterator::end(y.getArray().range());
      for (MultiIndexRangeIterator i =
             MultiIndexRangeIterator::begin(y.getArray().range()); i != end; ++i) {
        y.getArray()(*i) =
          ext::filled_array<FieldTuple>(stlib::ext::sum(*i / Index(2)));
      }
      x.restrict(y);
    }
    {
      Range interior = x.getInteriorArray().range();
      const Array& whole = x.getArray();
      const MultiIndexRangeIterator end = MultiIndexRangeIterator::end(whole.range());
      for (MultiIndexRangeIterator i =
             MultiIndexRangeIterator::begin(whole.range()); i != end; ++i) {
        if (isIn(interior, *i)) {
          assert(whole(*i) == ext::filled_array<FieldTuple>(stlib::ext::sum(*i)));
        }
        else {
          assert(whole(*i) == ext::filled_array<FieldTuple>(7.));
        }
      }
    }
  }
  // Copy.
  {
    spatialIndex = SpatialIndex();
    spatialIndex.transformToChild(0);
    spatialIndex.transformToChild(Traits::NumberOfOrthants - 1);
    x = CellData(spatialIndex, arrayExtents,
                 ext::filled_array<FieldTuple>(0.));
    // Copy from each neighbor.
    for (int d = 0; d != 2 * Dimension; ++d) {
      SpatialIndex si = spatialIndex;
      si.transformToNeighbor(d);
      CellData y(si, arrayExtents, ext::filled_array<FieldTuple>(1.));
      x.copy(y);
    }

    std::size_t content = 1;
    // Surface area.
    for (std::size_t d = 0; d != Dimension - 1; ++d) {
      // Assume each extent is the same.
      content *= arrayExtents[0];
      assert(arrayExtents[d] == arrayExtents[Dimension - 1]);
    }
    // Number of sides.
    content *= 2 * Dimension;
    // Depth.
    content *= GhostWidth;
    std::size_t count = 0;
    for (const_iterator i = x.getArray().begin(); i != x.getArray().end();
         ++i) {
      if (*i == ext::filled_array<FieldTuple>(1.)) {
        ++count;
      }
    }
    assert(count == content);
  }
  // Message stream.
  {
    x = CellData(SpatialIndex(), arrayExtents,
                 ext::filled_array<FieldTuple>(1.));
    amr::MessageOutputStreamChecked out;
    out << x;
    amr::MessageInputStream in(out);
    assert(in == out);
    CellData y(SpatialIndex(), arrayExtents,
               ext::filled_array<FieldTuple>(3.));
    assert(!(x == y));
    in >> y;
    assert(x == y);
  }
  {
    assert(x.getMessageStreamSize() ==
           CellData::getMessageStreamSize(arrayExtents));
    amr::MessageOutputStream out(x.getMessageStreamSize());
    out << x;
    amr::MessageInputStream in(out);
    CellData y(SpatialIndex(), arrayExtents,
               ext::filled_array<FieldTuple>(0.));
    assert(!(x == y));
    in >> y;
    assert(x == y);
  }
  // File output.
  if (Dimension <= 2) {
    std::cout << x;
  }
}

int
main()
{
  test<1, 10, 0>();
  test<1, 10, 4>();
  test<2, 8, 0>();
  test<2, 8, 3>();
  test<3, 6, 0>();
  test<3, 6, 2>();
  test<4, 4, 0>();
  test<4, 4, 1>();

  return 0;
}
