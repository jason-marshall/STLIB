// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/amr/PatchAdjacent.h"
#include "stlib/amr/Orthtree.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/CellData.h"

#include <iostream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

using namespace stlib;

template<typename _Patch, class _Traits>
inline
void
synchronize(amr::Orthtree<_Patch, _Traits>* orthtree)
{
  typedef amr::Orthtree<_Patch, _Traits> Orthtree;
  typedef typename Orthtree::iterator iterator;
  typedef typename Orthtree::Patch::PatchData::ArrayView ArrayView;
  typedef typename Orthtree::Patch::PatchData::FieldTuple FieldTuple;
  typedef typename _Traits::IndexList IndexList;
  typedef typename _Traits::Index Index;

  const Index GhostWidth = Orthtree::Patch::PatchData::GhostWidth;
  for (iterator i = orthtree->begin(); i != orthtree->end(); ++i) {
    i->second.getPatchData().getArray().
    fill(ext::filled_array<FieldTuple>(0));
    i->second.getPatchData().getInteriorArray().
    fill(ext::filled_array<FieldTuple>(1));
  }
  orthtree->linkBalanced();
  orthtree->synchronizeBalanced();
  IndexList offset = ext::filled_array<IndexList>(0);
  for (iterator node = orthtree->begin(); node != orthtree->end(); ++node) {
    for (std::size_t d = 0; d != 2 * _Traits::Dimension; ++d) {
      if (node->second.adjacent[d] != 0) {
        ArrayView array = node->second.getPatchData().getInteriorArray();
        const typename ArrayView::iterator end = array.end();
        for (typename ArrayView::iterator i = array.begin(); i != end; ++i) {
          offset[d / 2] = 2 * (d % 2) - 1;
          for (Index g = 1; g <= GhostWidth; ++g) {
            assert(node->second.getPatchData().getArray()(i.indexList() +
                   g * offset) ==
                   ext::filled_array<FieldTuple>(1));
          }
          std::fill(offset.begin(), offset.end(), 0);
        }
      }
    }
  }
}

template<std::size_t Dimension, std::size_t MaximumLevel>
inline
void
test()
{
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n";
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef typename Traits::SpatialIndex SpatialIndex;
  typedef typename Traits::Point Point;
  typedef typename Traits::SizeList SizeList;

  typedef amr::CellData < Traits, 1 /*Depth*/, 1 /*GhostWidth*/, int > CellData;
  typedef typename CellData::FieldTuple FieldTuple;
  typedef amr::PatchAdjacent<CellData, Traits> Patch;

  typedef amr::Orthtree<Patch, Traits> Orthtree;

  // Constructor.
  const Point lowerCorner = ext::filled_array<Point>(0.),
              extents = ext::filled_array<Point>(1.);
  Orthtree orthtree(lowerCorner, extents);
  {
    // Key at level 0.
    const SpatialIndex key;
    // Patch with one element.
    const Patch patch(key, ext::filled_array<SizeList>(1));
    // Insert the node.
    orthtree.insert(key, patch);
  }

  // Level 0.
  {
    Patch& patch = orthtree.begin()->second;
    for (int d = 0; d != 2 * Dimension; ++d) {
      assert(patch.adjacent[d] == 0);
    }
    orthtree.linkBalanced();
    for (int d = 0; d != 2 * Dimension; ++d) {
      assert(patch.adjacent[d] == 0);
    }
    synchronize(&orthtree);
  }
  // Level 1.
  {
    orthtree.split(orthtree.begin());
    // Test the first node.
    Patch& patch = orthtree.begin()->second;
    orthtree.linkBalanced();
    for (int d = 0; d != Dimension; ++d) {
      assert(patch.adjacent[2 * d] == 0);
      assert(patch.adjacent[2 * d + 1] != 0);
    }
    synchronize(&orthtree);
  }
  // Message stream.
  {
    Patch x(SpatialIndex(), ext::filled_array<SizeList>(1));
    x.getPatchData().getArray().fill(ext::filled_array<FieldTuple>(7));
    amr::MessageOutputStreamChecked out;
    out << x;
    amr::MessageInputStream in(out);
    Patch y(SpatialIndex(), ext::filled_array<SizeList>(1));
    assert(!(x == y));
    in >> y;
    assert(x == y);
  }
  {
    Patch x(SpatialIndex(), ext::filled_array<SizeList>(1));
    assert(x.getMessageStreamSize() ==
           Patch::getMessageStreamSize(ext::filled_array<SizeList>(1)));
    x.getPatchData().getArray().fill(ext::filled_array<FieldTuple>(7));
    amr::MessageOutputStream out(x.getMessageStreamSize());
    out << x;
    amr::MessageInputStream in(out);
    Patch y(SpatialIndex(), ext::filled_array<SizeList>(1));
    assert(!(x == y));
    in >> y;
    assert(x == y);
  }
}

template<typename _Node>
inline
_Node
getAdjacent(const _Node node, const int direction)
{
  return node->second.adjacent[direction];
}

template<typename _Node>
inline
int&
getData(const _Node node)
{
  return (*node->second.getPatchData().getArray().begin())[0];
}

template<typename _Node>
inline
int&
getAdjacentData(const _Node node, const int direction)
{
  return getData(getAdjacent(&*node, direction));
}

template<std::size_t MaximumLevel>
inline
void
test2()
{
  const int Dimension = 2;
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n";
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef typename Traits::SpatialIndex SpatialIndex;
  typedef typename Traits::Point Point;
  typedef typename Traits::SizeList SizeList;

  typedef amr::CellData < Traits, 1 /*Depth*/, 1 /*GhostWidth*/, int > CellData;
  typedef amr::PatchAdjacent<CellData, Traits> Patch;
  typedef amr::Orthtree<Patch, Traits> Orthtree;
  typedef typename Orthtree::iterator iterator;

  const Point lowerCorner = ext::filled_array<Point>(0.),
              extents = ext::filled_array<Point>(1.);
  Orthtree orthtree(lowerCorner, extents);
  {
    // Key at level 0.
    const SpatialIndex key;
    // Insert the node.
    orthtree.insert(key, Patch(key, ext::filled_array<SizeList>(2)));
  }

  // Multi-level
  orthtree.split(orthtree.begin());
  orthtree.split(orthtree.begin());
  assert(orthtree.size() == 7);

  // Test synchronize.
  synchronize(&orthtree);

  orthtree.linkBalanced();
  int n = 0;
  for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
    getData(i) = n++;
  }
  n = 0;
  for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
    std::cout << "\nNode " << n++ << '\n';
    for (int d = 0; d != 2 * Dimension; ++d) {
      std::cout << d << ' ';
      if (getAdjacent(&*i, d) != 0) {
        std::cout << getAdjacentData(i, d) << '\n';
      }
      else {
        std::cout << "null\n";
      }
    }
  }
  /*
     --- ---
    |   |   |
    | 5 | 6 |
    |   |   |
    |- - ---
    |2|3|   |
    |- -  4 |
    |0|1|   |
     - - ---
  */
  // 0
  iterator i = orthtree.begin();
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 1);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 2);
  // 1
  ++i;
  assert(getAdjacentData(i, 0) == 0);
  assert(getAdjacentData(i, 1) == 4);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 3);
  // 2
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 3);
  assert(getAdjacentData(i, 2) == 0);
  assert(getAdjacentData(i, 3) == 5);
  // 3
  ++i;
  assert(getAdjacentData(i, 0) == 2);
  assert(getAdjacentData(i, 1) == 4);
  assert(getAdjacentData(i, 2) == 1);
  assert(getAdjacentData(i, 3) == 5);
  // 4
  ++i;
  assert(getAdjacentData(i, 0) == 1);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 6);
  // 5
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 6);
  assert(getAdjacentData(i, 2) == 2);
  assert(getAdjacent(&*i, 3) == 0);
  // 6
  ++i;
  assert(getAdjacentData(i, 0) == 5);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacentData(i, 2) == 4);
  assert(getAdjacent(&*i, 3) == 0);

  orthtree.erase(orthtree.begin());
  orthtree.linkBalanced();
  /*
     --- ---
    |   |   |
    | 5 | 6 |
    |   |   |
    |- - ---
    |2|3|   |
    |- -  4 |
    | |1|   |
     - - ---
  */
  // 1
  i = orthtree.begin();
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 4);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 3);
  // 2
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 3);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 5);
  // 3
  ++i;
  assert(getAdjacentData(i, 0) == 2);
  assert(getAdjacentData(i, 1) == 4);
  assert(getAdjacentData(i, 2) == 1);
  assert(getAdjacentData(i, 3) == 5);
  // 4
  ++i;
  assert(getAdjacentData(i, 0) == 1);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 6);
  // 5
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 6);
  assert(getAdjacentData(i, 2) == 2);
  assert(getAdjacent(&*i, 3) == 0);
  // 6
  ++i;
  assert(getAdjacentData(i, 0) == 5);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacentData(i, 2) == 4);
  assert(getAdjacent(&*i, 3) == 0);


  orthtree.erase(orthtree.begin());
  orthtree.linkBalanced();
  /*
     --- ---
    |   |   |
    | 5 | 6 |
    |   |   |
    |- - ---
    |2|3|   |
    |- -  4 |
    | | |   |
     - - ---
  */
  // 2
  i = orthtree.begin();
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 3);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 5);
  // 3
  ++i;
  assert(getAdjacentData(i, 0) == 2);
  assert(getAdjacentData(i, 1) == 4);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 5);
  // 4
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacent(&*i, 2) == 0);
  assert(getAdjacentData(i, 3) == 6);
  // 5
  ++i;
  assert(getAdjacent(&*i, 0) == 0);
  assert(getAdjacentData(i, 1) == 6);
  assert(getAdjacentData(i, 2) == 2);
  assert(getAdjacent(&*i, 3) == 0);
  // 6
  ++i;
  assert(getAdjacentData(i, 0) == 5);
  assert(getAdjacent(&*i, 1) == 0);
  assert(getAdjacentData(i, 2) == 4);
  assert(getAdjacent(&*i, 3) == 0);
}

int
main()
{
  test<1, 10>();
  test<2, 8>();
  test<3, 6>();
  test<4, 4>();

  test2<8>();

  return 0;
}
