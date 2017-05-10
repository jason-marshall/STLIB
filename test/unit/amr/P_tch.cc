// -*- C++ -*-

// Note this file is intentionally misspelled. On Windows, running any 
// program with the word "patch" in its name will bring up a dialog asking,
// "Do you want to allow the following program from an 
// unknown publisher to make changes to this computer?"

#include "stlib/amr/Patch.h"
#include "stlib/amr/Orthtree.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/CellData.h"

#include <iostream>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;

using namespace stlib;

template<std::size_t Dimension, std::size_t MaximumLevel>
inline
void
test()
{
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n";
  const std::size_t Depth = 1;
  const std::size_t GhostWidth = 1;

  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef typename Traits::SpatialIndex SpatialIndex;
  typedef typename Traits::Point Point;

  // The patch holds integer data.
  typedef amr::CellData<Traits, Depth, GhostWidth, int> CellData;
  typedef typename CellData::SizeList SizeList;
  typedef typename CellData::FieldTuple FieldTuple;
  typedef amr::Patch<CellData, Traits> Patch;

  typedef amr::Orthtree<Patch, Traits> Orthtree;

  // Constructor.
  const Point lowerCorner = ext::filled_array<Point>(0.);
  const Point extents = ext::filled_array<Point>(1.);
  Orthtree orthtree(lowerCorner, extents);
  const SizeList patchExtents = ext::filled_array<SizeList>(1);
  {
    // Key at level 0.
    const SpatialIndex key;
    // Patch with one element.
    const Patch patch(key, patchExtents);
    // Insert the node.
    orthtree.insert(key, patch);
  }

  // Level 0.
  {
    assert(orthtree.size() == 1);
    assert(orthtree.begin()->second.getPatchData().getInteriorArray().extents()
           == patchExtents);
    assert(orthtree.begin()->second.getPatchData().getArray().extents() ==
           patchExtents + 2 * GhostWidth);
  }
  // Level 1.
  {
    orthtree.split(orthtree.begin());
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
    Patch x(SpatialIndex(), patchExtents);
    assert(x.getMessageStreamSize() ==
           Patch::getMessageStreamSize(patchExtents));
    std::fill(x.getPatchData().getArray().begin(),
              x.getPatchData().getArray().end(),
              ext::filled_array<FieldTuple>(7));
    amr::MessageOutputStream out(x.getMessageStreamSize());
    out << x;
    amr::MessageInputStream in(out);
    Patch y(SpatialIndex(), patchExtents);
    assert(!(x == y));
    in >> y;
    assert(x == y);
  }
  // File output.
  if (Dimension <= 2) {
    std::cout << orthtree;
  }
}

template<typename _Node>
inline
int&
getData(const _Node node)
{
  return (*node->second.getPatchData().getArray().data())[0];
}

template<std::size_t MaximumLevel>
inline
void
test2()
{
  const std::size_t Dimension = 2;
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n";
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef typename Traits::SpatialIndex SpatialIndex;
  typedef typename Traits::Point Point;

  typedef amr::CellData < Traits, 1 /*Depth*/, 1 /*GhostWidth*/, int > CellData;
  typedef typename CellData::SizeList SizeList;
  typedef amr::Patch<CellData, Traits> Patch;

  typedef amr::Orthtree<Patch, Traits> Orthtree;
  typedef typename Orthtree::iterator iterator;

  Orthtree orthtree(ext::filled_array<Point>(0.),
                    ext::filled_array<Point>(1.));
  {
    // Key at level 0.
    const SpatialIndex key;
    // 2 x 2 patch.
    const Patch patch(key, ext::filled_array<SizeList>(2));
    // Insert the node.
    orthtree.insert(key, patch);
  }

  // Multi-level
  orthtree.split(orthtree.begin());
  orthtree.split(orthtree.begin());
  assert(orthtree.size() == 7);

  int n = 0;
  for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
    getData(i) = n++;
  }
  n = 0;
  for (iterator i = orthtree.begin(); i != orthtree.end(); ++i) {
    assert(getData(i) == n++);
  }
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
