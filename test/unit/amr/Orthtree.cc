// -*- C++ -*-

#include "stlib/amr/Orthtree.h"
#include "stlib/amr/Traits.h"
#include "stlib/amr/Patch.h"
#include "stlib/amr/CellData.h"

using namespace stlib;

template<int Dimension, int MaximumLevel>
void
test()
{
  std::cout << "----------------------------------------------------------\n"
            << "Dimension = " << Dimension
            << ", MaximumLevel = " << MaximumLevel << "\n";
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef amr::CellData<Traits, 1U, 0U> PatchData;
  typedef amr::Patch<PatchData, Traits> Patch;
  typedef typename Patch::SizeList SizeList;
  typedef amr::Orthtree<Patch, Traits> Orthtree;
  typedef typename Orthtree::SpatialIndex SpatialIndex;
  typedef typename Orthtree::iterator iterator;
  typedef typename Orthtree::Point Point;
  static_assert(Orthtree::Dimension == Dimension, "Bad dimension.");
  static_assert(Orthtree::MaximumLevel == MaximumLevel, "Bad maximum level.");
  static_assert(Orthtree::NumberOfOrthants == 1 << Dimension,
                "Bad number of orthants.");
  {
    // Key at level 0.
    const SpatialIndex key;
    // Patch with one element.
    const Patch patch(key, ext::filled_array<SizeList>(1));

    // Constructor.
    Point lowerCorner = ext::filled_array<Point>(0),
          extents = ext::filled_array<Point>(1);
    Orthtree x(lowerCorner, extents);
    // Accessors.
    assert(x.begin() == x.end());
    assert(x.size() == 0);
    assert(x.max_size() > 0);
    assert(x.empty());
    assert(x.isBalanced());
    for (int level = 0; level <= MaximumLevel; ++level) {
      assert(! hasNodesAtLevel(x, level));
    }
    // Search.
    assert(x.count(key) == 0);
    // Insert.
    iterator i = x.insert(key, patch);
    assert(i->first == key);
    assert(i->second == patch);
    assert(x.begin()->first == key);
    assert(x.begin()->second == patch);
    assert(x.begin() != x.end());
    assert(x.size() == 1);
    assert(! x.empty());
    assert(hasNodesAtLevel(x, 0));
    for (int level = 1; level <= MaximumLevel; ++level) {
      assert(! hasNodesAtLevel(x, level));
    }
    // Search.
    assert(x.count(key) == 1);
    assert(x.find(key) == i);
    // Print.
    std::cout << x;
    amr::printVtkUnstructuredGrid(std::cout, x);
    // Erase through iterator.
    x.erase(i);
    assert(x.empty());
    // Erase through key.
    x.insert(key, patch);
    assert(x.erase(key) == 1);
    assert(x.empty());

    //
    // Refine.
    //
    // Insert the node.
    x.insert(key, patch);
    while (x.canBeRefined(x.begin())) {
      x.split(x.begin());
    }
    assert(x.isBalanced());
    std::cout << "Size after refinement of the first node = " << x.size()
              << "\n";

    x.clear();
    x.insert(key, patch);
    x.split(x.begin());
    while (x.canBeRefined(++x.begin())) {
      x.split(++x.begin());
    }
    assert(! x.isBalanced());
    std::cout << "Size after refinement of the second node = " << x.size()
              << "\n";
  }
}

template<int Dimension, int MaximumLevel>
void
testArray()
{
  std::cout << "Array elements.\n";
  typedef container::MultiArray<double, Dimension> Data;
  typedef amr::Traits<Dimension, MaximumLevel> Traits;
  typedef amr::Orthtree<Data, Traits> Orthtree;
  typedef typename Orthtree::Point Point;
  static_assert(Orthtree::Dimension == Dimension, "Bad dimension.");
  static_assert(Orthtree::MaximumLevel == MaximumLevel, "Bad maximum level.");
  static_assert(Orthtree::NumberOfOrthants == 1 << Dimension,
                "Bad number of orthants.");
  {
    // Constructor.
    Point lowerCorner = ext::filled_array<Point>(0),
          extents = ext::filled_array<Point>(1);
    Orthtree x(lowerCorner, extents);
  }
}

int
main()
{
  test<1, 10>();
  test<2, 8>();
  test<3, 6>();
  test<4, 4>();

  testArray<1, 10>();
  testArray<2, 8>();
  testArray<3, 6>();
  testArray<4, 4>();

  return 0;
}
