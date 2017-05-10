// -*- C++ -*-

#include "stlib/sfc/LinearOrthantTrie.h"

#include <random>

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


template<typename _Cell, typename _Traits, bool _StoreDel>
std::size_t
_countDfs
(const stlib::sfc::LinearOrthantTrie<_Cell, _Traits, _StoreDel>& trie,
 const std::size_t n)
{
  // One for the nth cell.
  std::size_t count = 1;
  if (trie.isInternal(n)) {
    // Count the children.
    stlib::sfc::Siblings<_Traits::Dimension> children;
    trie.getChildren(n, &children);
    for (std::size_t i = 0; i != children.size(); ++i) {
      count += _countDfs(trie, children[i]);
    }
  }
  return count;
}


template<typename _Cell, typename _Traits, bool _StoreDel>
inline
std::size_t
countDfs
(const stlib::sfc::LinearOrthantTrie<_Cell, _Traits, _StoreDel>& trie)
{
  return _countDfs(trie, 0);
}


template<typename _Cell, typename _Traits, bool _StoreDel>
inline
std::size_t
countDfsStack
(const stlib::sfc::LinearOrthantTrie<_Cell, _Traits, _StoreDel>& trie)
{
  stlib::sfc::Siblings<_Traits::Dimension> children;
  std::vector<std::size_t> cells;
  if (trie.size() != 0) {
    cells.push_back(0);
  }
  std::size_t count = 0;
  while (! cells.empty()) {
    std::size_t const c = cells.back();
    REQUIRE((c == count));
    cells.pop_back();
    ++count;
    if (trie.isInternal(c)) {
      trie.getChildren(c, &children);
      cells.insert(cells.end(), children.rbegin(), children.rend());
    }
  }
  return count;
}


TEST_CASE("LinearOrthantTrie.", "[LinearOrthantTrie]")
{
  std::default_random_engine engine;
  std::uniform_real_distribution<double> distribution(0, 1);

  SECTION("1-D, 0 levels. Empty.") {
    typedef stlib::sfc::Traits<1> Traits;
    typedef stlib::sfc::LinearOrthantTrie<Traits, void, false> Trie;
    typedef Trie::Point Point;
    // Empty trie.
    Trie trie(Point{{0}}, Point{{1}}, 0);
    REQUIRE(trie.lowerCorner() == Point{{0}});
    REQUIRE(trie.lengths() == Point{{1}});
    REQUIRE(trie.numLevels() == 0);
    REQUIRE(trie.size() == 0);
    {
      // Copy constructor.
      Trie t = trie;
      // Assignment operator.
      t = trie;
    }
  }

  SECTION("Uniformly-spaced points.") {
    // 1-D, BBox, 20 levels.
    typedef stlib::sfc::Traits<1> Traits;
    const std::size_t NumLevels = 20;
    typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef stlib::sfc::LinearOrthantTrie<Traits, Cell, true> Trie;
    typedef Trie::Float Float;
    typedef Trie::Point Point;
    typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
    typedef stlib::sfc::UniformCells<Traits, Cell, true> UniformCells;

    // Uniformly-spaced points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    // Accessors.
    {
      Trie trie(Point{{0}}, Point{{1}}, NumLevels);
      REQUIRE(trie.lowerCorner() == Point{{0}});
      REQUIRE(trie.lengths() == Point{{1}});
      REQUIRE(trie.numLevels() == NumLevels);
      REQUIRE(trie.size() == 0);
      REQUIRE(trie.calculateHighestLevel() == 0);
      REQUIRE(trie.calculateMaxLeafSize() == 0);

      trie.buildCells(&objects);
      trie.checkValidity();
      REQUIRE(trie.calculateHighestLevel() == trie.numLevels());
      REQUIRE(trie.calculateMaxLeafSize() == 1);
      REQUIRE(trie.countLeaves() == objects.size());
      // The root cell contains all objects.
      REQUIRE((trie.delimiter(trie.next(0)) - trie.delimiter(0)) ==
              objects.size());
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          REQUIRE((trie.delimiter(i + 1) - trie.delimiter(i)) ==
                  std::size_t(1));
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) ==
                  std::size_t(1));
        }
        else {
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) >=
                  std::size_t(1));
        }
      }

      // Perform a DFS of the trie.
      REQUIRE(countDfs(trie) == trie.size());
      REQUIRE(countDfsStack(trie) == trie.size());
      const std::size_t size = trie.size();
      trie.shrink_to_fit();
      trie.checkValidity();
      REQUIRE(trie.size() == size);
      trie.clear();
      trie.checkValidity();
      REQUIRE(trie.size() == 0);
      trie.shrink_to_fit();
      trie.checkValidity();
      REQUIRE(trie.size() == 0);
    }

    // Build from AdaptiveCells.
    {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      cells.checkValidity();

      Trie trie(cells);
      trie.checkValidity();
      // Check the number of leaves.
      REQUIRE(trie.countLeaves() == cells.size());
      // Check the number of objects in the leaves.
      std::size_t j = 0;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) ==
                  (cells.delimiter(j + 1) - cells.delimiter(j)));
          ++j;
        }
      }
    }

    // Build from UniformCells.
    {
      UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      cells.checkValidity();

      Trie trie(cells);
      trie.checkValidity();
      // Check the number of leaves.
      REQUIRE(trie.countLeaves() == cells.size());
      // Check the number of objects in the leaves.
      std::size_t j = 0;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) ==
                  (cells.delimiter(j + 1) - cells.delimiter(j)));
          ++j;
        }
      }
    }

    // Serialize.
    {
      Trie x(Point{{0}}, Point{{1}}, NumLevels);
      x.buildCells(&objects);
      std::vector<unsigned char> buffer;
      x.serialize(&buffer);
      Trie y(Point{{0}}, Point{{1}}, NumLevels);
      y.unserialize(buffer);
      REQUIRE(x == y);
    }

    // A single point at the origin.
    objects.resize(1);
    objects[0][0] = 0;

    // Insert internal.
    {
      Trie trie(Point{{0}}, Point{{1}}, NumLevels);
      trie.buildCells(&objects);
      trie.checkValidity();
      REQUIRE(trie.size() == trie.numLevels() + 1);
      for (std::size_t i = 0; i != trie.size() - 1; ++i) {
        REQUIRE(trie.isInternal(i));
      }
      REQUIRE(trie.isLeaf(trie.size() - 1));
    }
  }

  SECTION("Random points. 1-D.") {
    // 1-D, BBox, 20 levels.
    typedef stlib::sfc::Traits<1> Traits;
    const std::size_t NumLevels = 20;
    typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef stlib::sfc::LinearOrthantTrie<Traits, Cell, true> Trie;
    typedef Trie::Point Point;
    typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;

    // Random points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = distribution(engine);
    }

    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    cells.checkValidity();

    for (std::size_t leafSize = 1; leafSize <= objects.size(); leafSize *= 2) {
      cells.coarsenCellSize(leafSize);
      cells.checkValidity();
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      for (std::size_t i = 0; i != cells.size(); ++i) {
        REQUIRE(cells.delimiter(i + 1) > cells.delimiter(i));
      }

      Trie trie(cells);
      trie.checkValidity();
      REQUIRE(trie.countLeaves() == cells.size());
      // Check the number of objects in the leaves.
      std::size_t j = 0;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) ==
                  (cells.delimiter(j + 1) - cells.delimiter(j)));
          ++j;
        }
      }
      // Perform a DFS of the trie.
      REQUIRE(countDfs(trie) == trie.size());
      REQUIRE(countDfsStack(trie) == trie.size());
    }
    REQUIRE(cells.size() == 1);
  }

  SECTION("Random points. 3-D.") {
    // 3-D, BBox, 10 levels.
    typedef stlib::sfc::Traits<3> Traits;
    const std::size_t NumLevels = 10;
    typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef stlib::sfc::LinearOrthantTrie<Traits, Cell, true> Trie;
    typedef Trie::Point Point;
    typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;

    // Random points.
    std::vector<Point> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      for (std::size_t j = 0; j != Traits::Dimension; ++j) {
        objects[i][j] = distribution(engine);
      }
    }

    AdaptiveCells cells(stlib::ext::filled_array<Point>(0),
                          stlib::ext::filled_array<Point>(1), NumLevels);
    cells.buildCells(&objects);
    cells.checkValidity();

    // coarsenCellSize()
    for (std::size_t leafSize = 1; leafSize <= objects.size(); leafSize *= 2) {
      cells.coarsenCellSize(leafSize);
      cells.checkValidity();
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      for (std::size_t i = 0; i != cells.size(); ++i) {
        REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) <=
                leafSize);
      }

      Trie trie(cells);
      trie.checkValidity();
      REQUIRE(trie.countLeaves() == cells.size());
      // Check the number of objects in the leaves.
      std::size_t j = 0;
      for (std::size_t i = 0; i != trie.size(); ++i) {
        if (trie.isLeaf(i)) {
          REQUIRE((trie.delimiter(trie.next(i)) - trie.delimiter(i)) ==
                  (cells.delimiter(j + 1) - cells.delimiter(j)));
          ++j;
        }
      }
      // Perform a DFS of the trie.
      REQUIRE(countDfs(trie) == trie.size());
      REQUIRE(countDfsStack(trie) == trie.size());
    }
    REQUIRE(cells.size() == 1);
  }
}


TEST_CASE("linearOrthantTrie()", "[LinearOrthantTrie]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::LinearOrthantTrie<Traits, void, true> LinearOrthantTrie;
  typedef LinearOrthantTrie::Float Float;
  typedef LinearOrthantTrie::Point Object;
  using stlib::sfc::linearOrthantTrie;

  {
    // No objects.
    std::vector<Object> objects;
    LinearOrthantTrie const trie =
      linearOrthantTrie<LinearOrthantTrie>(&objects);
    REQUIRE(objects.empty());
    REQUIRE(trie.empty());
  }
  {
    // Uniformly-spaced points.
    std::vector<Object> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    {
      // Default maximum number of objects per cell.
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects);
      REQUIRE(objects.size() == 512);
      REQUIRE_FALSE(trie.empty());
    }
    {
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects, 64);
      REQUIRE(trie.calculateMaxLeafSize() <= 64);
    }
    {
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects, 1);
      REQUIRE(trie.calculateMaxLeafSize() == 1);
    }
  }
}


TEST_CASE("linearOrthantTrie() with OrderedObjects", "[LinearOrthantTrie]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::LinearOrthantTrie<Traits, void, true> LinearOrthantTrie;
  typedef LinearOrthantTrie::Float Float;
  typedef LinearOrthantTrie::Point Object;
  using stlib::sfc::linearOrthantTrie;

  stlib::sfc::OrderedObjects orderedObjects;
  {
    // No objects.
    std::vector<Object> const original;
    std::vector<Object> objects = original;
    LinearOrthantTrie const trie =
      linearOrthantTrie<LinearOrthantTrie>(&objects, &orderedObjects);
    REQUIRE(objects.empty());
    REQUIRE(trie.empty());
    orderedObjects.restore(objects.begin(), objects.end());
    REQUIRE(objects == original);
  }
  {
    // Uniformly-spaced points.
    std::vector<Object> original(512);
    for (std::size_t i = 0; i != original.size(); ++i) {
      original[i][0] = Float(i) / original.size();
    }

    {
      // Default maximum number of objects per cell.
      std::vector<Object> objects = original;
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects, &orderedObjects);
      REQUIRE(objects.size() == 512);
      REQUIRE_FALSE(trie.empty());
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
    {
      std::vector<Object> objects = original;
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects, &orderedObjects, 64);
      REQUIRE(trie.calculateMaxLeafSize() <= 64);
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
    {
      std::vector<Object> objects = original;
      LinearOrthantTrie const trie =
        linearOrthantTrie<LinearOrthantTrie>(&objects, &orderedObjects, 1);
      REQUIRE(trie.calculateMaxLeafSize() == 1);
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
  }
}
