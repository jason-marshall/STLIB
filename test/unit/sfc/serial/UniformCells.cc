// -*- C++ -*-

#include "stlib/sfc/UniformCells.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"

using namespace stlib;

TEST_CASE("UniformCells. 1-D, void, 0 levels.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;
  typedef UniformCells::Point Point;
  UniformCells cells(Point{{0}}, Point{{1}}, 0);

  SECTION("I/O.") {
    std::cout << cells;
  }  
  SECTION("Accessors.") {
    REQUIRE(cells.lowerCorner() == Point{{0}});
    REQUIRE(cells.lengths() == Point{{1}});
    REQUIRE(cells.numLevels() == 0);
    REQUIRE(cells.size() == 0);
  }
  SECTION("Copy, assign.") {
    // Copy constructor.
    UniformCells c = cells;
    // Assignment operator.
    c = cells;
  }
  SECTION("Order constructor.") {
    UniformCells c(cells.grid());
    REQUIRE(c.lowerCorner() == cells.lowerCorner());
    REQUIRE(c.lengths() == cells.lengths());
    REQUIRE(c.numLevels() == cells.numLevels());
    REQUIRE(c.size() == 0);
  }
  SECTION("Coarsen.") {
    REQUIRE(cells.coarsen(-1) == 0);
  }
}


TEST_CASE("UniformCells. 1-D, BBox, 20 levels, false. "
          "Uniformly-spaced points.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef sfc::UniformCells<Traits, Cell, false> UniformCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  SECTION("Build.") {
    UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
    REQUIRE(cells.lowerCorner() == Point{{0}});
    REQUIRE(cells.lengths() == Point{{1}});
    REQUIRE(cells.numLevels() == NumLevels);
    REQUIRE(cells.size() == 0);
    std::cout << cells;

    // Build.
    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());

    SECTION("Build from sorted.") {
      UniformCells fromSorted(cells.grid());
      fromSorted.buildCells(objects);
      REQUIRE(fromSorted.size() == cells.size());
      for (std::size_t i = 0; i != fromSorted.size(); ++i) {
        REQUIRE(fromSorted[i] == cells[i]);
      }
    }

    SECTION("Rebuild.") {
      cells.clear();
      REQUIRE(cells.size() == 0);
      cells.shrink_to_fit();
      REQUIRE(cells.size() == 0);
      cells.buildCells(&objects);
      REQUIRE(cells.size() == objects.size());
      cells.shrink_to_fit();
      REQUIRE(cells.size() == objects.size());
    }

    SECTION("Coarsen.") {
      while (cells.numLevels() != 0) {
        cells.coarsen();
      }
      REQUIRE(cells.numLevels() == 0);
      REQUIRE(cells.size() == 1);
      REQUIRE(cells[0].lower[0] == 0);
      REQUIRE(cells[0].upper[0] ==
              Float(objects.size() - 1) / objects.size());
    }
  }

  SECTION("Serialize.") {
    UniformCells x(Point{{0}}, Point{{1}}, NumLevels);
    x.buildCells(&objects);
    std::vector<unsigned char> buffer;
    x.serialize(&buffer);
    UniformCells y(Point{{0}}, Point{{1}}, NumLevels);
    y.unserialize(buffer);
    REQUIRE(x == y);
  }
}


TEST_CASE("UniformCells. 1-D, BBox, 20 levels. "
          "Uniformly-spaced points.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef sfc::UniformCells<Traits, Cell, false> UniformCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  SECTION("Build.") {
    UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
    REQUIRE(cells.lowerCorner() == Point{{0}});
    REQUIRE(cells.lengths() == Point{{1}});
    REQUIRE(cells.numLevels() == NumLevels);
    REQUIRE(cells.size() == 0);

    // Build.
    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());

    SECTION("Build from sorted.") {
      UniformCells fromSorted(cells.grid());
      fromSorted.buildCells(objects);
      REQUIRE(fromSorted.size() == cells.size());
      for (std::size_t i = 0; i != fromSorted.size(); ++i) {
        REQUIRE(fromSorted[i] == cells[i]);
      }
    }

    SECTION("Rebuild.") {
      cells.clear();
      REQUIRE(cells.size() == 0);
      cells.shrink_to_fit();
      REQUIRE(cells.size() == 0);
      cells.buildCells(&objects);
      REQUIRE(cells.size() == objects.size());
      cells.shrink_to_fit();
      REQUIRE(cells.size() == objects.size());
    }

    SECTION("Coarsen.") {
      while (cells.numLevels() != 0) {
        cells.coarsen();
      }
      REQUIRE(cells.numLevels() == 0);
      REQUIRE(cells.size() == 1);
      REQUIRE(cells[0].lower[0] == 0);
      REQUIRE(cells[0].upper[0] ==
              Float(objects.size() - 1) / objects.size());
    }
  }

  SECTION("Serialize.") {
    UniformCells x(Point{{0}}, Point{{1}}, NumLevels);
    x.buildCells(&objects);
    std::vector<unsigned char> buffer;
    x.serialize(&buffer);
    UniformCells y(Point{{0}}, Point{{1}}, NumLevels);
    y.unserialize(buffer);
    REQUIRE(x == y);
  }
}


TEST_CASE("UniformCells. Order and restore.",
          "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef sfc::UniformCells<Traits, void, false> UniformCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points, shuffled.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }
  std::random_shuffle(objects.begin(), objects.end());

  std::vector<Point> const originalObjects(objects);
  UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
  sfc::OrderedObjects orderedObjects;
  cells.buildCells(&objects, &orderedObjects);
  std::vector<Point> copy(originalObjects);
  orderedObjects.order(copy.begin(), copy.end());
  REQUIRE(copy == objects);
  orderedObjects.restore(copy.begin(), copy.end());
  REQUIRE(copy == originalObjects);
}


TEST_CASE("UniformCells. 1-D, void, 20 levels. Coarsen.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
  cells.buildCells(&objects);

  // Coarsen.
  REQUIRE(cells.coarsen(0) == 0);
  REQUIRE(cells.coarsen(1) == 20 - 9);
  REQUIRE(cells.numLevels() == 9);
  REQUIRE(cells.coarsen(2) == 1);
  REQUIRE(cells.numLevels() == 8);
  REQUIRE(cells.size() == 256);
}


TEST_CASE("UniformCells. 1-D, void, 3 levels. Coarsen.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 3;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(8);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
  cells.buildCells(&objects);

  // Coarsen.
  REQUIRE(cells.coarsen(1) == 0);
  REQUIRE(cells.coarsen(2) == 1);
  REQUIRE(cells.numLevels() == 2);
}


TEST_CASE("UniformCells. 1-D, void, 20 levels. Merge.", "[UniformCells]")
{
  typedef sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef sfc::UniformCells<Traits, void, true> UniformCells;
  typedef UniformCells::Code Code;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  SECTION("Merge with a copy.") {
    UniformCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    cells.checkValidity();
    REQUIRE(cells.size() == objects.size());
    UniformCells copy = cells;
    cells += copy;
    cells.checkValidity();
    REQUIRE(cells.size() == objects.size());
    for (std::size_t i = 0; i != cells.size(); ++i) {
      REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) ==
              Code(2));
    }
  }
  SECTION("Merge separate halves.") {
    std::vector<Point> objects1(objects.begin(),
                                objects.begin() + objects.size() / 2);
    std::vector<Point> objects2(objects.begin() + objects.size() / 2,
                                objects.end());
    UniformCells cells1(Point{{0}}, Point{{1}}, NumLevels);
    cells1.buildCells(&objects1);
    REQUIRE(cells1.size() == objects1.size());
    cells1.checkValidity();
    UniformCells cells2(Point{{0}}, Point{{1}}, NumLevels);
    cells2.buildCells(&objects2);
    REQUIRE(cells2.size() == objects2.size());
    cells2.checkValidity();

    cells1 += cells2;
    cells1.checkValidity();
    REQUIRE(cells1.size() == objects.size());
  }
}
