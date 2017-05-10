// -*- C++ -*-

#include "stlib/sfc/AdaptiveCells.h"

#define CATCH_CONFIG_MAIN
#include "catch.hpp"


TEST_CASE("AdaptiveCells. 1-D, std::size_t.", "[AdaptiveCells]") {
  typedef stlib::sfc::Traits<1> Traits;
  typedef std::size_t Cell;
  typedef stlib::sfc::AdaptiveCells<Traits, Cell, false> AdaptiveCells;
  typedef AdaptiveCells::Point Point;
  typedef AdaptiveCells::BBox BBox;

  AdaptiveCells cells(Point{{0}}, Point{{1}}, 0);
  REQUIRE(cells.lowerCorner() == Point{{0}});
  REQUIRE(cells.lengths() == Point{{1}});
  REQUIRE(cells.numLevels() == 0);
  REQUIRE(cells.size() == 0);
  REQUIRE(cells.calculateHighestLevel() == 0);
  {
    // Copy constructor.
    AdaptiveCells c = cells;
    // Assignment operator.
    c = cells;
  }
  {
    // Order constructor.
    AdaptiveCells c(cells.grid());
    REQUIRE(c.lowerCorner() == cells.lowerCorner());
    REQUIRE(c.lengths() == cells.lengths());
    REQUIRE(c.numLevels() == cells.numLevels());
    REQUIRE(c.size() == 0);
    REQUIRE(c.calculateHighestLevel() == 0);
  }
  {
    // Cell length constructor.
    AdaptiveCells c(BBox{{{0}}, {{1}}}, 0.1);
    REQUIRE(c.lowerCorner()[0] <= cells.lowerCorner()[0]);
    REQUIRE(c.lengths()[0] >= cells.lengths()[0]);
    REQUIRE(c.numLevels() == 3);
    REQUIRE(c.size() == 0);
    REQUIRE(c.calculateHighestLevel() == 0);
  }
  // setNumLevelsToFit()
  cells.setNumLevelsToFit();
  REQUIRE(cells.numLevels() == 0);
}


TEST_CASE("AdaptiveCells. 1-D, void, 2 levels.", "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::BlockCode<Traits> BlockCode;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Point Point;

  const Point LowerCorner = {{0}};
  const Point Lengths = {{1}};
  const std::size_t NumLevels = 2;
  BlockCode blockCode(LowerCorner, Lengths, NumLevels);
  AdaptiveCells cells(LowerCorner, Lengths, NumLevels);
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(6); // 01 10
    objectCodes.push_back(10); // 10 10
    objectCodes.push_back(14); // 11 10
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 1);
      // Since we are only interested in delimiters, we use the object codes
      // for the objects.
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 4);
      REQUIRE(cells.code(0) == Code(2)); // 00 10
      REQUIRE(cells.code(1) == Code(6)); // 01 10
      REQUIRE(cells.code(2) == Code(10)); // 10 10
      REQUIRE(cells.code(3) == Code(14)); // 11 10
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(1));
      REQUIRE(cells.delimiter(2) == std::size_t(2));
      REQUIRE(cells.delimiter(3) == std::size_t(3));
      REQUIRE(cells.delimiter(4) == std::size_t(4));
    }
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 2);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 2);
      REQUIRE(cells.code(0) == Code(1)); // 00 01
      REQUIRE(cells.code(1) == Code(9)); // 10 01
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(2));
      REQUIRE(cells.delimiter(2) == std::size_t(4));
    }
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 4);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 1);
      REQUIRE(cells.code(0) == Code(0)); // 00 00
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(4));
    }
  }
  {
    std::vector<Code> objectCodes;
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(2); // 00 10
    objectCodes.push_back(6); // 01 10
    objectCodes.push_back(10); // 10 10
    objectCodes.push_back(14); // 11 10
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 1);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 4);
      REQUIRE(cells.code(0) == Code(2)); // 00 10
      REQUIRE(cells.code(1) == Code(6)); // 01 10
      REQUIRE(cells.code(2) == Code(10)); // 10 10
      REQUIRE(cells.code(3) == Code(14)); // 11 10
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(2));
      REQUIRE(cells.delimiter(2) == std::size_t(3));
      REQUIRE(cells.delimiter(3) == std::size_t(4));
      REQUIRE(cells.delimiter(4) == std::size_t(5));
    }
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 2);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 3);
      REQUIRE(cells.code(0) == Code(2)); // 00 10
      REQUIRE(cells.code(1) == Code(6)); // 01 10
      REQUIRE(cells.code(2) == Code(9)); // 10 01
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(2));
      REQUIRE(cells.delimiter(2) == std::size_t(3));
      REQUIRE(cells.delimiter(3) == std::size_t(5));
    }
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 3);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 2);
      REQUIRE(cells.code(0) == Code(1)); // 00 01
      REQUIRE(cells.code(1) == Code(9)); // 10 01
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(3));
      REQUIRE(cells.delimiter(2) == std::size_t(5));
    }
    {
      std::vector<Code> cellCodes;
      coarsen(blockCode, objectCodes, &cellCodes, 5);
      cells.buildCells(cellCodes, objectCodes, objectCodes);
      REQUIRE(cells.size() == 1);
      REQUIRE(cells.code(0) == Code(0)); // 00 00
      REQUIRE(cells.delimiter(0) == std::size_t(0));
      REQUIRE(cells.delimiter(1) == std::size_t(5));
    }
  }
  // Build from code/count pairs.
  {
    typedef std::pair<Code, std::size_t> Pair;
    Code const Guard = Traits::GuardCode;

    cells.buildCells(std::vector<Pair>{{Guard, 0}});
    REQUIRE(cells.size() == 0);
    REQUIRE(cells.delimiter(0) == 0);

    cells.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    REQUIRE(cells.size() == 1);
    REQUIRE(cells.code(0) == Code(0));
    REQUIRE(cells.delimiter(0) == 0);
    REQUIRE(cells.delimiter(1) == 1);

    cells.buildCells(std::vector<Pair>{{0, 10}, {Guard, 0}});
    REQUIRE(cells.size() == 1);
    REQUIRE(cells.code(0) == Code(0));
    REQUIRE(cells.delimiter(0) == 0);
    REQUIRE(cells.delimiter(1) == 10);

    cells.buildCells(std::vector<Pair>{{1, 1}, {9, 1}, {Guard, 0}});
    REQUIRE(cells.size() == 2);
    REQUIRE(cells.code(0) == Code(1));
    REQUIRE(cells.code(1) == Code(9));
    REQUIRE(cells.delimiter(0) == 0);
    REQUIRE(cells.delimiter(1) == 1);
    REQUIRE(cells.delimiter(2) == 2);

    // Convert between cell indices and codes.
    {
      // Four cells at level 2.
      // 00 10, 01 10, 10 10, 11 10
      cells.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1},
                                                             {Guard, 0}});
      {
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i != cells.size() + 1; ++i) {
          REQUIRE(cells.codesToCells(cells.codes(indices)) == indices);
          indices.push_back(i);
        }
      }

      // Cells at levels 1 and 2.
      // 00 01, 10 10, 11 10
      cells.buildCells(std::vector<Pair>{{1, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      {
        std::vector<std::size_t> indices;
        for (std::size_t i = 0; i != cells.size() + 1; ++i) {
          REQUIRE(cells.codesToCells(cells.codes(indices)) == indices);
          indices.push_back(i);
        }
      }
    }
  }
}


TEST_CASE("AdaptiveCells. 1-D, void, 2 levels, merge", "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Point Point;
  typedef std::pair<Code, std::size_t> Pair;
 
  Code const Guard = Traits::GuardCode;
  Point const LowerCorner = {{0}};
  Point const Lengths = {{1}};
  std::size_t const NumLevels = 2;
  AdaptiveCells a(LowerCorner, Lengths, NumLevels);
  AdaptiveCells b(LowerCorner, Lengths, NumLevels);
  AdaptiveCells c(LowerCorner, Lengths, NumLevels);
 
  SECTION("Both empty.") {
    a += b;
    REQUIRE(a.size() == 0);
  }
  SECTION("One is empty. Level 0") {
    a.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    AdaptiveCells const result = a;
    a += b;
    REQUIRE(a == result);
    b += a;
    REQUIRE(b == result);
  }
  SECTION("One is empty. Level 1 followed by level 2.") {
    // 00 01, 10 10 
    a.buildCells(std::vector<Pair>{{1, 1}, {10, 1}, {Guard, 0}});
    AdaptiveCells const result = a;
    a += b;
    REQUIRE(a == result);
    b += a;
    REQUIRE(b == result);
  }
  SECTION("One is empty. Level 2 followed by level 1.") {
    // 00 10, 10 01
    a.buildCells(std::vector<Pair>{{2, 1}, {9, 1}, {Guard, 0}});
    AdaptiveCells const result = a;
    a += b;
    REQUIRE(a == result);
    b += a;
    REQUIRE(b == result);
  }
  SECTION("Level 0. Level 0.") {
    a.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    b.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    c.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
    a += b;
    REQUIRE(a == c);
  }
  SECTION("Level 0. Level 1.") {
    a.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    // 00 01
    b.buildCells(std::vector<Pair>{{1, 1}, {Guard, 0}});
    c.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
    a += b;
    REQUIRE(a == c);
  }
  SECTION("Level 0. Level 1.") {
    // 00 01
    a.buildCells(std::vector<Pair>{{1, 1}, {Guard, 0}});
    b.buildCells(std::vector<Pair>{{0, 1}, {Guard, 0}});
    c.buildCells(std::vector<Pair>{{0, 2}, {Guard, 0}});
    a += b;
    REQUIRE(a == c);
  }
  SECTION("00 01 += 10 10") {
    a.buildCells(std::vector<Pair>{{1, 1}, {Guard, 0}});
    b.buildCells(std::vector<Pair>{{10, 1}, {Guard, 0}});
    c.buildCells(std::vector<Pair>{{1, 1}, {10, 1}, {Guard, 0}});
    a += b;
    REQUIRE(a == c);
  }
  SECTION("01 10 += 10 01") {
    a.buildCells(std::vector<Pair>{{6, 1}, {Guard, 0}});
    b.buildCells(std::vector<Pair>{{9, 1}, {Guard, 0}});
    c.buildCells(std::vector<Pair>{{6, 1}, {9, 1}, {Guard, 0}});
    a += b;
    REQUIRE(a == c);
  }
}


template<bool _StoreDel>
void
testCropVoid()
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef Traits::Code Code;
  typedef Traits::Point Point;
  typedef std::pair<Code, std::size_t> Pair;

  Code const Guard = Traits::GuardCode;
  Point const LowerCorner = {{0}};
  Point const Lengths = {{1}};
  std::size_t const NumLevels = 2;

  {
    typedef stlib::sfc::AdaptiveCells<Traits, void, _StoreDel>
      AdaptiveCells;

    AdaptiveCells a(LowerCorner, Lengths, NumLevels);
    AdaptiveCells b(LowerCorner, Lengths, NumLevels);

    // Empty.
    {
      a.crop(std::vector<std::size_t>{});
      a.checkValidity();
      REQUIRE((a.size() == 0));
    }
    // 00 10, 01 10, 10 10, 11 10
    {
      a.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      a.crop(std::vector<std::size_t>{});
      a.checkValidity();
      REQUIRE((a.size() == 0));

      a.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      a.crop(std::vector<std::size_t>{0});
      a.checkValidity();
      b.buildCells(std::vector<Pair>{{2, 1}, {Guard, 0}});
      REQUIRE((a == b));

      a.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      a.crop(std::vector<std::size_t>{1, 3});
      a.checkValidity();
      b.buildCells(std::vector<Pair>{{6, 1}, {14, 1}, {Guard, 0}});
      REQUIRE((a == b));

      a.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      a.crop(std::vector<std::size_t>{0, 1, 2, 3});
      a.checkValidity();
      b.buildCells(std::vector<Pair>{{2, 1}, {6, 1}, {10, 1}, {14, 1}, {Guard, 0}});
      REQUIRE((a == b));
    }
  }
}


TEST_CASE("AdaptiveCells. crop.", "[AdaptiveCells]")
{
  testCropVoid<false>();
  testCropVoid<true>();
}


template<bool _StoreDel>
void
testCropBBox()
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef Traits::Point Point;
  typedef Traits::BBox BBox;

  Point const LowerCorner = {{0}};
  Point const Lengths = {{1}};
  std::size_t const NumLevels = 2;

  {
    typedef stlib::sfc::AdaptiveCells<Traits, BBox, _StoreDel>
      AdaptiveCells;

    AdaptiveCells a(LowerCorner, Lengths, NumLevels);
    AdaptiveCells b(LowerCorner, Lengths, NumLevels);

    // Empty.
    {
      a.crop(std::vector<std::size_t>{});
      a.checkValidity();
      REQUIRE((a.size() == 0));
    }
    {
      std::vector<Point> objects = {{{0}}, {{0.25}}, {{0.5}}, {{0.75}}};
      std::vector<Point> subset;

      a.buildCells(&objects);
      a.crop(std::vector<std::size_t>{});
      a.checkValidity();
      REQUIRE((a.size() == 0));

      a.buildCells(&objects);
      a.crop(std::vector<std::size_t>{0});
      a.checkValidity();
      {
        std::vector<Point> subset = {{{0}}};
        b.buildCells(&subset);
      }
      REQUIRE((a == b));

      a.buildCells(&objects);
      a.crop(std::vector<std::size_t>{1, 3});
      a.checkValidity();
      {
        std::vector<Point> subset = {{{0.25}}, {{0.75}}};
        b.buildCells(&subset);
      }
      REQUIRE((a == b));

      a.buildCells(&objects);
      a.crop(std::vector<std::size_t>{0, 1, 2, 3});
      a.checkValidity();
      b.buildCells(&objects);
      REQUIRE((a == b));
    }
  }
}


TEST_CASE("AdaptiveCells. crop BBox.", "[AdaptiveCells]")
{
  testCropBBox<false>();
  testCropBBox<true>();
}


TEST_CASE("AdaptiveCells. 1-D, BBox, 20 levels.", "[AdaptiveCells]")
{
  const std::size_t NumLevels = 20;
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
  typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  SECTION("Accessors.")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    REQUIRE(cells.lowerCorner() == Point{{0}});
    REQUIRE(cells.lengths() == Point{{1}});
    REQUIRE(cells.numLevels() == NumLevels);
    REQUIRE(cells.size() == 0);
    REQUIRE(cells.calculateHighestLevel() == 0);

    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());
    REQUIRE(cells.calculateHighestLevel() == cells.numLevels());
    for (std::size_t i = 0; i != cells.size(); ++i) {
      REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) ==
              Code(1));
    }
    // CONTINUE
#if 0
    // Copy without keeping the object delimiters.
    stlib::sfc::AdaptiveCells<Traits, Cell, false> copy(cells);
    REQUIRE(copy.codes() == cells.codes());
    REQUIRE(copy.size() == cells.size());
    for (std::size_t i = 0; i != copy.size(); ++i) {
      REQUIRE(copy[i] == cells[i]);
    }
    REQUIRE(copy.grid() == cells.grid());
#endif
  }

  SECTION("setNumLevelsToFit.")
  {
    {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.setNumLevelsToFit();
      REQUIRE(cells.numLevels() == 0);
    }
    {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      cells.setNumLevelsToFit();
      REQUIRE(cells.numLevels() == NumLevels);
      cells.coarsenWithoutMerging();
      cells.setNumLevelsToFit();
      REQUIRE(cells.numLevels() == 9);
      REQUIRE(cells.grid().levelBits() == 4);
      for (std::size_t i = 0; i != cells.size(); ++i) {
        REQUIRE(cells.grid().level(cells.code(i)) == cells.numLevels());
        REQUIRE((cells.code(i) >> cells.grid().levelBits()) == i);
      }
    }
  }
  
  SECTION("Serialize.")
  {
    AdaptiveCells x(Point{{0}}, Point{{1}}, NumLevels);
    x.buildCells(&objects);
    std::vector<unsigned char> buffer;
    x.serialize(&buffer);
    AdaptiveCells y(Point{{0}}, Point{{1}}, NumLevels);
    y.unserialize(buffer);
    REQUIRE(x == y);
  }

  SECTION("Coarsening with LevelGreaterThan. Level 0.")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    stlib::sfc::LevelGreaterThan pred = {0};
    cells.coarsen(pred);
    REQUIRE(cells.size() == 1);
    REQUIRE(cells.calculateHighestLevel() == 0);
    REQUIRE(cells.delimiter(0) == Code(0));
    REQUIRE(cells.delimiter(cells.size()) == objects.size());
    REQUIRE(cells[0].lower[0] == 0);
    REQUIRE(cells[0].upper[0] ==
            Float(objects.size() - 1) / objects.size());
  }

  SECTION("Coarsening with LevelGreaterThan. Levels 0 through 9.")
  {
    for (std::size_t level = 0; level != 10; ++level) {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      stlib::sfc::LevelGreaterThan pred = {level};
      cells.coarsen(pred);
      const std::size_t NumCells = 1 << level;
      REQUIRE(cells.size() == NumCells);
      REQUIRE(cells.calculateHighestLevel() == level);
      REQUIRE(cells.delimiter(0) == Code(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      REQUIRE(cells[0].lower[0] == 0);
      REQUIRE(cells[0].upper[0] < Float(1) / NumCells);
    }
  }

  SECTION("Coarsening with coarsenCellSize().")
  {
    for (std::size_t size = 1; size <= objects.size(); size *= 2) {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      cells.coarsenCellSize(size);
      const std::size_t NumCells = objects.size() / size;
      REQUIRE(cells.size() == NumCells);
      REQUIRE(cells.delimiter(0) == Code(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      REQUIRE(cells[0].lower[0] == 0);
      REQUIRE(cells[0].upper[0] < Float(1) / NumCells);
    }
  }

  SECTION("Coarsening to a specified number of cells.")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());
    cells.coarsenMaxCells(objects.size() / 2);
    REQUIRE(cells.size() <= objects.size() / 2);
    cells.coarsenMaxCells(std::size_t(1));
    REQUIRE(cells.size() == 1);
    REQUIRE((cells.delimiter(1) - cells.delimiter(0)) == objects.size());
  }

  SECTION("Coarsen without merging.")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());
    cells.coarsenWithoutMerging();
    REQUIRE(cells.size() == objects.size());
  }

  SECTION("Merge with a copy.")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    REQUIRE(cells.size() == objects.size());
    AdaptiveCells copy = cells;
    cells += copy;
    cells.checkValidity();
    REQUIRE(cells.size() == objects.size());
    for (std::size_t i = 0; i != cells.size(); ++i) {
      REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) ==
              Code(2));
    }
  }

  SECTION("Merge separate halves.")
  {
    std::vector<Point> objects1(objects.begin(),
                                objects.begin() + objects.size() / 2);
    std::vector<Point> objects2(objects.begin() + objects.size() / 2,
                                objects.end());
    AdaptiveCells cells1(Point{{0}}, Point{{1}}, NumLevels);
    cells1.buildCells(&objects1);
    REQUIRE(cells1.size() == objects1.size());
    cells1.checkValidity();
    AdaptiveCells cells2(Point{{0}}, Point{{1}}, NumLevels);
    cells2.buildCells(&objects2);
    REQUIRE(cells2.size() == objects2.size());
    cells2.checkValidity();

    cells1 += cells2;
    cells1.checkValidity();
    REQUIRE(cells1.size() == objects.size());
  }
}


TEST_CASE("AdaptiveCells. 1-D, BBox, 20 levels. Objects at 1/2^n.",
          "[AdaptiveCells]")
{
  const std::size_t NumLevels = 20;
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
  typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Point;

  // 1/2^n.
  std::vector<Point> objects(NumLevels + 1);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(1) / (2 << i);
  }

  SECTION("coarsenCellSize()")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    cells.coarsenCellSize(1);
    REQUIRE(cells.size() == objects.size());
    for (std::size_t i = 0; i != cells.size(); ++i) {
      REQUIRE(cells.delimiter(i) == i);
    }
    REQUIRE(cells.delimiter(cells.size()) == objects.size());
  }

  SECTION("Coarsen LevelGreaterThan.")
  {
    for (std::size_t level = 0; level <= NumLevels; ++level) {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      stlib::sfc::LevelGreaterThan pred = {level};
      cells.coarsen(pred);
      REQUIRE(cells.size() == level + 1);
      REQUIRE(cells.delimiter(0) == Code(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      // Check that the bounding boxes contain the objects.
      for (std::size_t i = 0; i != cells.size(); ++i) {
        for (std::size_t j = cells.delimiter(i);
             j != cells.delimiter(i + 1); ++j) {
          REQUIRE(isInside(cells[i], objects[j]));
        }
      }
    }
  }
}


TEST_CASE("AdaptiveCells. 1-D, BBox, 20 levels. Objects at 1 - 1/2^n.",
          "[AdaptiveCells]")
{
  const std::size_t NumLevels = 20;
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
  typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Point;

  // 1 - 1/2^n.
  std::vector<Point> objects(NumLevels);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = 1 - Float(1) / (2 << i);
  }

  SECTION("coarsenCellSize()")
  {
    AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
    cells.buildCells(&objects);
    cells.coarsenCellSize(1);
    REQUIRE(cells.size() == objects.size());
    for (std::size_t i = 0; i != cells.size(); ++i) {
      REQUIRE(cells.delimiter(i) == i);
    }
    REQUIRE(cells.delimiter(cells.size()) == objects.size());
  }

  SECTION("Coarsen LevelGreaterThan.")
  {
    for (std::size_t level = 0; level <= NumLevels; ++level) {
      AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
      cells.buildCells(&objects);
      stlib::sfc::LevelGreaterThan pred = {level};
      cells.coarsen(pred);
      if (level == 0) {
        REQUIRE(cells.size() == 1);
      }
      else {
        REQUIRE(cells.size() == level);
      }
      REQUIRE(cells.delimiter(0) == Code(0));
      REQUIRE(cells.delimiter(cells.size()) == objects.size());
      // Check that the bounding boxes contain the objects.
      for (std::size_t i = 0; i != cells.size(); ++i) {
        for (std::size_t j = cells.delimiter(i); j != cells.delimiter(i + 1);
             ++j) {
          REQUIRE(isInside(cells[i], objects[j]));
        }
      }
    }
  }
}

TEST_CASE("AdaptiveCells. UniformCells constructor.", "[AdaptiveCells]") {
  typedef stlib::sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef stlib::geom::BBox<Traits::Float, Traits::Dimension> Cell;
  typedef stlib::sfc::UniformCells<Traits, Cell, true> UniformCells;
  typedef stlib::sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
  typedef UniformCells::Float Float;
  typedef UniformCells::Point Point;

  // Uniformly-spaced points.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }

  // Build UniformCells.
  UniformCells uniform(Point{{0}}, Point{{1}}, NumLevels);
  uniform.buildCells(&objects);

  // Build AdaptiveCells.
  AdaptiveCells multiLevel(uniform);
  multiLevel.checkValidity();
  REQUIRE(multiLevel.lowerCorner() == uniform.lowerCorner());
  REQUIRE(multiLevel.lengths() == uniform.lengths());
  REQUIRE(multiLevel.numLevels() == uniform.numLevels());
  REQUIRE(multiLevel.size() == uniform.size());
  for (std::size_t i = 0; i != multiLevel.size(); ++i) {
    REQUIRE(multiLevel[i] == uniform[i]);
  }
}


TEST_CASE("AdaptiveCells. Order and restore.",
          "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  const std::size_t NumLevels = 20;
  typedef stlib::sfc::AdaptiveCells<Traits, void, false> AdaptiveCells;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Point;

  // Uniformly-spaced points, shuffled.
  std::vector<Point> objects(512);
  for (std::size_t i = 0; i != objects.size(); ++i) {
    objects[i][0] = Float(i) / objects.size();
  }
  std::random_shuffle(objects.begin(), objects.end());

  std::vector<Point> const originalObjects(objects);
  AdaptiveCells cells(Point{{0}}, Point{{1}}, NumLevels);
  stlib::sfc::OrderedObjects orderedObjects;
  cells.buildCells(&objects, &orderedObjects);
  std::vector<Point> copy(originalObjects);
  orderedObjects.order(copy.begin(), copy.end());
  REQUIRE(copy == objects);
  orderedObjects.restore(copy.begin(), copy.end());
  REQUIRE(copy == originalObjects);
}


TEST_CASE("adaptiveCells()", "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Object;
  using stlib::sfc::adaptiveCells;

  {
    // No objects.
    std::vector<Object> objects;
    AdaptiveCells const cells = adaptiveCells<AdaptiveCells>(&objects);
    REQUIRE(objects.empty());
    REQUIRE(cells.empty());
  }
  {
    // Uniformly-spaced points.
    std::vector<Object> objects(512);
    for (std::size_t i = 0; i != objects.size(); ++i) {
      objects[i][0] = Float(i) / objects.size();
    }

    {
      // Default maximum number of objects per cell.
      AdaptiveCells const cells = adaptiveCells<AdaptiveCells>(&objects);
      REQUIRE(objects.size() == 512);
      REQUIRE_FALSE(cells.empty());
    }
    {
      AdaptiveCells const cells =
        adaptiveCells<AdaptiveCells>(&objects, 64);
      for (std::size_t i = 0; i != cells.size(); ++i) {
        REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) <= 64);
      }
    }
    {
      AdaptiveCells const cells =
        adaptiveCells<AdaptiveCells>(&objects, 1);
      REQUIRE(cells.size() == objects.size());
    }
  }
}


TEST_CASE("adaptiveCells() with OrderedObjects", "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  typedef AdaptiveCells::Float Float;
  typedef AdaptiveCells::Point Object;
  using stlib::sfc::adaptiveCells;

  stlib::sfc::OrderedObjects orderedObjects;
  {
    // No objects.
    std::vector<Object> const original;
    std::vector<Object> objects = original;
    AdaptiveCells const cells =
      adaptiveCells<AdaptiveCells>(&objects, &orderedObjects);
    REQUIRE(objects.empty());
    REQUIRE(cells.empty());
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
      AdaptiveCells const cells =
        adaptiveCells<AdaptiveCells>(&objects, &orderedObjects);
      REQUIRE(objects.size() == 512);
      REQUIRE_FALSE(cells.empty());
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
    {
      std::vector<Object> objects = original;
      AdaptiveCells const cells =
        adaptiveCells<AdaptiveCells>(&objects, &orderedObjects, 64);
      for (std::size_t i = 0; i != cells.size(); ++i) {
        REQUIRE((cells.delimiter(i + 1) - cells.delimiter(i)) <= 64);
      }
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
    {
      std::vector<Object> objects = original;
      AdaptiveCells const cells =
        adaptiveCells<AdaptiveCells>(&objects, &orderedObjects, 1);
      REQUIRE(cells.size() == objects.size());
      orderedObjects.restore(objects.begin(), objects.end());
      REQUIRE(objects == original);
    }
  }
}


TEST_CASE("AdaptiveCells. areCompatible().", "[AdaptiveCells]")
{
  typedef stlib::sfc::Traits<1> Traits;
  typedef stlib::sfc::AdaptiveCells<Traits, void, true> AdaptiveCells;
  //typedef AdaptiveCells::Code Code;
  typedef AdaptiveCells::Point Point;
  using stlib::sfc::areCompatible;

  // Different lower corner.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{-1}}, Point{{1}}, 3);
    REQUIRE_FALSE(areCompatible(a, b));
  }
  // Different extents.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{2}}, 3);
    REQUIRE_FALSE(areCompatible(a, b));
  }
  // Different levels of refinement.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 4);
    REQUIRE_FALSE(areCompatible(a, b));
  }
  // Both empty.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    REQUIRE(areCompatible(a, b));
  }
  // One is empty.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    std::vector<Point> objects = {{{0}}};
    b.buildCells(&objects, 1);
    REQUIRE(areCompatible(a, b));
  }
  // Same level.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    std::vector<Point> objects = {{{0}}, {{0.5}}};
    a.buildCells(&objects, 1);
    b.buildCells(&objects, 1);
    REQUIRE(areCompatible(a, b));
  }
  // Different level.
  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    std::vector<Point> objects = {{{0}}, {{0.5}}};
    a.buildCells(&objects, 1);
    b.buildCells(&objects, 2);
    REQUIRE_FALSE(areCompatible(a, b));
  }

  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    {
      std::vector<Point> objects = {{{0}}, {{0.25}}};
      a.buildCells(&objects, 1);
    }
    {
      std::vector<Point> objects = {{{0.5}}, {{0.75}}};
      b.buildCells(&objects, 1);
    }
    REQUIRE(areCompatible(a, b));
  }

  {
    AdaptiveCells a(Point{{0}}, Point{{1}}, 3);
    AdaptiveCells b(Point{{0}}, Point{{1}}, 3);
    {
      std::vector<Point> objects = {{{0}}};
      a.buildCells(&objects, 1);
    }
    {
      std::vector<Point> objects = {{{0.5}}, {{0.75}}};
      b.buildCells(&objects, 1);
    }
    REQUIRE_FALSE(areCompatible(a, b));
  }
}


