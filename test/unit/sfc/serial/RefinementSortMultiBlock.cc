// -*- C++ -*-

#include "stlib/sfc/RefinementSortMultiBlock.h"
#include "stlib/sfc/AdaptiveCells.h"

using namespace stlib;

int
main()
{
  // 1-D, 0 levels
  {
    typedef sfc::Traits<1> Traits;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef sfc::RefinementSortMultiBlock<Traits>::Pair Pair;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 0);
    {
      std::vector<Pair> pairs;
      sfc::refinementSortMultiBlock(blockCode, &pairs, 0);
      assert(pairs.empty());
    }
    {
      std::vector<Pair> pairs(10);
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = blockCode.code(Point{{0}});
        pairs[i].second = i;
      }
      std::vector<Pair> p(pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, 1);
      assert(p == pairs);
    }
  }
  // 1-D, 1 level
  {
    typedef sfc::Traits<1> Traits;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Code Code;
    typedef BlockCode::Point Point;
    typedef sfc::RefinementSortMultiBlock<Traits>::Pair Pair;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 1);
    {
      std::vector<Pair> pairs;
      sfc::refinementSortMultiBlock(blockCode, &pairs, 0);
      assert(pairs.empty());
    }
    // Put all of the points at the origin.
    {
      std::vector<Pair> pairs(10);
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = blockCode.code(Point{{0}});
        pairs[i].second = i;
      }
      std::vector<Pair> p(pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, 1);
      assert(p == pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, p.size());
      for (std::size_t i = 0; i != p.size(); ++i) {
        assert(p[i].first == 0);
        assert(p[i].second == i);
      }
    }
    // Put all of the points at the midpoint.
    {
      std::vector<Pair> pairs(10);
      Code const code = blockCode.code(Point{{0.5}});
      assert(code == 3);
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = code;
        pairs[i].second = i;
      }
      std::vector<Pair> p(pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, 1);
      assert(p == pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, p.size());
      for (std::size_t i = 0; i != p.size(); ++i) {
        assert(p[i].first == 0);
        assert(p[i].second == i);
      }
    }
  }

  // 1-D, 4 levels
  {
    typedef sfc::Traits<1> Traits;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef BlockCode::Float Float;
    typedef sfc::RefinementSortMultiBlock<Traits>::Pair Pair;

    BlockCode blockCode(Point{{0}}, Point{{1}}, 4);
    {
      std::vector<Pair> pairs;
      {
        std::size_t const Extent = 1 << blockCode.numLevels();
        for (std::size_t i = 0; i != Extent; ++i) {
          pairs.push_back(Pair{blockCode.code(Point{{Float(i) / Extent}}),
                i});
        }
      }
      std::vector<Pair> p(pairs);
#if 0
      for (std::size_t i = 0; i != p.size(); ++i) {
        numerical::printBits(std::cerr, p[i].first);
        std::cerr << ' ' << p[i].second << '\n';
      }
#endif
      sfc::refinementSortMultiBlock(blockCode, &p, 1);
      assert(p.size() == pairs.size());
#if 0
      for (std::size_t i = 0; i != p.size(); ++i) {
        numerical::printBits(std::cerr, p[i].first);
        std::cerr << ' ' << p[i].second << '\n';
      }
#endif
      auto f = [](Pair const& a, Pair const& b){return a.first < b.first;};
      assert(std::is_sorted(p.begin(), p.end(), f));
    }
  }

  // 3-D, 0 levels
  {
    typedef sfc::Traits<3> Traits;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef sfc::RefinementSortMultiBlock<Traits>::Pair Pair;

    BlockCode blockCode(Point{{0, 0, 0}}, Point{{1, 1, 1}}, 0);
    {
      std::vector<Pair> pairs;
      sfc::refinementSortMultiBlock(blockCode, &pairs, 0);
      assert(pairs.empty());
    }
    {
      std::vector<Pair> pairs(10);
      for (std::size_t i = 0; i != pairs.size(); ++i) {
        pairs[i].first = blockCode.code(Point{{0, 0, 0}});
        pairs[i].second = i;
      }
      std::vector<Pair> p(pairs);
      sfc::refinementSortMultiBlock(blockCode, &p, 1);
      assert(p == pairs);
    }
  }

  // 3-D, 4 levels
  {
    typedef sfc::Traits<3> Traits;
    typedef geom::BBox<Traits::Float, Traits::Dimension> Cell;
    typedef sfc::AdaptiveCells<Traits, Cell, true> AdaptiveCells;
    typedef sfc::BlockCode<Traits> BlockCode;
    typedef BlockCode::Point Point;
    typedef BlockCode::Float Float;
    typedef sfc::RefinementSortMultiBlock<Traits>::Pair Pair;

    BlockCode blockCode(Point{{0, 0, 0}}, Point{{1, 1, 1}}, 4);
    {
      std::vector<Pair> pairs;
      {
        std::size_t n = 0;
        std::size_t const Extent = 1 << blockCode.numLevels();
        for (std::size_t k = 0; k != Extent; ++k) {
          for (std::size_t j = 0; j != Extent; ++j) {
            for (std::size_t i = 0; i != Extent; ++i) {
              pairs.push_back(Pair{blockCode.code(
                    Point{{Float(i) / Extent,
                          Float(j) / Extent,
                          Float(k) / Extent}}), n++});
            }
          }
        }
      }
      for (std::size_t n = 0; n <= 4; ++n) {
        std::vector<Pair> p(pairs);
        std::size_t const maxElementsPerCell = std::size_t(1) << (3 * n);
        sfc::refinementSortMultiBlock(blockCode, &p, maxElementsPerCell);
        assert(p.size() == pairs.size());
#if 0
        std::cerr << "maxElementsPerCell = " << maxElementsPerCell << '\n';
        if (! std::is_sorted(p.begin(), p.end())) {
          for (std::size_t i = 0; i != p.size(); ++i) {
            numerical::printBits(std::cerr, p[i].first);
            std::cerr << ' ' << p[i].second << '\n';
          }
        }
#endif
        auto f = [](Pair const& a, Pair const& b){return a.first < b.first;};
        assert(std::is_sorted(p.begin(), p.end(), f));
        for (std::size_t i = 0; i != p.size(); ++i) {
          assert(blockCode.level(p[i].first) == 4 - n);
        }
      }
    }
    // Sort objects.
    {
      std::vector<Point> objects;
      {
        std::size_t const Extent = 8;
        for (std::size_t k = 0; k != Extent; ++k) {
          for (std::size_t j = 0; j != Extent; ++j) {
            for (std::size_t i = 0; i != Extent; ++i) {
              objects.push_back(Point{{Float(i) / Extent,
                      Float(j) / Extent,
                      Float(k) / Extent}});
            }
          }
        }
      }
      {
        std::vector<Point> obj(objects);
        std::vector<Pair> codeIndexPairs;
        sfc::refinementSortMultiBlock(blockCode, &obj, &codeIndexPairs, 1);
        assert(codeIndexPairs.size() == objects.size());
        auto f = [](Pair const& a, Pair const& b){return a.first < b.first;};
        assert(std::is_sorted(codeIndexPairs.begin(), codeIndexPairs.end(), f));
      }
      {
        AdaptiveCells cells(blockCode);
        std::vector<Point> obj(objects);
        cells.buildCells(&obj, 1);
        cells.checkValidity();
        assert(cells.size() == objects.size());
      }
      {
        AdaptiveCells cells(blockCode);
        std::vector<Point> obj(objects);
        cells.buildCells(&obj, 8);
        cells.checkValidity();
        assert(cells.size() == objects.size() / 8);
      }
      {
        AdaptiveCells cells(blockCode);
        std::vector<Point> obj(objects);
        cells.buildCells(&obj, objects.size());
        cells.checkValidity();
        assert(cells.size() == 1);
      }
    }
  }
}
