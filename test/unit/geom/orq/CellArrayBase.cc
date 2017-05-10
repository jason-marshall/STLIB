// -*- C++ -*-

#include "stlib/geom/orq/CellArrayBase.h"

#include "stlib/ads/functor/Dereference.h"

#include <iostream>
#include <functional>

#include <cassert>

using namespace stlib;

template<typename _Float>
void
test()
{
  {
    const std::size_t N = 3;
    typedef std::array<_Float, N> Value;
    typedef std::vector<Value> ValueContainer;
    typedef typename ValueContainer::const_iterator Record;
    typedef geom::CellArrayBase<N, ads::Dereference<Record> > ORQDS;
    typedef typename ORQDS::Point Point;
    typedef typename ORQDS::BBox BBox;

    //
    // Constructors
    //

    {
      // Construct from grid dimensions and Cartesian domain.
      ORQDS x(Point{{0.101, 0.101, 0.101}},
              BBox{{{0., 0., 0.}}, {{1., 1., 1.}}});
      std::cout << "CellArrayBase((0.101,0.101,0.101), (0,0,0,1,1,1)) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(x.getDomain() == (BBox{{{0., 0., 0.}}, {{1., 1., 1.}}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{10, 10, 10}}));
      assert(x.getInverseCellSizes() == (Point{{10., 10., 10.}}));
    }
    {
      // Construct from grid dimensions and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      ORQDS x(Point{{0.1, 0.1, 0.1}}, data.begin(), data.end());
      std::cout << "CellArrayBase((0.1,0.1,0.1)) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(! isInside(x.getDomain(), Point{{1., 1., 1.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 1, 1}}));
      assert(x.getInverseCellSizes() == (Point{{0., 0., 0.}}));
    }
    {
      // Construct from grid dimensions and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{1., 1., 1.}});
      ORQDS x(Point{{0.1, 0.1, 0.1}}, data.begin(), data.end());
      std::cout << "CellArrayBase((0.1,0.1,0.1)) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{1., 1., 1.}}));
      for (std::size_t n = 0; n != N; ++n) {
        assert(x.getExtents()[n] >= 10);
        assert(x.getInverseCellSizes()[n] >= 10);
      }
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      ORQDS x(1, data.begin(), data.end());
      std::cout << "CellArrayBase(1) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(! isInside(x.getDomain(), Point{{1., 1., 1.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 1, 1}}));
      assert(x.getInverseCellSizes() == (Point{{0., 0., 0.}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{1., 1., 1.}});
      ORQDS x(1, data.begin(), data.end());
      std::cout << "CellArrayBase(1) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{1., 1., 1.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 1, 1}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{1., 1., 1.}});
      ORQDS x(1000, data.begin(), data.end());
      std::cout << "CellArrayBase(1000) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{1., 1., 1.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{10, 10, 10}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{1., 2., 3.}});
      ORQDS x(6, data.begin(), data.end());
      std::cout << "CellArrayBase(6) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{1., 2., 3.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 2, 3}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{1., 10., 100.}});
      ORQDS x(1000, data.begin(), data.end());
      std::cout << "CellArrayBase(1000) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{1., 10., 100.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 10, 100}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{100., 10., 1.}});
      ORQDS x(1000, data.begin(), data.end());
      std::cout << "CellArrayBase(1000) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{100., 10., 1.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{100, 10, 1}}));
    }
    {
      // Construct from number of cells and a range of records.
      ValueContainer data;
      data.push_back(Value{{0., 0., 0.}});
      data.push_back(Value{{0., 10., 100.}});
      ORQDS x(1000, data.begin(), data.end());
      std::cout << "CellArrayBase(1000) = \n"
                << x << '\n';
      assert(x.size() == 0);
      assert(x.empty());
      assert(isInside(x.getDomain(), Point{{0., 0., 0.}}));
      assert(isInside(x.getDomain(), Point{{0., 10., 100.}}));
      assert(x.getExtents() == (std::array<std::size_t, N>{{1, 10, 100}}));
    }

    //
    // Accesors: Cartesian box
    //

    {
      ORQDS x(Point{{0.1, 0.1, 0.1}},
              BBox{{{1., 2., 3.}}, {{2., 3., 5.}}});
      assert(x.getDomain() ==
             (BBox{{{1., 2., 3.}}, {{2., 3., 5.}}}));
    }
  }
}

int
main()
{
  test<float>();
  test<double>();

  return 0;
}
