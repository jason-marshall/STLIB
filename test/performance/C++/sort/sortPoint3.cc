// -*- C++ -*-

#include "stlib/performance/SimpleTimer.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <vector>
#include <random>


using Float = float;
std::size_t constexpr D = 3;
using Point = std::array<Float, D>;


template<std::size_t N>
struct lessFixed {
  bool
  operator()(Point const& a, Point const& b) const
  {
    return a[N] < b[N];
  }
};

struct lessDynamic {
  std::size_t n;
  
  bool
  operator()(Point const& a, Point const& b) const
  {
    return a[n] < b[n];
  }
};


int
main()
{
  std::mt19937_64 engine;
  std::uniform_real_distribution<Float> distribution(0, 1);
  
  std::vector<Point> points(1 << 20);
  for (auto&& p: points) {
    for (auto&& x: p) {
      x = distribution(engine);
    }
  }

  stlib::performance::SimpleTimer timer;
  Float meaningless = 0;

  std::cout << "Direct sorting:\n";
  {
    std::vector<Point> p(points);
    timer.start();
    std::sort(p.begin(), p.end(), [](Point const& a, Point const& b)
              {return a[0] < b[0];});
    timer.stop();
    std::cout << "lamda = " << timer.elapsed() << ".\n";
    meaningless += p[0][0];
  }
  {
    std::vector<Point> p(points);
    std::size_t d = 0;
    timer.start();
    std::sort(p.begin(), p.end(), [d](Point const& a, Point const& b)
              {return a[d] < b[d];});
    timer.stop();
    std::cout << "lamda, capture d = " << timer.elapsed() << ".\n";
    meaningless += p[0][0];
  }
  {
    std::vector<Point> p(points);
    timer.start();
    std::sort(p.begin(), p.end(), lessFixed<0>{});
    timer.stop();
    std::cout << "lessFixed = " << timer.elapsed() << ".\n";
    meaningless += p[0][0];
  }
  {
    std::vector<Point> p(points);
    timer.start();
    std::sort(p.begin(), p.end(), lessDynamic{0});
    timer.stop();
    std::cout << "lessDynamic = " << timer.elapsed() << ".\n";
    meaningless += p[0][0];
  }
  
  std::cout << "\nIndirect sorting:\n";
  {
    std::vector<Point> p(points);
    timer.start();
    std::vector<Point const*> ptr(points.size());
    for (std::size_t i = 0; i != ptr.size(); ++i) {
      ptr[i] = &points[i];
    }
    timer.stop();
    double const pointer = timer.elapsed();

    timer.start();
    std::sort(ptr.begin(), ptr.end(), [](Point const* a, Point const* b)
              {return a[0] < b[0];});
    timer.stop();
    double const sort = timer.elapsed();

    timer.start();
    std::vector<Point> c(points.size());
    for (std::size_t i = 0; i != c.size(); ++i) {
      c[i] = *ptr[i];
    }
    p.swap(c);
    timer.stop();
    double const order = timer.elapsed();

    std::cout << "By pointer = " << pointer + sort + order << ".\n"
              << "  pointer = " << pointer << '\n'
              << "  sort = " << sort << '\n'
              << "  order = " << order << '\n';
    meaningless += p[0][0];
  }
  {
    std::vector<Point> p(points);
    timer.start();
    std::vector<std::size_t> indices(points.size());
    for (std::size_t i = 0; i != indices.size(); ++i) {
      indices[i] = i;
    }
    timer.stop();
    double const timeIndices = timer.elapsed();

    timer.start();
    std::sort(indices.begin(), indices.end(),
              [&p](std::size_t a, std::size_t b){return p[a] < p[b];});
    timer.stop();
    double const timeSort = timer.elapsed();

    timer.start();
    std::vector<Point> c(points.size());
    for (std::size_t i = 0; i != c.size(); ++i) {
      c[i] = p[indices[i]];
    }
    p.swap(c);
    timer.stop();
    double const timeOrder = timer.elapsed();

    std::cout << "By indices = " << timeIndices + timeSort + timeOrder << ".\n"
              << "  indices = " << timeIndices << '\n'
              << "  sort = " << timeSort << '\n'
              << "  order = " << timeOrder << '\n';
    meaningless += p[0][0];
  }
  {
    using Pair = std::pair<Point, Point const*>;
    std::vector<Point> p(points);
    timer.start();
    std::vector<Pair> pairs(points.size());
    for (std::size_t i = 0; i != pairs.size(); ++i) {
      pairs[i].first = p[i];
      pairs[i].second = &p[i];
    }
    timer.stop();
    double const timePairs = timer.elapsed();

    timer.start();
    std::sort(pairs.begin(), pairs.end(), [](Pair const& a, Pair const& b)
              {return a.first[0] < b.first[0];});
    timer.stop();
    double const timeSort = timer.elapsed();

    timer.start();
    for (std::size_t i = 0; i != p.size(); ++i) {
      p[i] = *pairs[i].second;
    }
    timer.stop();
    double const timeOrder = timer.elapsed();

    std::cout << "By point/pointer pair = "
              << timePairs + timeSort + timeOrder << ".\n"
              << "  pairs = " << timePairs << '\n'
              << "  sort = " << timeSort << '\n'
              << "  order = " << timeOrder << '\n';
    meaningless += p[0][0];
  }

  std::cout << "Meaningless result = " << meaningless << ".\n";
  return 0;
}
