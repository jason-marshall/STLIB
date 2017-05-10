// -*- C++ -*-

#include "stlib/geom/tree/BBoxTree.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  //
  // 1-D
  //
  {
    std::cout << "1-D\n"
              << "--------------------------------------------------------\n";

    typedef geom::BBoxTree<1> BBT;
    typedef BBT::Point Point;
    typedef BBT::BBox BB;

    std::vector<BB> boxes;

    {
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 0);
      assert(x.isEmpty());
      x.checkValidity();
      std::vector<int> inter;
      x.computePointQuery(std::back_inserter(inter), Point{{0.}});
      assert(inter.size() == 0);
      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0}}, Point{{1}}});
      assert(inter.size() == 0);
      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0.}});
      assert(inter.size() == 0);
      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }

    {
      boxes.push_back(BB{Point{{0}}, Point{{1}}});
      boxes.push_back(BB{Point{{1}}, Point{{2}}});
      boxes.push_back(BB{Point{{2}}, Point{{3}}});
      boxes.push_back(BB{Point{{3}}, Point{{4}}});
      boxes.push_back(BB{Point{{4}}, Point{{5}}});
      boxes.push_back(BB{Point{{5}}, Point{{6}}});
      boxes.push_back(BB{Point{{6}}, Point{{7}}});
      boxes.push_back(BB{Point{{7}}, Point{{8}}});
      boxes.push_back(BB{Point{{8}}, Point{{9}}});
      boxes.push_back(BB{Point{{9}}, Point{{10}}});
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 10);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1.}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0.}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{0.5}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1.}});
      assert(inter.size() == 2);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{-2}}, Point{{-1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{0.1}}, Point{{0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{1}}, Point{{9}}});
      assert(inter.size() == 10);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{-1.}});
      assert(inter.size() == 1);
      assert(inter[0] == 0);
      inter.clear();


      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1.}});
      assert(inter.size() == 2);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{10.}});
      assert(inter.size() == 1);
      inter.clear();

      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }

    {
      boxes.clear();
      boxes.push_back(BB{Point{{9}}, Point{{10}}});
      boxes.push_back(BB{Point{{8}}, Point{{9}}});
      boxes.push_back(BB{Point{{7}}, Point{{8}}});
      boxes.push_back(BB{Point{{6}}, Point{{7}}});
      boxes.push_back(BB{Point{{5}}, Point{{6}}});
      boxes.push_back(BB{Point{{4}}, Point{{5}}});
      boxes.push_back(BB{Point{{3}}, Point{{4}}});
      boxes.push_back(BB{Point{{2}}, Point{{3}}});
      boxes.push_back(BB{Point{{1}}, Point{{2}}});
      boxes.push_back(BB{Point{{0}}, Point{{1}}});
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 10);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1.}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0.}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{0.5}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1.}});
      assert(inter.size() == 2);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{-2}}, Point{{-1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{0.1}}, Point{{0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter), BB{Point{{1}}, Point{{9}}});
      assert(inter.size() == 10);

      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }
  }


  //
  // 2-D
  //
  {
    std::cout << "2-D\n"
              << "--------------------------------------------------------\n";

    typedef geom::BBoxTree<2> BBT;
    typedef BBT::Number Number;
    typedef BBT::Point Point;
    typedef BBT::BBox BB;

    std::vector<BB> boxes;

    {
      boxes.clear();
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 0);
      assert(x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;
      x.computePointQuery(std::back_inserter(inter), Point{{0, 0}});
      assert(inter.size() == 0);
      x.computeWindowQuery(std::back_inserter(inter), BB{{{0, 0}}, {{1, 1}}});
      assert(inter.size() == 0);
      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }

    {
      boxes.clear();
      for (Number i = 0; i != 4; ++i) {
        for (Number j = 0; j != 4; ++j) {
          boxes.push_back(BB{{{i, j}}, {{i + 1, j + 1}}});
        }
      }
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 16);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1, -1}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0.5, 0.5}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 0.5}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1}});
      assert(inter.size() == 4);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{-2, -2}}, Point{{-1, -1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0.1, 0.1}}, Point{{0.9, 0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{1, 1}}, Point{{3, 3}}});
      assert(inter.size() == 16);
      inter.clear();


      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0, 0}});
      assert(inter.size() == 3);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1, 1}});
      assert(inter.size() == 8);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{2, 2}});
      assert(inter.size() == 12);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0.5, 0.5}});
      assert(inter.size() == 4);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1.5, 1.5}});
      assert(inter.size() == 9);
      inter.clear();

      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }
    {
      boxes.clear();
      for (Number i = 0; i != 100; ++i) {
        for (Number j = 0; j != 100; ++j) {
          boxes.push_back(BB{{{i, j}}, {{i + 1, j + 1}}});
        }
      }
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 10000);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1, -1}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0.5, 0.5}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 0.5}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1}});
      assert(inter.size() == 4);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{-2, -2}}, Point{{-1, -1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0.1, 0.1}}, Point{{0.9, 0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{1, 1}}, Point{{99, 99}}});
      assert(inter.size() == 10000);
      inter.clear();


      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0, 0}});
      assert(inter.size() == 3);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1, 1}});
      assert(inter.size() == 8);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{2, 2}});
      assert(inter.size() == 12);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0.5, 0.5}});
      assert(inter.size() == 4);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1.5, 1.5}});
      assert(inter.size() == 9);
      inter.clear();

      std::cout << "size = " << x.getSize()
                << ", memory usage = " << x.getMemoryUsage() << "\n\n";
    }
  }



  //
  // 3-D
  //
  {
    std::cout << "3-D\n"
              << "--------------------------------------------------------\n";

    typedef geom::BBoxTree<3> BBT;
    typedef BBT::Number Number;
    typedef BBT::Point Point;
    typedef BBT::BBox BB;

    std::vector<BB> boxes;

    {
      boxes.clear();
      BBT x(boxes.begin(), boxes.end());
      std::cout << x;
      assert(x.getSize() == 0);
      assert(x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;
      x.computePointQuery(std::back_inserter(inter), Point{{0, 0, 0}});
      assert(inter.size() == 0);
      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0, 0, 0}}, Point{{1, 1, 1}}});
      assert(inter.size() == 0);
      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }

    {
      boxes.clear();
      for (Number i = 0; i != 4; ++i) {
        for (Number j = 0; j != 4; ++j) {
          for (Number k = 0; k != 4; ++k) {
            boxes.push_back(BB{{{i, j, k}}, {{i + 1, j + 1, k + 1}}});
          }
        }
      }
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 64);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1, -1, -1}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0, 0, 0}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 0, 0}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1.5, 1.5}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1, 1.5}});
      assert(inter.size() == 4);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1, 1}});
      assert(inter.size() == 8);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{-2, -2, -2}}, Point{{-1, -1, -1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0.1, 0.1, 0.1}}, Point{{0.9, 0.9, 0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{1, 1, 1}}, Point{{3, 3, 3}}});
      assert(inter.size() == 64);
      inter.clear();


      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0, 0, 0}});
      // There are 7 boxes within sqrt(2).
      // 1 at distance of 0.
      // 3 at distance of 1.
      // 3 at distance of sqrt(2).
      assert(inter.size() == 7);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1, 1, 1}});
      // There are 26 boxes within sqrt(2).
      // 8 at a distance of 0.
      // 4*3 at a distance of 1.
      // 2*3 at a distance of sqrt(2).
      assert(inter.size() == 26);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{2, 2, 2}});
      // There are 56 boxes within sqrt(2).
      // 8 at a distance of 0.
      // 4*6 at a distance of 1.
      // 2*12 at a distance of sqrt(2).
      assert(inter.size() == 56);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter),
                                    Point{{0.5, 0.5, 0.5}});
      // There are 8 boxes within sqrt(3) / 2.
      // 1 at a distance of 0.
      // 3 at a distance of 1 / 2.
      // 3 at a distance of sqrt(2) / 2.
      // 1 at a distance of sqrt(3) / 2.
      assert(inter.size() == 8);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter),
                                    Point{{1.5, 1.5, 1.5}});
      // There are 27 boxes within sqrt(3) / 2.
      // 1 at a distance of 0.
      // 6 at a distance of 1 / 2.
      // 12 at a distance of sqrt(2) / 2.
      // 8 at a distance of sqrt(3) / 2.
      assert(inter.size() == 27);
      inter.clear();

      std::cout << x << "memory usage = " << x.getMemoryUsage() << "\n\n";
    }
    {
      boxes.clear();
      for (Number i = 0; i != 10; ++i) {
        for (Number j = 0; j != 10; ++j) {
          for (Number k = 0; k != 10; ++k) {
            boxes.push_back(BB{{{i, j, k}}, {{i + 1, j + 1, k + 1}}});
          }
        }
      }
      BBT x(boxes.begin(), boxes.end());
      assert(x.getSize() == 1000);
      assert(!x.isEmpty());
      x.checkValidity();

      std::vector<int> inter;

      x.computePointQuery(std::back_inserter(inter), Point{{-1, -1, -1}});
      assert(inter.size() == 0);

      x.computePointQuery(std::back_inserter(inter), Point{{0, 0, 0}});
      assert(inter.size() == 1);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 0, 0}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1.5, 1.5}});
      assert(inter.size() == 2);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1, 1.5}});
      assert(inter.size() == 4);
      inter.clear();

      x.computePointQuery(std::back_inserter(inter), Point{{1, 1, 1}});
      assert(inter.size() == 8);
      inter.clear();


      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{-2, -2, -2}}, Point{{-1, -1, -1}}});
      assert(inter.size() == 0);

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{0.1, 0.1, 0.1}}, Point{{0.9, 0.9, 0.9}}});
      assert(inter.size() == 1);
      inter.clear();

      x.computeWindowQuery(std::back_inserter(inter),
                           BB{Point{{1, 1, 1}}, Point{{9, 9, 9}}});
      assert(inter.size() == 1000);
      inter.clear();


      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{0, 0, 0}});
      // There are 7 boxes within sqrt(2).
      // 1 at distance of 0.
      // 3 at distance of 1.
      // 3 at distance of sqrt(2).
      assert(inter.size() == 7);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{1, 1, 1}});
      // There are 26 boxes within sqrt(2).
      // 8 at a distance of 0.
      // 4*3 at a distance of 1.
      // 2*3 at a distance of sqrt(2).
      assert(inter.size() == 26);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter), Point{{2, 2, 2}});
      // There are 56 boxes within sqrt(2).
      // 8 at a distance of 0.
      // 4*6 at a distance of 1.
      // 2*12 at a distance of sqrt(2).
      assert(inter.size() == 56);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter),
                                    Point{{0.5, 0.5, 0.5}});
      // There are 8 boxes within sqrt(3) / 2.
      // 1 at a distance of 0.
      // 3 at a distance of 1 / 2.
      // 3 at a distance of sqrt(2) / 2.
      // 1 at a distance of sqrt(3) / 2.
      assert(inter.size() == 8);
      inter.clear();

      x.computeMinimumDistanceQuery(std::back_inserter(inter),
                                    Point{{1.5, 1.5, 1.5}});
      // There are 27 boxes within sqrt(3) / 2.
      // 1 at a distance of 0.
      // 6 at a distance of 1 / 2.
      // 12 at a distance of sqrt(2) / 2.
      // 8 at a distance of sqrt(3) / 2.
      assert(inter.size() == 27);
      inter.clear();

      std::cout << "size = " << x.getSize()
                << ", memory usage = " << x.getMemoryUsage() << "\n\n";
    }
  }

  return 0;
}

