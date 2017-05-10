// -*- C++ -*-

#include "stlib/cpt/Grid.h"

#include "stlib/container/MultiArray.h"

#include <iostream>

using namespace stlib;
using namespace cpt;

int
main()
{
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  // 3-D
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  {
    typedef Grid<3> Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;

    // Construct from the grids.
    SizeList extents = {{2, 2, 2}};
    IndexList bases = {{}};
    double d[2 * 2 * 2];
    double gd[2 * 2 * 2 * 3];
    double cp[2 * 2 * 2 * 3];
    int cf[2 * 2 * 2];
    std::cout << "Grid<3>() = " << '\n' << Grid() << '\n';
    std::cout << "Grid<3>(extents, bases, false, false, false) = \n"
              << Grid(extents, bases, false, false, false) << '\n';
    std::cout << "Grid<3>(extents, bases, true, true, true) = \n"
              << Grid(extents, bases, true, true, true) << '\n';
    std::cout << "Grid<3>(extents, bases, 0, 0, 0, 0) = \n"
              << Grid(extents, bases, 0, 0, 0, 0) << '\n';
    std::cout << "Grid<3>(extents, bases, d, gd, cp, cf) = \n"
              << Grid(extents, bases, d, gd, cp, cf) << '\n';
  }
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  // 2-D
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  {
    typedef Grid<2> Grid;
    typedef Grid::SizeList SizeList;
    typedef Grid::IndexList IndexList;

    // Construct from the grids.
    SizeList extents = {{2, 2}};
    IndexList bases = {{}};
    double d[2 * 2];
    double gd[2 * 2 * 3];
    double cp[2 * 2 * 3];
    int cf[2 * 2];
    std::cout << "Grid<2>() = " << '\n' << Grid() << '\n';
    std::cout << "Grid<2>(extents, bases, false, false, false) = \n"
              << Grid(extents, bases, false, false, false) << '\n';
    std::cout << "Grid<2>(extents, bases, true, true, true) = \n"
              << Grid(extents, bases, true, true, true) << '\n';
    std::cout << "Grid<2>(extents, bases, 0, 0, 0, 0) = \n"
              << Grid(extents, bases, 0, 0, 0, 0) << '\n';
    std::cout << "Grid<2>(extents, bases, d, gd, cp, cf) = \n"
              << Grid(extents, bases, d, gd, cp, cf) << '\n';
  }

  // CONTINUE
#if 0
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  // 1-D
  //--------------------------------------------------------------------------
  //--------------------------------------------------------------------------
  {
    // Default constructor
    Grid<1> grid;
    std::cout << "Grid<1>() = " << '\n' << grid << '\n';
  } {
    typedef Grid<1> Grid;
    typedef Grid::Point Point;
    typedef Grid::Index Index;
    typedef Grid::BBox BBox;

    // Construct from grid information.
    Index extents(2);
    BBox domain(Point(0.0), Point(1.0));
    ads::Array<1, double> distance(extents);
    ads::Array<1, Point> gradientOfDistance(extents);
    ads::Array<1, Point> closestPoint(extents);
    ads::Array<1, int> closestFace(extents);

    Grid grid(domain, &distance, &gradientOfDistance, &closestPoint,
              &closestFace);
    std::cout
        << "Grid<1>(distance, gradientOfDistance, closestPoint, closestFace) = "
        << '\n' << grid << '\n';
  } {
    typedef Grid<1> Grid;
    typedef Grid::Point Point;
    typedef Grid::Index Index;
    typedef Grid::BBox BBox;

    // Construct from grid information, only distance.
    Index extents(2);
    BBox domain(Point(0.0), Point(1.0));
    ads::Array<1, double> distance(extents);
    ads::Array<1, Point> gradientOfDistance;
    ads::Array<1, Point> closestPoint;
    ads::Array<1, int> closestFace;

    Grid grid(domain, &distance, &gradientOfDistance, &closestPoint,
              &closestFace);

    std::cout << "only distance = "
              << '\n' << grid << '\n';
  }
#endif

  return 0;
}
