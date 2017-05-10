// -*- C++ -*-

#include "stlib/cpt/BRep.h"

#include <iostream>

using namespace stlib;

int
main()
{
  {
    typedef cpt::BRep<3> BRep;
    typedef BRep::Point Point;
    typedef BRep::BBox BBox;
    typedef std::array<std::size_t, 3> IndexedSimplex;
    {
      // Default constructor
      BRep brep;
      std::cout << "BRep() = " << '\n' << brep << '\n';
    }

    //
    // Data for a cube
    //
    std::vector<Point> vertices;
    vertices.push_back(Point{{0, 0, 0}});
    vertices.push_back(Point{{1, 0, 0}});
    vertices.push_back(Point{{1, 1, 0}});
    vertices.push_back(Point{{0, 1, 0}});
    vertices.push_back(Point{{0, 0, 1}});
    vertices.push_back(Point{{1, 0, 1}});
    vertices.push_back(Point{{1, 1, 1}});
    vertices.push_back(Point{{0, 1, 1}});
    std::vector<IndexedSimplex> simplices;
    simplices.push_back(IndexedSimplex{{0, 3, 1}});
    simplices.push_back(IndexedSimplex{{1, 3, 2}});
    simplices.push_back(IndexedSimplex{{0, 1, 4}});
    simplices.push_back(IndexedSimplex{{4, 1, 5}});
    simplices.push_back(IndexedSimplex{{1, 6, 5}});
    simplices.push_back(IndexedSimplex{{1, 2, 6}});
    simplices.push_back(IndexedSimplex{{2, 3, 6}});
    simplices.push_back(IndexedSimplex{{3, 7, 6}});
    simplices.push_back(IndexedSimplex{{0, 4, 3}});
    simplices.push_back(IndexedSimplex{{3, 4, 7}});
    simplices.push_back(IndexedSimplex{{4, 6, 7}});
    simplices.push_back(IndexedSimplex{{4, 5, 6}});
    {
      // Make from vertices and faces without clipping.
      BRep cube;
      cube.make(vertices, simplices);
      assert(cube.vertices.size() == vertices.size());
      assert(cube.indexedSimplices.size() == simplices.size());
      assert(cube.getFaceIdentifierUpperBound() == simplices.size());
    }
    {
      // Construct from vertices and faces with clipping.
      BRep cube(vertices, simplices,
                BBox{{{0, 0, 0}}, {{1, 1, 1}}}, 0.1);
      assert(cube.vertices.size() == vertices.size());
      assert(cube.indexedSimplices.size() == simplices.size());
      assert(cube.getFaceIdentifierUpperBound() == simplices.size());
    }
    // CONTINUE Write more tests.
  }


  //
  // 2-D
  //
  {
    typedef cpt::BRep<2> BRep;
    typedef BRep::Point Point;
    typedef BRep::BBox BBox;
    typedef std::array<std::size_t, 2> IndexedSimplex;
    {
      // Default constructor
      BRep brep;
    }
    {
      // Construct from vertices and faces.
      std::vector<Point> vertices;
      vertices.push_back(Point{{0.25, 0.25}});
      vertices.push_back(Point{{0.75, 0.25}});
      vertices.push_back(Point{{0.75, 0.75}});
      vertices.push_back(Point{{0.25, 0.75}});
      std::vector<IndexedSimplex> faces;
      faces.push_back(IndexedSimplex{{0, 1}});
      faces.push_back(IndexedSimplex{{1, 2}});
      faces.push_back(IndexedSimplex{{2, 3}});
      faces.push_back(IndexedSimplex{{3, 0}});
      BRep brep(vertices, faces, BBox{{{0, 0}}, {{1, 1}}}, 1.);

      assert(brep.vertices.size() == vertices.size());
      assert(brep.indexedSimplices.size() == faces.size());
    }
  }

  // CONTINUE
#if 0
  //
  // 1-D
  //
  {
    typedef cpt::BRep<1> BRep;
    typedef BRep::BBox BBox;
    typedef BRep::Point Point;
    {
      // Default constructor
      BRep brep;
      assert(brep.isEmpty());
      assert(brep.getSimplicesSize() == 0);
    }
    {
      // Construct from vertices and faces.
      std::vector<double> locations;
      locations.push_back(0.0);
      locations.push_back(0.25);
      locations.push_back(0.5);
      locations.push_back(0.75);
      locations.push_back(1.);
      std::vector<int> orientations;
      orientations.push_back(1);
      orientations.push_back(-1);
      orientations.push_back(1);
      orientations.push_back(-1);
      orientations.push_back(1);

      BRep brep(locations, orientations, BBox{{{0.0}}, {{1.1}}}, 1.0);

      // File I/O.
      std::cout << "1-D b-rep with 5 faces:\n";
      brep.displayInformation(std::cout);
      brep.display(std::cout);

      assert(brep.getSimplicesSize() == locations.size());
    }
  }
#endif

  return 0;
}
