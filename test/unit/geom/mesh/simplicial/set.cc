// -*- C++ -*-

//
// Test the set operations for SimpMeshRed.
//

#include "stlib/geom/mesh/simplicial/set.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

// Distance function for a sphere.
class Sphere :
  public std::unary_function<const std::array<double, 3>&, double>
{
private:
  // Types.
  typedef std::unary_function<const std::array<double, 3>&, double> Base;

  // Data.
  std::array<double, 3> _center;
  double _radius;

public:
  // Types.
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  // Constructor.
  Sphere(const std::array<double, 3> center = {{0., 0., 0.}},
         const double radius = 1.0) :
    _center(center),
    _radius(radius) {}

  // Functor.
  result_type
  operator()(argument_type x) const
  {
    static std::array<double, 3> y;
    y = x;
    y -= _center;
    return stlib::ext::magnitude(y) - _radius;
  }
};

int
main()
{
  using namespace geom;

  // 3-D space, 3-simplex.
  typedef SimpMeshRed<3> SM;

  typedef SM::Node Node;
  typedef SM::NodeIterator NI;
  typedef SM::NodeConstIterator NCI;

  typedef SM::CellIterator CI;
  typedef SM::CellConstIterator CCI;

  //
  // Data for an octahedron
  //
  const std::array<double, 3> vertices[] = {{{0, 0, 0}},
    {{1, 0, 0}},
    {{ -1, 0, 0}},
    {{0, 1, 0}},
    {{0, -1, 0}},
    {{0, 0, 1}},
    {{0, 0, -1}}
  };
  const std::size_t numVertices = sizeof(vertices) /
                                  sizeof(std::array<double, 3>);
  const std::array<std::size_t, 4> simplices[] = {{{0, 1, 3, 5}},
    {{0, 3, 2, 5}},
    {{0, 2, 4, 5}},
    {{0, 4, 1, 5}},
    {{0, 3, 1, 6}},
    {{0, 2, 3, 6}},
    {{0, 4, 2, 6}},
    {{0, 1, 4, 6}}
  };
  const std::size_t numSimplices = sizeof(simplices) /
                                   sizeof(std::array<std::size_t, 4>);

  // Build from an indexed simplex set.
  SM x;
  x.build(vertices, vertices + numVertices,
          simplices, simplices + numSimplices);

  //-------------------------------------------------------------------------
  // Vertices outside an object.
  //-------------------------------------------------------------------------
  {
    std::vector<NI> s;
    // Get vertex iterators through the non-const member function.
    geom::determineNodesOutside(x, Sphere(), std::back_inserter(s));
    // None of the vertices are outside the unit sphere.
    assert(s.size() == 0);

    geom::determineNodesOutside(x, Sphere(std::array<double, 3>{{0., 0., 0.}},
                                          0.5),
                                std::back_inserter(s));
    assert(s.size() == 6);

    s.clear();
    geom::determineNodesOutside(x, Sphere(std::array<double, 3>{{3., 0., 0.}},
                                          1),
                                std::back_inserter(s));
    assert(s.size() == 7);
  }
  {
    std::vector<NCI> s;
    // Get vertex const iterators through the non-const member function.
    geom::determineNodesOutside(x, Sphere(), std::back_inserter(s));
    // None of the vertices are outside the unit sphere.
    assert(s.size() == 0);
  }
  {
    std::vector<NCI> s;
    const SM& y = x;
    // Get vertex const iterators through the const member function.
    geom::determineNodesOutside(y, Sphere(), std::back_inserter(s));
    // None of the vertices are outside the unit sphere.
    assert(s.size() == 0);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<NI> s;
    const SM& y = x;
    // Get vertex iterators through the const member function.
    geom::determineNodesOutside(y, Sphere(), std::back_inserter(s));
    // None of the vertices are outside the unit sphere.
    assert(s.size() == 0);
#endif
  }



  //-------------------------------------------------------------------------
  // Cells outside an object.
  //-------------------------------------------------------------------------
  {
    std::vector<CI> s;
    // Get cell iterators through the non-const member function.
    geom::determineCellsOutside(x, Sphere(), std::back_inserter(s));
    // None of the cells are outside the unit sphere.
    assert(s.size() == 0);

    geom::determineCellsOutside(x, Sphere(std::array<double, 3>{{0., 0., 0.}},
                                          0.1),
                                std::back_inserter(s));
    assert(s.size() == 8);

    s.clear();
    geom::determineCellsOutside(x, Sphere(std::array<double, 3>{{3., 0., 0.}},
                                          1),
                                std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    std::vector<CCI> s;
    // Get cell const iterators through the non-const member function.
    geom::determineCellsOutside(x, Sphere(), std::back_inserter(s));
    // None of the cells are outside the unit sphere.
    assert(s.size() == 0);
  }
  {
    std::vector<CCI> s;
    const SM& y = x;
    // Get cell const iterators through the const member function.
    geom::determineCellsOutside(y, Sphere(), std::back_inserter(s));
    // None of the cells are outside the unit sphere.
    assert(s.size() == 0);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<CI> s;
    const SM& y = x;
    // Get cell iterators through the const member function.
    geom::determineCellsOutside(y, Sphere(), std::back_inserter(s));
    // None of the cells are outside the unit sphere.
    assert(s.size() == 0);
#endif
  }




  //-------------------------------------------------------------------------
  // Interior vertices.
  //-------------------------------------------------------------------------
  {
    std::vector<NI> s;
    // Get vertex iterators through the non-const member function.
    geom::determineInteriorNodes(x, std::back_inserter(s));
    assert(s.size() == 1);
  }
  {
    std::vector<NCI> s;
    // Get vertex const iterators through the non-const member function.
    geom::determineInteriorNodes(x, std::back_inserter(s));
    assert(s.size() == 1);
  }
  {
    std::vector<NCI> s;
    const SM& y = x;
    // Get vertex const iterators through the const member function.
    geom::determineInteriorNodes(y, std::back_inserter(s));
    assert(s.size() == 1);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<NI> s;
    const SM& y = x;
    // Get vertex iterators through the const member function.
    geom::determineInteriorNodes(y, std::back_inserter(s));
    assert(s.size() == 1);
#endif
  }




  //-------------------------------------------------------------------------
  // Boundary vertices.
  //-------------------------------------------------------------------------
  {
    std::vector<Node*> s;
    // Get vertex iterators through the non-const member function.
    geom::determineBoundaryNodes(x, std::back_inserter(s));
    assert(s.size() == 6);
  }
  {
    std::vector<const Node*> s;
    // Get vertex const iterators through the non-const member function.
    geom::determineBoundaryNodes(x, std::back_inserter(s));
    assert(s.size() == 6);
  }
  {
    std::vector<const Node*> s;
    const SM& y = x;
    // Get vertex const iterators through the const member function.
    geom::determineBoundaryNodes(y, std::back_inserter(s));
    assert(s.size() == 6);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<Node*> s;
    const SM& y = x;
    // Get vertex iterators through the const member function.
    geom::determineBoundaryNodes(y, std::back_inserter(s));
    assert(s.size() == 6);
#endif
  }



  //-------------------------------------------------------------------------
  // Cells with minimum adjacencies.
  //-------------------------------------------------------------------------
  {
    std::vector<CI> s;
    // Get cell iterators through the non-const member function.
    geom::determineCellsWithRequiredAdjacencies(x, 0, std::back_inserter(s));
    assert(s.size() == 8);

    s.clear();
    geom::determineCellsWithRequiredAdjacencies(x, 1, std::back_inserter(s));
    assert(s.size() == 8);

    s.clear();
    geom::determineCellsWithRequiredAdjacencies(x, 2, std::back_inserter(s));
    assert(s.size() == 8);

    s.clear();
    geom::determineCellsWithRequiredAdjacencies(x, 3, std::back_inserter(s));
    assert(s.size() == 8);

    s.clear();
    geom::determineCellsWithRequiredAdjacencies(x, 4, std::back_inserter(s));
    assert(s.size() == 0);
  }
  {
    std::vector<CCI> s;
    // Get cell const iterators through the non-const member function.
    geom::determineCellsWithRequiredAdjacencies(x, 3, std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    std::vector<CCI> s;
    const SM& y = x;
    // Get cell const iterators through the const member function.
    geom::determineCellsWithRequiredAdjacencies(y, 3, std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<CI> s;
    const SM& y = x;
    // Get cell iterators through the const member function.
    geom::determineCellsWithRequiredAdjacencies(y, 3, std::back_inserter(s));
    assert(s.size() == 8);
#endif
  }




  //-------------------------------------------------------------------------
  // Cells with low adjacencies.
  //-------------------------------------------------------------------------
  {
    std::vector<CI> s;
    // Get cell iterators through the non-const member function.
    geom::determineCellsWithLowAdjacencies(x, 1, std::back_inserter(s));
    assert(s.size() == 0);

    s.clear();
    geom::determineCellsWithLowAdjacencies(x, 2, std::back_inserter(s));
    assert(s.size() == 0);

    s.clear();
    geom::determineCellsWithLowAdjacencies(x, 3, std::back_inserter(s));
    assert(s.size() == 0);

    s.clear();
    geom::determineCellsWithLowAdjacencies(x, 4, std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    std::vector<CCI> s;
    // Get cell const iterators through the non-const member function.
    geom::determineCellsWithLowAdjacencies(x, 4, std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    std::vector<CCI> s;
    const SM& y = x;
    // Get cell const iterators through the const member function.
    geom::determineCellsWithLowAdjacencies(y, 4, std::back_inserter(s));
    assert(s.size() == 8);
  }
  {
    // This is not allowed.  One cannot assign an iterator to a const
    // iterator.
#if 0
    std::vector<CI> s;
    const SM& y = x;
    // Get cell iterators through the const member function.
    geom::determineCellsWithLowAdjacencies(y, 4, std::back_inserter(s));
    assert(s.size() == 8);
#endif
  }

  //-------------------------------------------------------------------------
  // Neighbors.
  //-------------------------------------------------------------------------
  {
    SM::NodePointerSet neighbors;
    determineNeighbors(x, &*x.getNodesBeginning(), &neighbors);
    assert(neighbors.size() == 6);
    determineNeighbors(x, &*x.getNodesBeginning(), 0, &neighbors);
    assert(neighbors.size() == 1);
    determineNeighbors(x, &*x.getNodesBeginning(), 1, &neighbors);
    assert(neighbors.size() == 7);
  }


  return 0;
}
