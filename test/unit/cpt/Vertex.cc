// -*- C++ -*-

#include "stlib/cpt/Vertex.h"

#include <iostream>
#include <cmath>

using namespace stlib;
using namespace cpt;

int
main()
{
  const double eps = 10 * std::numeric_limits<double>::epsilon();

  //
  // 3-D
  //
  {
    typedef Vertex<3> Vertex;
    // Default constructor
    Vertex vertex;
  }
  {
    typedef Vertex<3> Vertex;
    typedef Vertex::Point Point;
    // Construct from b-rep information.
    std::vector<Point> neighbors;
    neighbors.push_back(Point{{-1, 0, 0}});
    neighbors.push_back(Point{{0, -1, 0}});
    neighbors.push_back(Point{{0, 0, -1}});
    std::vector<Point> faceNormals;
    faceNormals.push_back(Point{{0, 0, 1}});
    faceNormals.push_back(Point{{1, 0, 0}});
    faceNormals.push_back(Point{{0, 1, 0}});
    Point normal = {{
        1.0 / std::sqrt(3.0), 1.0 / std::sqrt(3.0),
        1.0 / std::sqrt(3.0)
      }
    };
    Vertex vertex(Point{{0, 0, 0}}, normal, neighbors, faceNormals, 1234);
    std::cout << "Vertex((0,0,0), neighbors, faceNormals, 1234) = "
              << '\n' << vertex << '\n';

    assert(vertex.getLocation() == (Point{{0, 0, 0}}));
    assert(vertex.isConvex());
    assert(! vertex.isConcave());

    Vertex copy = vertex;
    assert(copy == vertex);

    Vertex assign;
    assign = vertex;
    assert(assign == vertex);

    //
    // distance
    //
    assert(std::abs(vertex.computeDistance(Point{{0, 0, 0}}) - 0) < eps);
    assert(std::abs(vertex.computeDistance(Point{{1, 0, 1}}) - std::sqrt(2.0))
           < eps);
    assert(std::abs(vertex.computeDistance(Point{{2, 0, 1}}) - std::sqrt(5.0))
           < eps);

    assert(std::abs(vertex.computeDistanceUnsigned(Point{{0, 0, 0}}) - 0)
           < eps);
    assert(std::abs(vertex.computeDistanceUnsigned(Point{{1, 0, 1}})
                    - std::sqrt(2.0)) < eps);
    assert(std::abs(vertex.computeDistanceUnsigned(Point{{2, 0, 1}})
                    - std::sqrt(5.0)) < eps);

    //
    // closest point
    //
    Point cp;

    assert(std::abs(vertex.computeClosestPoint(Point{{0, 0, 0}}, &cp) - 0)
           < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(vertex.computeClosestPoint(Point{{1, 0, 1}}, &cp)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(vertex.computeClosestPoint(Point{{2, 0, 1}}, &cp)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);


    assert(std::abs(vertex.computeClosestPointUnsigned(Point{{0, 0, 0}}, &cp)
                    - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(vertex.computeClosestPointUnsigned(Point{{1, 0, 1}}, &cp)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(vertex.computeClosestPointUnsigned(Point{{2, 0, 1}}, &cp)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    //
    // Gradient
    //
    Point grad;

    // Here the gradient is the first face normal.
    assert(std::abs(vertex.computeGradient(Point{{0, 0, 0}}, &grad) - 0)
           < eps);
    assert(geom::computeDistance(grad, faceNormals[0]) < eps);

    assert(std::abs(vertex.computeGradient(Point{{1, 0, 1}}, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(grad, Point{{1, 0, 1}} / std::sqrt(2.0))
           < eps);

    assert(std::abs(vertex.computeGradient(Point{{2, 0, 1}}, &grad)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(grad, Point{{2, 0, 1}} / std::sqrt(5.0))
           < eps);


    assert(std::abs(vertex.computeGradientUnsigned(Point{{0, 0, 0}}, &grad)
                    - 0) < eps);
    assert(geom::computeDistance(grad, faceNormals[0]) < eps);

    assert(std::abs(vertex.computeGradientUnsigned(Point{{1, 0, 1}}, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(grad, Point{{1, 0, 1}} / std::sqrt(2.0))
           < eps);

    assert(std::abs(vertex.computeGradientUnsigned(Point{{2, 0, 1}}, &grad)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(grad, Point{{2, 0, 1}} / std::sqrt(5.0))
           < eps);

    //
    // Closest point and gradient.
    //

    // Here the gradient is the first face normal.
    assert(std::abs(vertex.computeClosestPointAndGradient
                    (Point{{0, 0, 0}}, &cp, &grad) - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, faceNormals[0]) < eps);

    assert(std::abs(vertex.computeClosestPointAndGradient
                    (Point{{1, 0, 1}}, &cp, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{1, 0, 1}} / std::sqrt(2.0))
           < eps);

    assert(std::abs(vertex.computeClosestPointAndGradient
                    (Point{{2, 0, 1}}, &cp, &grad)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{2, 0, 1}} / std::sqrt(5.0))
           < eps);


    assert(std::abs(vertex.computeClosestPointAndGradientUnsigned
                    (Point{{0, 0, 0}}, &cp, &grad) - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, faceNormals[0]) < eps);

    assert(std::abs(vertex.computeClosestPointAndGradientUnsigned
                    (Point{{1, 0, 1}}, &cp, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{1, 0, 1}} / std::sqrt(2.0))
           < eps);

    assert(std::abs(vertex.computeClosestPointAndGradientUnsigned
                    (Point{{2, 0, 1}}, &cp, &grad)
                    - std::sqrt(5.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{2, 0, 1}} / std::sqrt(5.0))
           < eps);

    // CONTINUE
  }


  //
  // 2-D
  //
  {
    // Default constructor
    typedef Vertex<2> Vertex;
    Vertex vertex;
  }
  {
    // A straight line.
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{2, 3}};
    Point rightNormal = {{0, 1}};
    Point leftNormal = {{0, 1}};
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);
    assert(v.getLocation() == location);
    assert(v.getRightNormal() == rightNormal);
    assert(v.getLeftNormal() == leftNormal);
    assert(v.getFaceIndex() == faceIndex);
    assert(v.getSignOfDistance() == 0);
    assert(v.isConvexOrConcave() == false);

    Vertex u;
    u.make(location, rightNormal, leftNormal, faceIndex);
    assert(u == v);
  }
  {
    // convex
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{2, 3}};
    Point rightNormal = {{1, 0}};
    Point leftNormal = {{0, 1}};
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);
    assert(v.getLocation() == location);
    assert(v.getRightNormal() == rightNormal);
    assert(v.getLeftNormal() == leftNormal);
    assert(v.getFaceIndex() == faceIndex);
    assert(v.getSignOfDistance() == 1);
    assert(v.isConvexOrConcave() == true);
    assert(std::abs(v.computeDistance(Point{{3, 3}}) - 1) < eps);
    assert(std::abs(v.computeDistanceUnsigned(Point{{3, 3}}) - 1) < eps);

    Vertex u;
    u.make(location, rightNormal, leftNormal, faceIndex);
    assert(u == v);
  }
  {
    // convex
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{0, 0}};
    Point rightNormal = {{1, 0}};
    Point leftNormal = {{1, 1}};
    stlib::ext::normalize(&leftNormal);
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);

    Vertex::Polygon polygon;
    v.buildCharacteristicPolygon(&polygon, 1);
    std::cout << "Characteristic polygon at a convex point, angle = pi/4:\n"
              << polygon;
  }
  {
    // convex
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{0, 0}};
    Point rightNormal = {{1, 0}};
    Point leftNormal = {{0, 1}};
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);

    Vertex::Polygon polygon;
    v.buildCharacteristicPolygon(&polygon, 1);
    std::cout << "Characteristic polygon at a convex point, angle = pi/2:\n"
              << polygon;
  }
  {
    // convex
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{0, 0}};
    Point rightNormal = {{1, 0}};
    Point leftNormal = {{ - 1, eps}};
    stlib::ext::normalize(&leftNormal);
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);

    Vertex::Polygon polygon;
    v.buildCharacteristicPolygon(&polygon, 1);
    std::cout << "Characteristic polygon at a convex point, "
              << "angle = almost pi:\n"
              << polygon;
  }
  {
    // concave
    typedef Vertex<2> Vertex;
    typedef Vertex::Point Point;
    Point location = {{2, 3}};
    Point rightNormal = {{ -1, 0}};
    Point leftNormal = {{0, 1}};
    std::size_t faceIndex = 47;

    Vertex v(location, rightNormal, leftNormal, faceIndex);
    assert(v.getLocation() == location);
    assert(v.getRightNormal() == rightNormal);
    assert(v.getLeftNormal() == leftNormal);
    assert(v.getFaceIndex() == faceIndex);
    assert(v.getSignOfDistance() == -1);
    assert(v.isConvexOrConcave() == true);
    assert(std::abs(v.computeDistance(Point{{3, 3}}) + 1) < eps);
    assert(std::abs(v.computeDistanceUnsigned(Point{{3, 3}}) - 1) < eps);

    Vertex u;
    u.make(location, rightNormal, leftNormal, faceIndex);
    assert(u == v);

    Vertex::Polygon polygon;
    v.buildCharacteristicPolygon(&polygon, 1);
    std::cout << "Characteristic polygon at a concave point:\n"
              << polygon;
  }

  return 0;
}
