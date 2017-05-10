// -*- C++ -*-

#include "stlib/cpt/Edge.h"

#include <iostream>
#include <cmath>

using namespace stlib;
using namespace cpt;

int
main()
{
  typedef Edge<3> Edge;
  typedef Edge::Number Number;
  typedef Edge::Point Point;

  const double eps = 10 * std::numeric_limits<double>::epsilon();

  {
    // Default constructor
    Edge edge;
  }
  {
    // Regular constructor
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    assert(edge.isValid());
    std::cout  << "Edge(Point{{0,0,0), Point{{1,0,0), Point{{0,1,0), Point{{0,0,1), 1234) = "
               << '\n' << edge << '\n';
  }
  {
    // Equality
    Edge e1(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge e2(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    assert(e1 == e2);
  }
  {
    // Inequality
    Edge e1(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge e2(Point{{0.1, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge e3(Point{{0, 0, 0}}, Point{{1.1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge e4(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, -1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge e5(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, -1}}, 1234);
    Edge e6(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1235);
    assert(e1 != e2);
    assert(e1 != e3);
    assert(e1 != e4);
    assert(e1 != e5);
    assert(e1 != e6);
  }
  {
    // Copy constructor
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge edge2(edge);
    assert(edge2 == edge);
  }
  {
    // Assignment operator.
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Edge edge2 = edge;
    assert(edge2 == edge);
  }
  {
    // distance
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);

    assert(std::abs(edge.computeDistance(Point{{0, 0, 0}}) - 0) < eps);
    assert(std::abs(edge.computeDistance(Point{{1, 0, 1}}) - 1) < eps);
    assert(std::abs(edge.computeDistance(Point{{2, 0, 1}}) - 1) < eps);
    assert(std::abs(edge.computeDistance(Point{{0, 1, -1}}) - std::sqrt(2.0)) < eps);

    assert(std::abs(edge.computeDistanceUnsigned(Point{{0, 0, 0}}) - 0) < eps);
    assert(std::abs(edge.computeDistanceUnsigned(Point{{1, 0, 1}}) - 1) < eps);
    assert(std::abs(edge.computeDistanceUnsigned(Point{{2, 0, 1}}) - 1) < eps);
    assert(std::abs(edge.computeDistanceUnsigned(Point{{0, 1, -1}}) - std::sqrt(2.0))
           < eps);
  }
  {
    // closest point
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Point cp;

    assert(std::abs(edge.computeClosestPoint(Point{{0, 0, 0}}, &cp) - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPoint(Point{{1, 0, 1}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPoint(Point{{2, 0, 1}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, Point{{2, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPoint(Point{{0, 1, -1}}, &cp) -
                    std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);


    assert(std::abs(edge.computeClosestPointUnsigned(Point{{0, 0, 0}}, &cp) - 0)
           < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPointUnsigned(Point{{1, 0, 1}}, &cp) - 1)
           < eps);
    assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPointUnsigned(Point{{2, 0, 1}}, &cp) - 1)
           < eps);
    assert(geom::computeDistance(cp, Point{{2, 0, 0}}) < eps);

    assert(std::abs(edge.computeClosestPointUnsigned(Point{{0, 1, -1}}, &cp)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
  }
  {
    // gradient
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Point grad;

    assert(std::abs(edge.computeGradient(Point{{0, 0, 0}}, &grad) - 0) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1, 0}}) < eps);

    assert(std::abs(edge.computeGradient(Point{{1, 0, 1}}, &grad) - 1) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeGradient(Point{{2, 0, 1}}, &grad) - 1) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeGradient(Point{{0, 1, -1}}, &grad) - std::sqrt(2.0))
           < eps);
    assert(geom::computeDistance(grad,
                                 Point{{0, 1 / std::sqrt(2.0), -1 / std::sqrt(2.0)}})
           < eps);


    assert(std::abs(edge.computeGradientUnsigned(Point{{0, 0, 0}}, &grad) - 0) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1, 0}}) < eps);

    assert(std::abs(edge.computeGradientUnsigned(Point{{1, 0, 1}}, &grad) - 1) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeGradientUnsigned(Point{{2, 0, 1}}, &grad) - 1) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeGradientUnsigned(Point{{0, 1, -1}}, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(grad,
                                 Point{{0, 1 / std::sqrt(2.0), -1 / std::sqrt(2.0)}})
           < eps);
  }
  {
    // closest point and gradient
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Point cp, grad;

    assert(std::abs(edge.computeClosestPointAndGradient(Point{{0, 0, 0}}, &cp,
                    &grad) - 0)
           < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1, 0}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradient(Point{{1, 0, 1}}, &cp,
                    &grad) - 1)
           < eps);
    assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradient(Point{{2, 0, 1}}, &cp,
                    &grad) - 1)
           < eps);
    assert(geom::computeDistance(cp, Point{{2, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradient(Point{{0, 1, -1}}, &cp, &grad)
                    - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1 / std::sqrt(2.0),
                                          -1 / std::sqrt(2.0)}})
           < eps);


    assert(std::abs(edge.computeClosestPointAndGradientUnsigned
                    (Point{{0, 0, 0}}, &cp, &grad) - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1, 0}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradientUnsigned
                    (Point{{1, 0, 1}}, &cp, &grad) - 1) < eps);
    assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradientUnsigned
                    (Point{{2, 0, 1}}, &cp, &grad) - 1) < eps);
    assert(geom::computeDistance(cp, Point{{2, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

    assert(std::abs(edge.computeClosestPointAndGradientUnsigned
                    (Point{{0, 1, -1}}, &cp, &grad) - std::sqrt(2.0)) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
    assert(geom::computeDistance(grad, Point{{0, 1 / std::sqrt(2.0),
                                          -1 / std::sqrt(2.0)}})
           < eps);
  }
  {
    // distance (checked)
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Number dist;

    dist = edge.computeDistanceChecked(Point{{0, 0, 0}});
    assert(std::abs(dist - 0) < eps);

    dist = edge.computeDistanceChecked(Point{{1, 0, 1}});
    assert(std::abs(dist - 1) < eps);
  }
  {
    // closest point (checked)
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 1234);
    Number dist;
    Point cp = {{0, 0, 0}};

    dist = edge.computeClosestPointChecked(Point{{0, 0, 0}}, &cp);
    assert(std::abs(dist - 0) < eps);
    assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

    dist = edge.computeClosestPointChecked(Point{{1, 0, 1}}, &cp);
    assert(std::abs(dist - 1) < eps);
    assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);
  }
  {
#if 0
    // scan convert
    Edge edge(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{0, 1, 0}}, Point{{0, 0, 1}}, 0);
    std::vector<IndexList> indexSet;
    Grid grid(Grid::IndexList(10, 10, 10),
              BBox(Point{{0, 0, 0}}, Point{{1, 1, 1}}));
    Polyhedron poly;

    // CONTINUE
    /*
    std::cout << poly << '\n';

    edge.buildCharacteristicPolyhedron(&poly, std::sqrt(2.0), grid);
    assert(poly.scanConvert(indexSet, grid) == 1000);

    edge.buildCharacteristicPolyhedron(poly, eps, grid);
    assert(poly.scanConvert(indexSet, grid) == 10);
    */
#endif
  }

  return 0;
}
