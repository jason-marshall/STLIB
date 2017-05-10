// -*- C++ -*-

#include "stlib/cpt/Face.h"

#include <iostream>
#include <cmath>

using namespace stlib;
using namespace cpt;

int
main()
{
  const double eps = 10 * std::numeric_limits<double>::epsilon();

  {
    //
    // 3-D
    //

    typedef Face<3> Face;
    typedef Face::Number Number;
    typedef Face::Point Point;
    // CONTINUE
    //typedef Face::Grid Grid;
    //typedef Grid::Index Index;


    {
      // Default constructor
      Face face;
      std::cout << "Face() = " << '\n' << face << '\n';
    }
    {
      // Regular constructor
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      assert(face.isValid());
      std::cout << "Face() = " << '\n' << face << '\n';
    }
    {
      // Equality
      Face f1(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face f2(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      assert(f1 == f2);
    }
    {
      // Inequality
      Face f1(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face f2(Point{{.1, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face f3(Point{{0, 0, 0}}, Point{{1, .1, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face f4(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, .1}}, Point{{0, 0, 1}}, 1234);
      Face f5(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, -1}}, 1234);
      Face f6(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1235);
      assert(f1 != f2);
      assert(f1 != f3);
      assert(f1 != f4);
      assert(f1 != f5);
      assert(f1 != f6);
    }
    {
      // Copy constructor
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face face2(face);
      assert(face2 == face);
    }
    {
      // Assignment operator.
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face face2 = face;
      assert(face2 == face);
    }
    {
      // make
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      Face face2;
      face2.make(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 1234);
      assert(face2 == face);
    }
    {
      // distance
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);

      assert(std::abs(face.computeDistance(Point{{0, 0, 0}}) - 0) < eps);
      assert(std::abs(face.computeDistance(Point{{1, 0, 1}}) - 1) < eps);
      assert(std::abs(face.computeDistance(Point{{0, 1, -1}}) + 1) < eps);
      assert(std::abs(face.computeDistance(Point{{10, 10, 1}}) - 1) < eps);

      assert(std::abs(face.computeDistanceUnsigned(Point{{0, 0, 0}}) - 0) < eps);
      assert(std::abs(face.computeDistanceUnsigned(Point{{1, 0, 1}}) - 1) < eps);
      assert(std::abs(face.computeDistanceUnsigned(Point{{0, 1, -1}}) - 1) < eps);
      assert(std::abs(face.computeDistanceUnsigned(Point{{10, 10, 1}}) - 1) < eps);
    }
    {
      // closest point
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      Point cp;

      assert(std::abs(face.computeClosestPoint(Point{{0, 0, 0}}, &cp) - 0) < eps);
      assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

      assert(std::abs(face.computeClosestPoint(Point{{1, 0, 1}}, &cp) - 1) < eps);
      assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);

      assert(std::abs(face.computeClosestPoint(Point{{0, 1, -1}}, &cp) + 1) < eps);
      assert(geom::computeDistance(cp, Point{{0, 1, 0}}) < eps);

      assert(std::abs(face.computeClosestPoint(Point{{10, 10, 1}}, &cp) - 1) < eps);
      assert(geom::computeDistance(cp, Point{{10, 10, 0}}) < eps);


      assert(std::abs(face.computeClosestPointUnsigned(Point{{0, 0, 0}}, &cp) - 0)
             < eps);
      assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

      assert(std::abs(face.computeClosestPointUnsigned(Point{{1, 0, 1}}, &cp) - 1)
             < eps);
      assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);

      assert(std::abs(face.computeClosestPointUnsigned(Point{{0, 1, -1}}, &cp) - 1)
             < eps);
      assert(geom::computeDistance(cp, Point{{0, 1, 0}}) < eps);

      assert(std::abs(face.computeClosestPointUnsigned(Point{{10, 10, 1}}, &cp) - 1)
             < eps);
      assert(geom::computeDistance(cp, Point{{10, 10, 0}}) < eps);
    }
    {
      // gradient
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      Point grad;

      assert(std::abs(face.computeGradient(Point{{0, 0, 0}}, &grad) - 0) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeGradient(Point{{1, 0, 1}}, &grad) - 1) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeGradient(Point{{0, 1, -1}}, &grad) + 1) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeGradient(Point{{10, 10, 1}}, &grad) - 1) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);


      assert(std::abs(face.computeGradientUnsigned(Point{{0, 0, 0}}, &grad) - 0)
             < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeGradientUnsigned(Point{{1, 0, 1}}, &grad) - 1)
             < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeGradientUnsigned(Point{{0, 1, -1}}, &grad) - 1)
             < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, -1}}) < eps);

      assert(std::abs(face.computeGradientUnsigned(Point{{10, 10, 1}}, &grad) - 1)
             < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);
    }
    {
      // closest point and gradient
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      Point cp, grad;

      assert(std::abs(face.computeClosestPointAndGradient(Point{{0, 0, 0}}, &cp, &grad)
                      - 0) < eps);
      assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradient(Point{{1, 0, 1}}, &cp, &grad)
                      - 1) < eps);
      assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradient(Point{{0, 1, -1}}, &cp, &grad)
                      + 1) < eps);
      assert(geom::computeDistance(cp, Point{{0, 1, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradient(Point{{10, 10, 1}}, &cp, &grad)
                      - 1) < eps);
      assert(geom::computeDistance(cp, Point{{10, 10, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);


      assert(std::abs(face.computeClosestPointAndGradientUnsigned
                      (Point{{0, 0, 0}}, &cp, &grad) - 0) < eps);
      assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradientUnsigned
                      (Point{{1, 0, 1}}, &cp, &grad) - 1) < eps);
      assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradientUnsigned
                      (Point{{0, 1, -1}}, &cp, &grad) - 1) < eps);
      assert(geom::computeDistance(cp, Point{{0, 1, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, -1}}) < eps);

      assert(std::abs(face.computeClosestPointAndGradientUnsigned
                      (Point{{10, 10, 1}}, &cp, &grad) - 1) < eps);
      assert(geom::computeDistance(cp, Point{{10, 10, 0}}) < eps);
      assert(geom::computeDistance(grad, Point{{0, 0, 1}}) < eps);
    }
    {
      // distance (checked)
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      Number dist;

      dist = face.computeDistanceChecked(Point{{0, 0, 0}});
      assert(std::abs(dist - 0) < eps);

      dist = face.computeDistanceChecked(Point{{1, 0, 1}});
      assert(std::abs(dist - 1) < eps);

      dist = face.computeDistanceChecked(Point{{1, 1, -1}});
      assert(std::abs(dist + 1) < eps);
    }
    {
      // closest point (checked)
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      Number dist;
      Point cp;

      dist = face.computeClosestPointChecked(Point{{0, 0, 0}}, &cp);
      assert(std::abs(dist - 0) < eps);
      assert(geom::computeDistance(cp, Point{{0, 0, 0}}) < eps);

      dist = face.computeClosestPointChecked(Point{{1, 0, 1}}, &cp);
      assert(std::abs(dist - 1) < eps);
      assert(geom::computeDistance(cp, Point{{1, 0, 0}}) < eps);

      dist = face.computeClosestPointChecked(Point{{1, 1, -1}}, &cp);
      assert(std::abs(dist + 1) < eps);
      assert(geom::computeDistance(cp, Point{{1, 1, 0}}) < eps);
    }
    // CONTINUE
#if 0
    {
      // scan convert
      Face face(Point{{0, 0, 0}}, Point{{1, 0, 0}}, Point{{1, 1, 0}}, Point{{0, 0, 1}}, 0);
      std::vector<Index> indexSet;
      Grid grid(Index(10, 10, 10),
                BBox(Point{{0, 0, 0}}, Point{{1, 1, 1}}));

      assert(face.scanConvert(indexSet, grid, 1) == 550);
      assert(face.scanConvert(indexSet, grid, 0.5) == 275);
      assert(face.scanConvert(indexSet, grid, 0) == 55);
    }
#endif
  }






  {
    //
    // 2-D
    //

    typedef Face<2> Face;
    typedef Face::Point Point;
    {
      // Default constructor
      Face f;
    }
    {
      Point source = {{1, 0}};
      Point target = {{0, 0}};
      Point tangent = {{ -1, 0}};
      Point normal = {{0, 1}};
      std::size_t faceIndex = 19;

      Face a(source, target, normal, faceIndex);
      assert(a.getSource() == source);
      assert(a.getTarget() == target);
      assert(a.getTangent() == tangent);
      assert(a.getNormal() == normal);
      assert(a.getFaceIndex() == faceIndex);

      //
      // Distance
      //

      assert(std::abs(a.computeDistance(Point{{0, 1}}) - 1) < eps);
      assert(std::abs(a.computeDistance(Point{{1, 1}}) - 1) < eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{0, 1}}) - 1) < eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{1, 1}}) - 1) < eps);

      assert(std::abs(a.computeDistance(Point{{0, -1}}) + 1) < eps);
      assert(std::abs(a.computeDistance(Point{{1, -1}}) + 1) < eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{0, -1}}) - 1) < eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{1, -1}}) - 1) < eps);

      assert(std::abs(a.computeDistance(Point{{-1, 1}}) - std::sqrt(2.)) <
             eps);
      assert(std::abs(a.computeDistance(Point{{2, 1}}) - std::sqrt(2.)) <
             eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{-1, 1}}) - std::sqrt(2.)) <
             eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{2, 1}}) - std::sqrt(2.)) <
             eps);

      assert(std::abs(a.computeDistance(Point{{-1, -1}}) + std::sqrt(2.)) <
             eps);
      assert(std::abs(a.computeDistance(Point{{2, -1}}) + std::sqrt(2.)) <
             eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{-1, -1}}) - std::sqrt(2.))
             < eps);
      assert(std::abs(a.computeDistanceUnsigned(Point{{2, -1}}) - std::sqrt(2.)) <
             eps);

      //
      // Closest point
      //

      Point cp;
      assert(std::abs(a.computeClosestPoint(Point{{0, 1}}, &cp) - 1) < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPoint(Point{{1, 1}}, &cp) - 1) < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);

      assert(std::abs(a.computeClosestPointUnsigned(Point{{0, 1}}, &cp) - 1)
             < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPointUnsigned(Point{{1, 1}}, &cp) - 1)
             < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);


      assert(std::abs(a.computeClosestPoint(Point{{0, -1}}, &cp) + 1) <
             eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPoint(Point{{1, -1}}, &cp) + 1) <
             eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);

      assert(std::abs(a.computeClosestPointUnsigned(Point{{0, -1}}, &cp) - 1) <
             eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPointUnsigned(Point{{1, -1}}, &cp) - 1) <
             eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);


      assert(std::abs(a.computeClosestPoint(Point{{-1, 1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPoint(Point{{2, 1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);

      assert(std::abs(a.computeClosestPointUnsigned(Point{{-1, 1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPointUnsigned(Point{{2, 1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);


      assert(std::abs(a.computeClosestPoint(Point{{-1, -1}}, &cp) +
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPoint(Point{{2, -1}}, &cp) +
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);

      assert(std::abs(a.computeClosestPointUnsigned(Point{{-1, -1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(target, cp)) < eps);
      assert(std::abs(a.computeClosestPointUnsigned(Point{{2, -1}}, &cp) -
                      std::sqrt(2.)) < eps);
      assert(std::abs(geom::computeDistance(source, cp)) < eps);


      Face b;
      b.make(source, target, normal, faceIndex);
      assert(a == b);

      Face::Polygon polygon;
      a.buildCharacteristicPolygon(&polygon, 1);
      std::cout << "Characteristic polygon for the edge:\n"
                << polygon;
    }
  }


  {
    //
    // 1-D
    //

    typedef Face<1> Face;
    typedef Face::Number Number;
    typedef Face::Point Point;
    {
      // Default constructor
      Face f;
    }
    {
      // Positive orientation.
      Point location = 0.0;
      int orientation = 1;
      std::size_t index = 19;
      Point left = -1.0;
      Point right = 1.0;
      Number maximumDistance = 1.0;
      geom::BBox<double, 1> domain = {{{ -0.5}}, {{0.5}}};

      Face a(location, orientation, index, left, right, maximumDistance);
      std::cout << "\nFace<1> = \n" << a << '\n';

      assert(a.getLocation() == location);
      assert(a.getOrientation() == orientation);
      assert(a.getFaceIndex() == index);
      assert(a.getDomain() == domain);
      assert(a.isValid());

      //
      // Distance
      //

      assert(std::abs(a.computeDistance(0.0) - 0.0) < eps);
      assert(std::abs(a.computeDistance(1.0) - 1.0) < eps);
      assert(std::abs(a.computeDistance(-1.0) + 1.0) < eps);

      assert(std::abs(a.computeDistanceUnsigned(0.0) - 0.0) < eps);
      assert(std::abs(a.computeDistanceUnsigned(1.0) - 1.0) < eps);
      assert(std::abs(a.computeDistanceUnsigned(-1.0) - 1.0) < eps);
    }
    {
      // Negative orientation.
      Point location = 1.0;
      int orientation = -1;
      std::size_t index = 19;
      Point left = -3.0;
      Point right = 5.0;
      Number maximumDistance = 1.0;
      geom::BBox<double, 1> domain = {{{0.0}}, {{2.0}}};

      Face a(location, orientation, index, left, right, maximumDistance);

      assert(a.getLocation() == location);
      assert(a.getOrientation() == orientation);
      assert(a.getFaceIndex() == index);
      assert(a.getDomain() == domain);
      assert(a.isValid());

      //
      // Distance
      //

      assert(std::abs(a.computeDistance(1.0) - 0.0) < eps);
      assert(std::abs(a.computeDistance(2.0) + 1.0) < eps);
      assert(std::abs(a.computeDistance(0.0) - 1.0) < eps);

      assert(std::abs(a.computeDistanceUnsigned(1.0) - 0.0) < eps);
      assert(std::abs(a.computeDistanceUnsigned(2.0) - 1.0) < eps);
      assert(std::abs(a.computeDistanceUnsigned(0.0) - 1.0) < eps);
    }
  }

  return 0;
}
