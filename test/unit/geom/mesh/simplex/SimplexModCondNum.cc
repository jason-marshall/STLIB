// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{

  //
  // 2-D
  //
  {
    typedef geom::SimplexModCondNum<2> SCN;
    typedef SCN::Vertex Vertex;
    typedef SCN::Simplex Simplex;

    {
      // Default constructor.
      SCN triangle;
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Identity triangle:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                      {{0., 1.}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Reference triangle:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{10., 0.}},
                     {{0., 10.}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Scaled reference triangle:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1., 1e-8}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Almost flat triangle:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}},
                     {{1., 0.}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Inverted identity triangle:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{1., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SCN triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Flat triangle, two vertices coincide:\n"
                << "condition number = " << triangle() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
  }


  //
  // 3-D
  //
  {
    typedef geom::SimplexModCondNum<3> SCN;
    typedef SCN::Vertex Vertex;
    typedef SCN::Simplex Simplex;

    {
      // Default constructor.
      SCN tet;
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Identity tetrahedron:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{0., 0., 1.}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Reference tetrahedron:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{10., 0., 0.}},
                   {{0., 10., 0.}},
                   {{0., 0., 10.}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Scaled reference tetrahedron:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{1., 1., 1e-8}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Almost flat tetrahedron:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Inverted identity tetrahedron:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{1., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SCN tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Flat tetrahedron, two vertices coincide:\n"
                << "condition number = " << tet() << '\n'
                << "gradient condition number = " << gradient
                << '\n' << '\n';
    }
  }
  return 0;
}
