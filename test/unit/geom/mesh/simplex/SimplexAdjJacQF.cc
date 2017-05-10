// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexAdjJacQF.h"

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
    typedef geom::SimplexAdjJacQF<2> SAJQF;
    typedef SAJQF::Vertex Vertex;
    typedef SAJQF::Simplex Simplex;

    {
      // Dimension.
      assert(SAJQF::getDimension() == 2);
    }
    {
      // Default constructor.
      SAJQF triangle;
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Identity triangle:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{0., 1.}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Reference triangle:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{10., 0.}},
                     {{0., 10.}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Scaled reference triangle:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1., 1e-8}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Almost flat triangle:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}},
                     {{1., 0.}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Inverted identity triangle:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{1., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SAJQF triangle(tri);
      Vertex gradient;
      triangle.computeGradientContent(&gradient);
      std::cout << "Flat triangle, two vertices coincide:\n"
                << "determinant = " << triangle.getDeterminant() << '\n'
                << "content = " << triangle.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
  }


  //
  // 3-D
  //
  {
    typedef geom::SimplexAdjJacQF<3> SAJQF;
    typedef SAJQF::Vertex Vertex;
    typedef SAJQF::Simplex Simplex;

    {
      // Dimension.
      assert(SAJQF::getDimension() == 3);
    }
    {
      // Default constructor.
      SAJQF tet;
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                    {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SAJQF tet(t);
      Vertex gradient;
      tet.computeGradientContent(&gradient);
      std::cout << "Identity tetrahedron:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{0., 0., 1.}}}};
      SAJQF tet(t);
      Vertex gradient;
      tet.computeGradientContent(&gradient);
      std::cout << "Reference tetrahedron:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{10., 0., 0.}},
                   {{0., 10., 0.}},
                   {{0., 0., 10.}}}};
      SAJQF tet(t);
      Vertex gradient;
      tet.computeGradientContent(&gradient);
      std::cout << "Scaled reference tetrahedron:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                    {{1., 1., 1e-8}}}};
      SAJQF tet(t);
      Vertex gradient;
      std::cout << "Almost flat tetrahedron:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}},
                    {{1. / 2, std::sqrt(3.) / 2, 0.}}}};
      SAJQF tet(t);
      Vertex gradient;
      tet.computeGradientContent(&gradient);
      std::cout << "Inverted identity tetrahedron:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{1., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                    {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SAJQF tet(t);
      Vertex gradient;
      tet.computeGradientContent(&gradient);
      std::cout << "Flat tetrahedron, two vertices coincide:\n"
                << "determinant = " << tet.getDeterminant() << '\n'
                << "content = " << tet.computeContent() << '\n'
                << "gradient content = " << gradient
                << '\n' << '\n';
    }
  }
  return 0;
}
