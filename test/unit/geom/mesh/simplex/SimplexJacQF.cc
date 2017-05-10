// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexJacQF.h"

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
    typedef geom::SimplexJacQF<2> SJQF;
    typedef SJQF::Vertex Vertex;
    typedef SJQF::Simplex Simplex;

    {
      // Dimension.
      assert(SJQF::getDimension() == 2);
    }
    {
      // Default constructor.
      SJQF triangle;
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SJQF triangle(tri);
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
      SJQF triangle(tri);
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
      SJQF triangle(tri);
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
      SJQF triangle(tri);
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
      SJQF triangle(tri);
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
      SJQF triangle(tri);
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
    typedef geom::SimplexJacQF<3> SJQF;
    typedef SJQF::Vertex Vertex;
    typedef SJQF::Simplex Simplex;

    {
      // Dimension.
      assert(SJQF::getDimension() == 3);
    }
    {
      // Default constructor.
      SJQF tet;
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SJQF tet(t);
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
      SJQF tet(t);
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
      SJQF tet(t);
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
      SJQF tet(t);
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
      SJQF tet(t);
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
      SJQF tet(t);
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
