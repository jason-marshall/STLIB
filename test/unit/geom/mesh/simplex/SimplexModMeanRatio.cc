// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"

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
    typedef geom::SimplexModMeanRatio<2> SMR;
    typedef SMR::Vertex Vertex;
    typedef SMR::Simplex Simplex;

    {
      // Default constructor.
      SMR triangle;
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Identity triangle:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{0., 1.}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Reference triangle:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{10., 0.}},
                     {{0., 10.}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Scaled reference triangle:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1., 0.}},
                     {{1., 1e-8}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Almost flat triangle:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{0., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}},
                     {{1., 0.}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Inverted identity triangle:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex tri = {{{{1., 0.}},
                     {{1., 0.}},
                     {{1. / 2, std::sqrt(3.) / 2}}}};
      SMR triangle(tri);
      Vertex gradient;
      triangle.computeGradient(&gradient);
      std::cout << "Flat triangle, two vertices coincide:\n"
                << "mean ratio = " << triangle() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
  }


  //
  // 3-D
  //
  {
    typedef geom::SimplexModMeanRatio<3> SMR;
    typedef SMR::Vertex Vertex;
    typedef SMR::Simplex Simplex;

    {
      // Default constructor.
      SMR tet;
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Identity tetrahedron:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{0., 0., 1.}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Reference tetrahedron:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{10., 0., 0.}},
                   {{0., 10., 0.}},
                   {{0., 0., 10.}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Scaled reference tetrahedron:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{1., 1., 1e-8}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Almost flat tetrahedron:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Inverted identity tetrahedron:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
    {
      Simplex t = {{{{1., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      SMR tet(t);
      Vertex gradient;
      tet.computeGradient(&gradient);
      std::cout << "Flat tetrahedron, two vertices coincide:\n"
                << "mean ratio = " << tet() << '\n'
                << "gradient mean ratio = " << gradient
                << '\n' << '\n';
    }
  }
  return 0;
}
