// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexAdjJac.h"

#include "stlib/geom/mesh/simplex/SimplexJac.h"

#include <iostream>
#include <limits>

#include <cassert>

using namespace stlib;

int
main()
{

  //
  // 2-D
  //
  {
    typedef geom::SimplexJac<2> TriJac;
    typedef geom::SimplexAdjJac<2> TriAdjJac;
    typedef TriJac::Simplex Triangle;

    {
      // Default constructor.
      TriAdjJac triangle;
    }
    {
      Triangle tri = {{{{0., 0.}},
                      {{1., 0.}},
                      {{1. / 2, std::sqrt(3.) / 2}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Identity jacobian:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Triangle tri = {{{{0., 0.}},
                      {{1., 0.}},
                      {{0., 1.}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Reference jacobian:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Triangle tri = {{{{0., 0.}},
                      {{10., 0.}},
                      {{0., 10.}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Scaled reference jacobian:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Triangle tri = {{{{0., 0.}},
                      {{1., 0.}},
                      {{1., 1e-8}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Almost flat jacobian:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Triangle tri = {{{{0., 0.}},
                      {{1. / 2, std::sqrt(3.) / 2}},
                      {{1., 0.}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Inverted identity jacobian:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Triangle tri = {{{{1., 0.}},
                      {{1., 0.}},
                      {{1. / 2, std::sqrt(3.) / 2}}}};
      TriJac jacobian(tri);
      TriAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Flat jacobian, two vertices coincide:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
  }


  //
  // 3-D
  //
  {
    typedef geom::SimplexJac<3> TetJac;
    typedef geom::SimplexAdjJac<3> TetAdjJac;
    typedef TetJac::Matrix Matrix;
    typedef TetJac::Simplex Simplex;

    {
      // Default constructor.
      TetAdjJac adjoint;
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                    {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Identity tetrahedron:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{0., 0., 1.}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Reference tetrahedron:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{10., 0., 0.}},
                   {{0., 10., 0.}},
                   {{0., 0., 10.}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Scaled reference tetrahedron:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{0., 1., 0.}},
                   {{1., 1., 1e-8}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Almost flat tetrahedron:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Inverted identity tetrahedron:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }
    {
      Simplex t = {{{{1., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Flat tetrahedron, two vertices coincide:\n"
                << "det = " << jacobian.getDeterminant() << '\n'
                << "J * AJ = \n" << jacobian.getMatrix() * adjoint.getMatrix()
                << '\n';
    }

    {
      Simplex t = {{{{0., 0., 0.}},
                   {{1., 0., 0.}},
                   {{1. / 2, std::sqrt(3.) / 2, 0.}},
                   {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
      TetJac jacobian(t);
      TetAdjJac adjoint(jacobian.getMatrix());
      std::cout << "Identity tetrahedron:\n"
                << "Difference between analytical and numerical gradient "
                << "of the adjoint.\n";

      Matrix mp, mm;
      const double eps = std::pow(std::numeric_limits<double>::epsilon(),
                                  1.0 / 3.0);
      for (int n = 0; n != 3; ++n) {
        t[0][n] = eps;
        jacobian.set(t);
        adjoint.set(jacobian.getMatrix());
        mp = adjoint.getMatrix();
        t[0][n] = -eps;
        jacobian.set(t);
        adjoint.set(jacobian.getMatrix());
        mm = adjoint.getMatrix();
        t[0][n] = 0.0;
        jacobian.set(t);
        adjoint.set(jacobian.getMatrix());
        std::cout << adjoint.getGradientMatrix()[n] - (mp - mm) / (2.0 * eps)
                  << '\n';
      }
    }
  }
  return 0;
}
