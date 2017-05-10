// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexJac.h"

#include "stlib/geom/kernel/content.h"

#include <iostream>
#include <limits>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

template<typename _Float>
void
test2()
{
  typedef geom::SimplexJac<2, _Float> TriJac;
  typedef typename TriJac::Vertex Vertex;
  typedef typename TriJac::Simplex Simplex;

  {
    // Default constructor.
    TriJac triangle;
  }
  {
    TriJac triangle;
    Vertex v[3] = {{{0, 0}}, {{1, 0}}, {{_Float(1) / 2,
                                         std::sqrt(_Float(3)) / 2}}};
    for (std::size_t a = 0; a != 3; ++a) {
      for (std::size_t b = 0; b != 3; ++b) {
        if (b != a) {
          for (std::size_t c = 0; c != 3; ++c) {
            if (c != a && c != b) {
              triangle.set(Simplex{{v[a], v[b], v[c]}});
              std::cout << "Permutation "
                        << a << " " << b << " " << c << ": "
                        << triangle.getDeterminant() << '\n';
            }
          }
        }
      }
    }
  }
  {
    Simplex tri = {{Vertex{{0, 0}},
                    Vertex{{1, 0}},
                    Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}}}};
    TriJac triangle(tri);
    std::cout << "Identity triangle:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0, 0}},
                                          Vertex{{1, 0}},
                                          Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * c * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex tri = {{Vertex{{0, 0}},
                    Vertex{{1, 0}},
                    Vertex{{0, 1}}}};
    TriJac triangle(tri);
    std::cout << "Reference triangle:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0., 0.}},
                                          Vertex{{1., 0.}},
                                          Vertex{{0., 1.}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * c * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex tri = {{Vertex{{0, 0}},
                    Vertex{{10, 0}},
                    Vertex{{0, 10}}}};
    TriJac triangle(tri);
    std::cout << "Scaled reference triangle:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0., 0.}},
                                          Vertex{{10., 0.}},
                                          Vertex{{0., 10.}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * c * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex tri = {{Vertex{{0, 0}},
                    Vertex{{1, 0}},
                    Vertex{{1, _Float(1e-8)}}}};
    TriJac triangle(tri);
    std::cout << "Almost flat triangle:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0., 0.}},
                                          Vertex{{1., 0.}},
                                          Vertex{{1., 1e-8}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * c * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex tri = {{Vertex{{0, 0}},
                    Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}},
                    Vertex{{1, 0}}}};
    TriJac triangle(tri);
    std::cout << "Inverted identity triangle:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0, 0}},
                                          Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}},
                                          Vertex{{1, 0}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex tri = {{Vertex{{1, 0}},
                    Vertex{{1, 0}},
                    Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}}}};
    TriJac triangle(tri);
    std::cout << "Flat triangle, two vertices coincide:\n"
              << "det = " << triangle.getDeterminant()
              << "\ngrad det = " << triangle.getGradientDeterminant()
              << "\ncontent = " << triangle.computeContent()
              << "\ngrad content = " << triangle.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{1, 0}},
                                          Vertex{{1, 0}},
                                          Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2}});
    assert(std::abs(c - triangle.computeContent()) <
           10.0 * std::numeric_limits<_Float>::epsilon());
  }
}


template<typename _Float>
void
test3()
{
  typedef geom::SimplexJac<3, _Float> TetJac;
  typedef typename TetJac::Vertex Vertex;
  typedef typename TetJac::Simplex Simplex;

  {
    // Default constructor.
    TetJac tet;
  }
  {
    TetJac tet;
    Vertex v[4];
    v[0] = Vertex{{0, 0, 0}};
    v[1] = Vertex{{1, 0, 0}};
    v[2] = Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}};
    v[3] = Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}};
    for (std::size_t a = 0; a != 4; ++a) {
      for (std::size_t b = 0; b != 4; ++b) {
        if (b != a) {
          for (std::size_t c = 0; c != 4; ++c) {
            if (c != a && c != b) {
              for (std::size_t d = 0; d != 4; ++d) {
                if (d != a && d != b && d != c) {
                  tet.set(Simplex{{v[a], v[b], v[c], v[d]}});
                  std::cout << "Permutation "
                            << a << " " << b << " " << c << " " << d << ": "
                            << tet.getDeterminant() << '\n';
                }
              }
            }
          }
        }
      }
    }
  }
  {
    Simplex t = {{Vertex{{0, 0, 0}},
                  Vertex{{1, 0, 0}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}}}};
    TetJac tet(t);
    std::cout << "Identity tetrahedron:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c =
      geom::computeContent(Vertex{{0, 0, 0}},
                           Vertex{{1, 0, 0}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex t = {{Vertex{{0, 0, 0}},
                  Vertex{{1, 0, 0}},
                  Vertex{{0, 1, 0}},
                  Vertex{{0, 0, 1}}}};
    TetJac tet(t);
    std::cout << "Reference tetrahedron:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0, 0, 0}},
                                          Vertex{{1, 0, 0}},
                                          Vertex{{0, 1, 0}},
                                          Vertex{{0, 0, 1}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex t = {{Vertex{{0, 0, 0}},
                  Vertex{{10, 0, 0}},
                  Vertex{{0, 10, 0}},
                  Vertex{{0, 0, 10}}}};
    TetJac tet(t);
    std::cout << "Scaled reference tetrahedron:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0, 0, 0}},
                                          Vertex{{10, 0, 0}},
                                          Vertex{{0, 10, 0}},
                                          Vertex{{0, 0, 10}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex t = {{Vertex{{0, 0, 0}},
                  Vertex{{1, 0, 0}},
                  Vertex{{0, 1, 0}},
                  Vertex{{1, 1, _Float(1e-8)}}}};
    TetJac tet(t);
    std::cout << "Almost flat tetrahedron:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c = geom::computeContent(Vertex{{0, 0, 0}},
                                          Vertex{{1, 0, 0}},
                                          Vertex{{0, 1, 0}},
                                          Vertex{{1, 1, _Float(1e-8)}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex t = {{Vertex{{0, 0, 0}},
                  Vertex{{1, 0, 0}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}}}};
    TetJac tet(t);
    std::cout << "Inverted identity tetrahedron:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c =
      geom::computeContent(Vertex{{0, 0, 0}},
                           Vertex{{1, 0, 0}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::abs(c) * std::numeric_limits<_Float>::epsilon());
  }
  {
    Simplex t = {{Vertex{{1, 0, 0}},
                  Vertex{{1, 0, 0}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}},
                  Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}}}};
    TetJac tet(t);
    std::cout << "Flat tetrahedron, two vertices coincide:\n"
              << "det = " << tet.getDeterminant()
              << "\ngrad det = " << tet.getGradientDeterminant()
              << "\ncontent = " << tet.computeContent()
              << "\ngrad content = " << tet.computeGradientContent()
              << '\n' << '\n';
    const _Float c =
      geom::computeContent(Vertex{{1, 0, 0}},
                           Vertex{{1, 0, 0}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 2, 0}},
                           Vertex{{_Float(1) / 2, std::sqrt(_Float(3)) / 6, std::sqrt(_Float(2) / 3)}});
    assert(std::abs(c - tet.computeContent()) <
           10.0 * std::numeric_limits<_Float>::epsilon());
  }
}


int
main()
{
  test2<float>();
  test2<double>();
  test3<float>();
  test3<double>();

  return 0;
}
