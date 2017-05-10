// -*- C++ -*-

#include "stlib/geom/mesh/iss/ISS_SignedDistance.h"
#include "stlib/geom/mesh/iss/IndSimpSet.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{
  typedef geom::IndSimpSet<2, 1> ISS;
  typedef ISS::Vertex P;
  typedef geom::ISS_SignedDistance<ISS> ISS_SD;
  typedef geom::ISS_SD_ClosestPointDirection<ISS> ISS_SD_CPD;

  const double eps = 10 * std::numeric_limits<double>::epsilon();

  {
    //
    // Data for a square.
    //
    const std::size_t numVertices = 4;
    double vertices[] = { 0, 0,   // 0
                          1, 0,    // 1
                          1, 1,    // 2
                          0, 1
                        };  // 3
    const std::size_t numSimplices = 4;
    std::size_t simplices[] = { 0, 1,
                                1, 2,
                                2, 3,
                                3, 0
                              };

    ISS iss;
    geom::build(&iss, numVertices, vertices, numSimplices, simplices);
    ISS_SD x(iss);
    P cp, grad;
    std::size_t index;

    assert(std::abs(x(P{{0, 0}}) - 0) < eps);
    assert(std::abs(x(P{{0, 0}}, &cp) - 0) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(std::abs(x(P{{0, 0}}, &cp, &grad,
                      &index) - 0) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    // Gradient is between (-1,0) and (0,-1).
    assert(index == 0 || index == 3);

    assert(std::abs(x(P{{0, -1}}) - 1) < eps);
    assert(std::abs(x(P{{0, -1}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(std::abs(x(P{{0, -1}}, &cp, &grad,
                      &index) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(geom::computeDistance(grad, P{{0, -1}}) < eps);
    assert(index == 0 || index == 3);

    assert(std::abs(x(P{{-1, 0}}) - 1) < eps);
    assert(std::abs(x(P{{-1, 0}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(std::abs(x(P{{-1, 0}}, &cp, &grad,
                      &index) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(geom::computeDistance(grad, P{{-1, 0}}) < eps);
    assert(index == 0 || index == 3);

    assert(std::abs(x(P{{0.5, -1}}) - 1) < eps);
    assert(std::abs(x(P{{0.5, -1}}, &cp) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0.5, 0}}) < eps);
    assert(std::abs(x(P{{0.5, -1}}, &cp, &grad,
                      &index) - 1) < eps);
    assert(geom::computeDistance(cp, P{{0.5, 0}}) < eps);
    assert(geom::computeDistance(grad, P{{0, -1}}) < eps);
    assert(index == 0);

    assert(std::abs(x(P{{0.5, 0.25}}) + 0.25) < eps);
    assert(std::abs(x(P{{0.5, 0.25}}, &cp) + 0.25) < eps);
    assert(geom::computeDistance(cp, P{{0.5, 0}}) < eps);
    assert(std::abs(x(P{{0.5, 0.25}}, &cp, &grad,
                      &index) + 0.25) < eps);
    assert(geom::computeDistance(cp, P{{0.5, 0}}) < eps);
    assert(geom::computeDistance(grad, P{{0, -1}}) < eps);
    assert(index == 0);

    assert(std::abs(x(P{{-3, -4}}) - 5) < eps);
    assert(std::abs(x(P{{-3, -4}}, &cp) - 5) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(std::abs(x(P{{-3, -4}}, &cp, &grad,
                      &index) - 5) < eps);
    assert(geom::computeDistance(cp, P{{0, 0}}) < eps);
    assert(geom::computeDistance(grad, P{{-3. / 5.,
                                 -4. / 5.}}) < eps);
    assert(index == 0 || index == 3);


    geom::ISS_SD_Distance<ISS> df(x);
    assert(std::abs(df(P{{0, 0}}) - 0) < eps);

    ISS_SD_CPD cpd(x);
    std::cout << cpd(P{{0.4, 0.4}},
                     P{{-1.0 / std::sqrt(2.0), -1.0 / std::sqrt(2.0)}})
              << "\n";
    std::cout << cpd(P{{0.4, 0.3}},
                     P{{0.0 , -1.0}})
              << "\n";
  }

  return 0;
}
