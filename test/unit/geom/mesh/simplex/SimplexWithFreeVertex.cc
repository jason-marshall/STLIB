// -*- C++ -*-

#include "stlib/geom/mesh/simplex/SimplexWithFreeVertex.h"

#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include <iostream>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace geom;

  typedef SimplexWithFreeVertex<SimplexMeanRatio, 3> MR;
  typedef SimplexWithFreeVertex<SimplexModMeanRatio, 3> MMR;
  typedef SimplexWithFreeVertex<SimplexCondNum, 3> CN;
  typedef SimplexWithFreeVertex<SimplexModCondNum, 3> MCN;

  typedef MR::Vertex Vertex;
  typedef MR::Face Face;

  Vertex grad;

  {
    // Default constructor.
    MR simplex;
  }
  {
    Face face = {{{{1., 0., 0.}},
                 {{1. / 2, std::sqrt(3.) / 2, 0.}},
                  {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
    MR mr(face);
    MMR mmr(face);
    CN cn(face);
    MCN mcn(face);

    mr.set(Vertex{{0., 0., 0.}});
    mmr.set(Vertex{{0., 0., 0.}});
    cn.set(Vertex{{0., 0., 0.}});
    mcn.set(Vertex{{0., 0., 0.}});

    std::cout << "Identity tetrahedron:\n"
              << "det = " << mr.getDeterminant() << '\n'
              << "content = " << mr.computeContent() << '\n'
              << "eta = " << mr() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa = " << cn() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mr.computeGradient(&grad);
    std::cout << "D eta = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    cn.computeGradient(&grad);
    std::cout << "D kappa = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }
  {
    Face face = {{{{1., 0., 0.}},
                 {{0., 1., 0.}},
                 {{0., 0., 1.}}}};
    MR mr(face);
    MMR mmr(face);
    CN cn(face);
    MCN mcn(face);

    mr.set(Vertex{{0., 0., 0.}});
    mmr.set(Vertex{{0., 0., 0.}});
    cn.set(Vertex{{0., 0., 0.}});
    mcn.set(Vertex{{0., 0., 0.}});

    std::cout << "Reference tetrahedron:\n"
              << "det = " << mr.getDeterminant() << '\n'
              << "content = " << mr.computeContent() << '\n'
              << "eta = " << mr() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa = " << cn() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mr.computeGradient(&grad);
    std::cout << "D eta = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    cn.computeGradient(&grad);
    std::cout << "D kappa = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }
  {
    Face face = {{{{10., 0., 0.}},
                 {{0., 10., 0.}},
                 {{0., 0., 10.}}}};
    MR mr(face);
    MMR mmr(face);
    CN cn(face);
    MCN mcn(face);

    mr.set(Vertex{{0., 0., 0.}});
    mmr.set(Vertex{{0., 0., 0.}});
    cn.set(Vertex{{0., 0., 0.}});
    mcn.set(Vertex{{0., 0., 0.}});

    std::cout << "Scaled reference tetrahedron:\n"
              << "det = " << mr.getDeterminant() << '\n'
              << "content = " << mr.computeContent() << '\n'
              << "eta = " << mr() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa = " << cn() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mr.computeGradient(&grad);
    std::cout << "D eta = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    cn.computeGradient(&grad);
    std::cout << "D kappa = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }
  {
    Face face = {{{{1., 0., 0.}},
                 {{0., 1., 0.}},
                 {{1., 1., 1e-8}}}};
    MR mr(face);
    MMR mmr(face);
    CN cn(face);
    MCN mcn(face);

    mr.set(Vertex{{0., 0., 0.}});
    mmr.set(Vertex{{0., 0., 0.}});
    cn.set(Vertex{{0., 0., 0.}});
    mcn.set(Vertex{{0., 0., 0.}});

    std::cout << "Almost flat tetrahedron:\n"
              << "det = " << mr.getDeterminant() << '\n'
              << "content = " << mr.computeContent() << '\n'
              << "eta = " << mr() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa = " << cn() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mr.computeGradient(&grad);
    std::cout << "D eta = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    cn.computeGradient(&grad);
    std::cout << "D kappa = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }
  {
    Face face = {{{{1., 0., 0.}},
                 {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}},
                 {{1. / 2, std::sqrt(3.) / 2, 0.}}}};
    MMR mmr(face);
    MCN mcn(face);

    mmr.set(Vertex{{0., 0., 0.}});
    mcn.set(Vertex{{0., 0., 0.}});

    std::cout << "Inverted identity tetrahedron:\n"
              << "det = " << mmr.getDeterminant() << '\n'
              << "content = " << mmr.computeContent() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mmr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }
  {
    Face face = {{{{1., 0., 0.}},
                 {{1. / 2, std::sqrt(3.) / 2, 0.}},
                  {{1. / 2, std::sqrt(3.) / 6, std::sqrt(2. / 3.)}}}};
    MMR mmr(face);
    MCN mcn(face);

    mmr.set(Vertex{{1., 0., 0.}});
    mcn.set(Vertex{{1., 0., 0.}});

    std::cout << "Flat tetrahedron, two vertices coincide:\n"
              << "det = " << mmr.getDeterminant() << '\n'
              << "content = " << mmr.computeContent() << '\n'
              << "eta_m = " << mmr() << '\n'
              << "kappa_m = " << mcn() << '\n';

    mmr.computeGradientOfContent(&grad);
    std::cout << "D content = " << grad << '\n';
    mmr.computeGradient(&grad);
    std::cout << "D eta_m = " << grad << '\n';
    mcn.computeGradient(&grad);
    std::cout << "D kappa_m = " << grad << '\n' << '\n';
  }

  return 0;
}
