// -*- C++ -*-

#include "stlib/particle/coordinates.h"


USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
using namespace stlib;

template<typename _Float>
void
testPlain()
{
  const _Float Eps = std::numeric_limits<_Float>::epsilon();
  {
    // 1-D
    const std::size_t D = 1;
    typedef std::array<_Float, D> Point;
    typedef particle::DiscreteCoordinates<_Float, D, false>
    DiscreteCoordinates;
    typedef geom::BBox<_Float, D> BBox;
    const std::array<BBox, 2> domains = {{
        {{{0}}, {{1}}},
        {{{2}}, {{5}}}
      }
    };
    for (std::size_t i = 0; i != domains.size(); ++i) {
      const _Float length = domains[i].upper[0] - domains[i].lower[0];
      // Specify the size for a cell.
      {
        const DiscreteCoordinates f(domains[i], 10 * length);
        Point lc = domains[i].lower;
        lc -= _Float(4.5) * length;
        assert(stlib::ext::euclideanDistance(f.lowerCorner(), lc) < 10 * Eps);
        assert(f.cellLengths()[0] == 10 * length);
        assert(f.numLevels() == 0);

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        assert(g.cellLengths()[0] == 10 * length);
        g.setLevels(1);
        assert(g.numLevels() == 1);
        assert(g.cellLengths()[0] == 10 * length / 2);
      }
      {
        const DiscreteCoordinates f(domains[i], length);
        assert(f.lowerCorner() == domains[i].lower);
        assert(f.cellLengths()[0] == length);
        assert(f.numLevels() == 0);

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        assert(g.cellLengths()[0] == length);
        g.setLevels(1);
        assert(g.numLevels() == 1);
        assert(g.cellLengths()[0] == length / 2);
      }
      {
        const DiscreteCoordinates f(domains[i], length / 2);
        assert(f.lowerCorner() == domains[i].lower);
        assert(f.cellLengths()[0] == length / 2);
        assert(f.numLevels() == 1);

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        assert(g.cellLengths()[0] == length);
        g.setLevels(1);
        assert(g.numLevels() == 1);
        assert(g.cellLengths()[0] == length / 2);
        g.setLevels(2);
        assert(g.numLevels() == 2);
        assert(g.cellLengths()[0] == length / 4);
      }
      {
        const _Float cellLength = length / 4;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.lowerCorner() == domains[i].lower);
        assert(f.cellLengths()[0] == cellLength);
        assert(f.numLevels() == 2);

        assert(f.level(1 * length) == 0);
        assert(f.level(0.75 * length) == 0);
        assert(f.level(0.5 * length) == 1);
        assert(f.level(0.4 * length) == 1);
        assert(f.level(0.25 * length) == 2);
        assert(f.level(0) == f.numLevels());
      }
    }
  }
  {
    // 2-D
    const std::size_t D = 2;
    typedef std::array<_Float, D> Point;
    typedef particle::DiscreteCoordinates<_Float, D, false>
    DiscreteCoordinates;
    typedef geom::BBox<_Float, D> BBox;
    const std::array<BBox, 2> domains = {{
        {{{0, 0}}, {{1, 1}}},
        {{{2, 3}}, {{5, 7}}}
      }
    };
    for (std::size_t i = 0; i != domains.size(); ++i) {
      const _Float length = ext::max(domains[i].upper - domains[i].lower);
      // Specify the size for a cell.
      {
        const _Float cellLength = 10 * length;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.numLevels() == 0);
        Point lc = _Float(0.5) * (domains[i].lower + domains[i].upper) -
                   _Float(0.5) * cellLength;
        //std::cerr << f.lowerCorner() << ", " << lc << '\n'
        //          << f.lengths() << '\n';
        assert(stlib::ext::euclideanDistance(f.lowerCorner(), lc) < 10 * Eps);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.lengths()[j] == cellLength);
          assert(f.cellLengths()[j] == cellLength);
        }
      }
      {
        const _Float cellLength = length;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.numLevels() == 0);
        Point lc = _Float(0.5) * (domains[i].lower + domains[i].upper) -
                   _Float(0.5) * cellLength;
        assert(stlib::ext::euclideanDistance(f.lowerCorner(), lc) < 10 * Eps);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }
      }
      {
        const _Float cellLength = length / 2;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.numLevels() == 1);
        Point lc = _Float(0.5) * (domains[i].lower + domains[i].upper) -
                   cellLength;
        assert(stlib::ext::euclideanDistance(f.lowerCorner(), lc) < 10 * Eps);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }
      }
      {
        const _Float cellLength = length / 4;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.numLevels() == 2);
        Point lc = _Float(0.5) * (domains[i].lower + domains[i].upper) -
                   _Float(2) * cellLength;
        assert(stlib::ext::euclideanDistance(f.lowerCorner(), lc) < 10 * Eps);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }

        assert(f.level(1 * length) == 0);
        assert(f.level(0.75 * length) == 0);
        assert(f.level(0.5 * length) == 1);
        assert(f.level(0.4 * length) == 1);
        assert(f.level(0.25 * length) == 2);
        assert(f.level(0) == f.numLevels());

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length);
        }
        g.setLevels(1);
        assert(g.numLevels() == 1);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 2);
        }
        g.setLevels(2);
        assert(g.numLevels() == 2);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 4);
        }
      }
    }
  }
}


template<typename _Float>
void
testPeriodic()
{
  {
    // 1-D
    const std::size_t D = 1;
    typedef particle::DiscreteCoordinates<_Float, D, true>
    DiscreteCoordinates;
    typedef geom::BBox<_Float, D> BBox;
    const std::array<BBox, 2> domains = {{
        {{{0}}, {{1}}},
        {{{2}}, {{5}}}
      }
    };
    for (std::size_t i = 0; i != domains.size(); ++i) {
      const _Float length = domains[i].upper[0] - domains[i].lower[0];
      // Specify the size for a cell.
      {
        const DiscreteCoordinates f(domains[i], length);
        assert(f.numLevels() == 0);
        assert(f.lowerCorner() == domains[i].lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == length);
        }
      }
      {
        const DiscreteCoordinates f(domains[i], 0.51 * length);
        assert(f.numLevels() == 0);
        assert(f.lowerCorner() == domains[i].lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == length);
        }
      }
      {
        const DiscreteCoordinates f(domains[i], length / 2);
        assert(f.numLevels() == 1);
        assert(f.lowerCorner() == domains[i].lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == length / 2);
        }
      }
      {
        const DiscreteCoordinates f(domains[i], 1.01 * length / 3);
        assert(f.numLevels() == 1);
        assert(f.lowerCorner() == domains[i].lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == length / 2);
        }
      }
      {
        const _Float cellLength = length / 4;
        const DiscreteCoordinates f(domains[i], cellLength);
        assert(f.numLevels() == 2);
        assert(f.lowerCorner() == domains[i].lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }

        assert(f.level(1 * length) == 0);
        assert(f.level(0.75 * length) == 0);
        assert(f.level(0.5 * length) == 1);
        assert(f.level(0.4 * length) == 1);
        assert(f.level(0.25 * length) == 2);
        assert(f.level(0) == f.numLevels());

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length);
        }
        g.setLevels(1);
        assert(g.numLevels() == 1);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 2);
        }
        g.setLevels(2);
        assert(g.numLevels() == 2);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 4);
        }
      }
    }
  }
  {
    // 2-D
    const std::size_t D = 2;
    typedef particle::DiscreteCoordinates<_Float, D, true>
    DiscreteCoordinates;
    typedef typename particle::TemplatedTypes<_Float, D>::Point Point;
    typedef typename particle::TemplatedTypes<_Float, D>::DiscretePoint
      DiscretePoint;
    typedef geom::BBox<_Float, D> BBox;
    {
      const BBox domain = {{{0, 0}}, {{1, 1}}};
      const _Float length = stlib::ext::max(domain.upper - domain.lower);
      // Specify the size for a cell.
      {
        const _Float cellLength = length / 2;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 1);
        assert(f.lowerCorner() == domain.lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }
      }
      {
        const _Float cellLength = length / 4;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 2);
        assert(f.lowerCorner() == domain.lower);
        for (std::size_t j = 0; j != D; ++j) {
          assert(f.cellLengths()[j] == cellLength);
        }

        assert(f.level(1 * length) == 0);
        assert(f.level(0.75 * length) == 0);
        assert(f.level(0.5 * length) == 1);
        assert(f.level(0.4 * length) == 1);
        assert(f.level(0.25 * length) == 2);
        assert(f.level(0) == f.numLevels());

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length);
        }
        g.setLevels(1);
        assert(g.numLevels() == 1);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 2);
        }
        g.setLevels(2);
        assert(g.numLevels() == 2);
        for (std::size_t j = 0; j != D; ++j) {
          assert(g.cellLengths()[j] == length / 4);
        }
      }
    }
    {
      const BBox domain = {{{2, 3}}, {{5, 7}}};
      // Specify the size for a cell.
      {
        const _Float cellLength = 3;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 0);
        assert(f.cellExtents() ==
               (DiscretePoint{{1, 1}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{3, 4}}));
      }
      {
        const _Float cellLength = 2;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 1);
        assert(f.cellExtents() ==
               (DiscretePoint{{1, 2}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{3, 2}}));
      }
      {
        const _Float cellLength = 1.5;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 1);
        assert(f.cellExtents() ==
               (DiscretePoint{{2, 2}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{1.5, 2}}));
      }
      {
        const _Float cellLength = 1;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 2);
        assert(f.cellExtents() ==
               (DiscretePoint{{3, 4}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{1, 1}}));
      }
      {
        const _Float cellLength = 0.75;
        const DiscreteCoordinates f(domain, cellLength);
#if 0
        std::cerr << f.numLevels() << '\n'
                  << f.cellExtents() << '\n'
                  << f.lowerCorner() << '\n'
                  << f.lengths() << '\n'
                  << f.cellLengths() << '\n';
#endif
        assert(f.numLevels() == 3);
        assert(f.cellExtents() ==
               (DiscretePoint{{4, 5}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{0.75, 0.8}}));
      }
      {
        const _Float cellLength = 0.5;
        const DiscreteCoordinates f(domain, cellLength);
        assert(f.numLevels() == 3);
        assert(f.cellExtents() ==
               (DiscretePoint{{6, 8}}));
        assert(f.lowerCorner() == domain.lower);
        assert(f.lengths() == (Point{{3, 4}}));
        assert(f.cellLengths() == (Point{{0.5, 0.5}}));
      }
      {
        const _Float cellLength = 1;
        const DiscreteCoordinates f(domain, cellLength);

        const _Float length = 3;
        assert(f.level(1 * length) == 0);
        assert(f.level(0.75 * length) == 0);
        assert(f.level(0.5 * length) == 1);
        assert(f.level(0.4 * length) == 1);
        assert(f.level(0.25 * length) == 2);
        assert(f.level(0) == f.numLevels());

        DiscreteCoordinates g = f;
        g.setLevels(0);
        assert(g.numLevels() == 0);
        assert(g.cellLengths() == (Point{{4, 4}}));

        g = f;
        g.setLevels(1);
        assert(g.numLevels() == 1);
        assert(g.cellLengths() == (Point{{2, 2}}));

        g = f;
        g.setLevels(2);
        assert(g.numLevels() == 2);
        assert(g.cellLengths() == (Point{{1, 1}}));
      }
    }
  }
}


int
main()
{
  testPlain<float>();
  testPlain<double>();
  testPeriodic<float>();
  testPeriodic<double>();

  return 0;
}
