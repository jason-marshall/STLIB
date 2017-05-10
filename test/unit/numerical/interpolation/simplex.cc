// -*- C++ -*-

#include "stlib/numerical/interpolation/simplex.h"

#include <iostream>
#include <limits>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace numerical;

  const double eps = 100 * std::numeric_limits<double>::epsilon();

  //
  // 1-D simplex.  1-D space.
  //
  assert(std::abs(linear_interpolation(0.0, 1.0, 0.0, 1.0, 0.0) - 0.0)
         < eps);
  assert(std::abs(linear_interpolation(0.0, 1.0, 0.0, 1.0, 0.5) - 0.5)
         < eps);
  assert(std::abs(linear_interpolation(0.0, 1.0, 0.0, 1.0, 1.0) - 1.0)
         < eps);
  assert(std::abs(linear_interpolation(0.0, 1.0, 0.0, 1.0, -1.0) + 1.0)
         < eps);
  assert(std::abs(linear_interpolation(0.0, 1.0, 0.0, 1.0, 2.0) - 2.0)
         < eps);
  assert(std::abs(linear_interpolation(1.0, 3.0, 5.0, 7.0, 2.0) - 6.0)
         < eps);

  assert(std::abs(linear_interpolation(std::array<double, 2>{{0.0, 1.0}},
                                       std::array<double, 2>{{0.0, 1.0}},
                                       0.0) - 0.0) < eps);

  {
    std::array<std::array<double, 1>, 2> pos = {{{{0.0}}, {{1.0}}}};
    std::array<double, 2> val = {{0.0, 1.0}};
    std::array<double, 1> loc = {{0.0}};
    assert(std::abs(linear_interpolation(pos, val, loc) - 0.0) < eps);
  }


  //
  // 1-D simplex.  2-D space.
  //
  {
    typedef std::array<double, 2> Pt;

    assert(std::abs(linear_interpolation(Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}},
                                         2.0, 3.0,
                                         Pt{{0.0, 0.0}}) - 2.5) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}},
                                         2.0, 3.0,
                                         Pt{{1.0, 0.0}}) - 2.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}},
                                         2.0, 3.0,
                                         Pt{{0.0, 1.0}}) - 3.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}},
                                         2.0, 3.0,
                                         Pt{{1.0, 2.0}}) - 3.0) < eps);

    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{1.0, 2.0}}) - 2.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{3.0, 6.0}}) - 4.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{2.0, 4.0}}) - 3.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{-1.0, 3.0}}) - 2.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{0.0, 5.0}}) - 3.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{3.0, 6.0}},
                                         2.0, 4.0,
                                         Pt{{1.0, 7.0}}) - 4.0) < eps);

    std::array<Pt, 2> pos = {{Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}}}};
    std::array<double, 2> val = {{2.0, 3.0}};
    Pt loc = {{0.0, 0.0}};
    assert(std::abs(linear_interpolation(pos, val, loc) - 2.5) < eps);
  }

  //
  // 2-D simplex.  2-D space.
  //
  {
    typedef std::array<double, 2> Pt;
    // 2 x + 3 y
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0}}, Pt{{1.0, 0.0}},
                                         Pt{{0.0, 1.0}},
                                         0.0, 2.0, 3.0,
                                         Pt{{0.0, 0.0}}) - 0.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0}}, Pt{{1.0, 0.0}},
                                         Pt{{0.0, 1.0}},
                                         0.0, 2.0, 3.0,
                                         Pt{{1.0, 0.0}}) - 2.0) < eps);
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0}}, Pt{{1.0, 0.0}},
                                         Pt{{0.0, 1.0}},
                                         0.0, 2.0, 3.0,
                                         Pt{{0.0, 1.0}}) - 3.0) < eps);

    assert(std::abs(linear_interpolation(Pt{{1.0, 2.0}}, Pt{{4.0, 3.0}},
                                         Pt{{2.0, 7.0}},
                                         8.0, 17.0, 25.0,
                                         Pt{{4.0, 5.0}}) - 23.0) < eps);

    std::array<Pt, 3> pos = {{Pt{{0.0, 0.0}}, Pt{{1.0, 0.0}}, Pt{{0.0, 1.0}}}};
    std::array<double, 3> val = {{0.0, 2.0, 3.0}};
    Pt loc = {{0.0, 0.0}};
    assert(std::abs(linear_interpolation(pos, val, loc) - 0.0) < eps);
  }

  //
  // 3-D simplex.  3-D space.
  //
  {
    typedef std::array<double, 3> Pt;
    // 1 + 2 x + 3 y + 5 z
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0, 0.0}},
                                         Pt{{1.0, 0.0, 0.0}},
                                         Pt{{0.0, 1.0, 0.0}},
                                         Pt{{0.0, 0.0, 1.0}},
                                         1.0, 3.0, 4.0, 6.0,
                                         Pt{{0.0, 0.0, 0.0}}) - 1.0)
           < eps);
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0, 0.0}},
                                         Pt{{1.0, 0.0, 0.0}},
                                         Pt{{0.0, 1.0, 0.0}},
                                         Pt{{0.0, 0.0, 1.0}},
                                         1.0, 3.0, 4.0, 6.0,
                                         Pt{{1.0, 0.0, 0.0}}) - 3.0)
           < eps);
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0, 0.0}},
                                         Pt{{1.0, 0.0, 0.0}},
                                         Pt{{0.0, 1.0, 0.0}},
                                         Pt{{0.0, 0.0, 1.0}},
                                         1.0, 3.0, 4.0, 6.0,
                                         Pt{{0.0, 1.0, 0.0}}) - 4.0)
           < eps);
    assert(std::abs(linear_interpolation(Pt{{0.0, 0.0, 0.0}},
                                         Pt{{1.0, 0.0, 0.0}},
                                         Pt{{0.0, 1.0, 0.0}},
                                         Pt{{0.0, 0.0, 1.0}},
                                         1.0, 3.0, 4.0, 6.0,
                                         Pt{{0.0, 0.0, 1.0}}) - 6.0)
           < eps);


    std::array<Pt, 4> pos = {{
        Pt{{0.0, 0.0, 0.0}}, Pt{{1.0, 0.0, 0.0}},
        Pt{{0.0, 1.0, 0.0}}, Pt{{0.0, 0.0, 1.0}}
      }
    };
    std::array<double, 4> val = {{1.0, 3.0, 4.0, 6.0}};
    Pt loc = {{0.0, 0.0, 0.0}};
    assert(std::abs(linear_interpolation(pos, val, loc) - 1.0) < eps);
  }

  return 0;
}
