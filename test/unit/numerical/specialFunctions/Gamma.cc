// -*- C++ -*-

#include "stlib/numerical/specialFunctions/Gamma.h"

#include "stlib/numerical/specialFunctions/LogarithmOfFactorial.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  {
    const double Epsilon = std::sqrt(std::numeric_limits<double>::epsilon());

    numerical::LogarithmOfGamma<> f;
    assert(std::abs(f(1) - std::log(1.0)) < Epsilon);
    assert(std::abs(f(2) - std::log(1.0)) < Epsilon);
    assert(std::abs(f(3) - std::log(2.0)) < Epsilon);
    assert(std::abs(f(4) - std::log(6.0)) < Epsilon);
    assert(std::abs(f(5) - std::log(24.0)) < Epsilon);
    f = numerical::constructLogarithmOfGamma<double>();

    const double Values[] = {1e0, 1e1, 1e2, 1e3, 1e4, 1e5};
    const int NumberOfValues = sizeof(Values) / sizeof(double);

    numerical::LogarithmOfFactorial<> g;
    std::cout << "x is computed from LogarithmOfGamma.\n"
              << "y is computed from LogarithmOfFactorial.\n";

    double x, y, value;
    for (int i = 0; i != NumberOfValues; ++i) {
      value = Values[i];
      x = f(value + 1);
      y = g(int(value));
      std::cout << "x = " << x << ", y = " << y
                << ", x - y = " << x - y << "\n";
    }
  }

  {
    const double Arguments[] = {1e-8,
                                1e-7,
                                1e-6,
                                1e-5,
                                1e-4,
                                1e-3,
                                1e-2,
                                1e-1,
                                1e0,
                                1e1,
                                1e2,
                                1e3,
                                1e4,
                                1e5,
                                1e6
                               };
    const int Size = sizeof(Arguments) / sizeof(double);
    const double LogGamma[] = {18.420680738180208905, 16.118095593236761523, 13.815509980749431669,
                               11.512919692895825707, 9.2102826586339622584, 6.9071788853838536825,
                               4.5994798780420217225, 2.2527126517342059599, 0, 12.801827480081469611,
                               359.13420536957539878, 5905.2204232091812118, 82099.717496442377273,
                               1.0512877089736568949e6, 1.281550456914761166e7
                              };

    std::cout << "Gamma.txt\n";
    numerical::LogarithmOfGamma<> f;
    double x, v, r;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      v = f(x);
      if (LogGamma[i] != 0) {
        r = (v - LogGamma[i]) / LogGamma[i];
      }
      else {
        r = v - LogGamma[i];
      }
      std::cout << x << " " << v << " " << r << "\n";
    }
  }
  return 0;
}
