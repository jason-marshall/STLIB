// -*- C++ -*-

#include "stlib/numerical/specialFunctions/LogarithmOfFactorial.h"

#include <iostream>
#include <limits>

using namespace stlib;

int
main()
{
  {
    const int Size = 10;
    numerical::LogarithmOfFactorial<> f;
    const double Epsilon = 10 * std::numeric_limits<double>::epsilon();

    int factorial = 1;
    double logarithmOfFactorial;
    for (int n = 0; n != Size; ++n) {
      logarithmOfFactorial = std::log(factorial);
      assert(std::abs(f(n) - logarithmOfFactorial) <=
             logarithmOfFactorial * Epsilon);
      factorial *= n + 1;
    }
  }

  {
    const int Arguments[] = {0,
                             1,
                             10,
                             100,
                             1000,
                             10000,
                             100000,
                             1000000
                            };
    const int Size = sizeof(Arguments) / sizeof(int);
    const double LogFactorial[] = {0, 0, 15.104412573075515295, 363.73937555556349014,
                                   5912.1281784881633489, 82108.927836814353455, 1.0512992218991218651e6,
                                   1.2815518384658169624e7
                                  };

    std::cout << "LogarithmOfFactorial.txt\n";
    numerical::LogarithmOfFactorial<> f;
    int x;
    double v, r;
    for (int i = 0; i != Size; ++i) {
      x = Arguments[i];
      v = f(x);
      if (LogFactorial[i] != 0) {
        r = (v - LogFactorial[i]) / LogFactorial[i];
      }
      else {
        r = v - LogFactorial[i];
      }
      std::cout << x << " " << v << " " << r << "\n";
    }
  }

  return 0;
}
