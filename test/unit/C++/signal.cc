// -*- C++ -*-

#include <iostream>
#include <limits>

int
main()
{
  {
    double x, y;
    x = 1;
    y = 0;
    std::cout << x << " / " << y << " = " << x / y << '\n';
    x = 0;
    y = 0;
    std::cout << x << " / " << y << " = " << x / y << '\n';
    x = std::numeric_limits<double>::max();
    y = std::numeric_limits<double>::max();
    std::cout << x << " * " << y << " = " << x * y << '\n';
  }
#if 0
  // This will cause a floating point exception.
  {
    int x, y;
    x = 1;
    y = 0;
    std::cout << x << " / " << y << " = " << x / y << '\n';
  }
#endif

  return 0;
}
