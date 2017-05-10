// -*- C++ -*-

#include "stlib/numerical/integer/print.h"
#include <sstream>

using namespace stlib;

template<typename _Integer>
void
test()
{
  using numerical::printBits;
  {
    std::ostringstream out;
    printBits(out, _Integer(0));
    std::ostringstream solution;
    for (std::size_t i = 0;
         i != std::size_t(std::numeric_limits<_Integer>::digits); ++i) {
      solution << '0';
    }
    assert(out.str() == solution.str());
  }
  {
    std::ostringstream out;
    printBits(out, _Integer(1));
    std::ostringstream solution;
    for (std::size_t i = 0;
         i != std::size_t(std::numeric_limits<_Integer>::digits - 1); ++i) {
      solution << '0';
    }
    solution << '1';
    assert(out.str() == solution.str());
  }
  {
    std::ostringstream out;
    printBits(out, std::numeric_limits<_Integer>::max());
    std::ostringstream solution;
    for (std::size_t i = 0;
         i != std::size_t(std::numeric_limits<_Integer>::digits); ++i) {
      solution << '1';
    }
    assert(out.str() == solution.str());
  }
}

int
main()
{
  using numerical::printBits;

  {
    std::cout << "unsigned char\n";
    typedef unsigned char Integer;
    test<Integer>();

    const Integer x[] = {0, 1, std::numeric_limits<Integer>::max()};
    const char* names[] = {"0", "1", "Max"};

    const std::size_t begin = 0;
    const std::size_t end = std::numeric_limits<Integer>::digits;
    const std::size_t middle = end / 2;

    for (std::size_t i = 0; i != sizeof(x) / sizeof(Integer); ++i) {
      std::cout << names[i] << " = ";
      printBits(std::cout, x[i]);
      std::cout << " = ";
      printBits(std::cout, x[i], middle, end);
      std::cout << " + ";
      printBits(std::cout, x[i], begin, middle);
      std::cout << "\n";
    }
  }

  {
    std::cout << "unsigned short\n";
    typedef unsigned short Integer;
    test<Integer>();

    const Integer x[] = {0, 1, std::numeric_limits<Integer>::max()};
    const char* names[] = {"0", "1", "Max"};

    const std::size_t begin = 0;
    const std::size_t end = std::numeric_limits<Integer>::digits;
    const std::size_t middle = end / 2;

    for (std::size_t i = 0; i != sizeof(x) / sizeof(Integer); ++i) {
      std::cout << names[i] << " = ";
      printBits(std::cout, x[i]);
      std::cout << " = ";
      printBits(std::cout, x[i], middle, end);
      std::cout << " + ";
      printBits(std::cout, x[i], begin, middle);
      std::cout << "\n";
    }
  }

  {
    std::cout << "unsigned int\n";
    typedef unsigned int Integer;
    test<Integer>();

    const Integer x[] = {0, 1, std::numeric_limits<Integer>::max()};
    const char* names[] = {"0", "1", "Max"};

    const std::size_t begin = 0;
    const std::size_t end = std::numeric_limits<Integer>::digits;
    const std::size_t middle = end / 2;

    for (std::size_t i = 0; i != sizeof(x) / sizeof(Integer); ++i) {
      std::cout << names[i] << " = ";
      printBits(std::cout, x[i]);
      std::cout << " = ";
      printBits(std::cout, x[i], middle, end);
      std::cout << " + ";
      printBits(std::cout, x[i], begin, middle);
      std::cout << "\n";
    }
  }

  {
    std::cout << "unsigned long\n";
    typedef unsigned long Integer;
    test<Integer>();

    const Integer x[] = {0, 1, 31, std::numeric_limits<Integer>::max()};
    const char* names[] = {"0", "1", "31", "Max"};

    const std::size_t begin = 0;
    const std::size_t end = std::numeric_limits<Integer>::digits;
    const std::size_t middle = end / 2;

    for (std::size_t i = 0; i != sizeof(x) / sizeof(Integer); ++i) {
      std::cout << names[i] << " = ";
      printBits(std::cout, x[i]);
      std::cout << " = ";
      printBits(std::cout, x[i], middle, end);
      std::cout << " + ";
      printBits(std::cout, x[i], begin, middle);
      std::cout << "\n";
    }
  }

  return 0;
}
