// -*- C++ -*-

#include <iostream>
#include <limits>
#include <utility>

template<class _T1, class _T2>
struct Pair {
  _T1 first;
  _T2 second;
};

int
main()
{

  std::cout << "Fundamental types:\n"
            << "sizeof(char) = " << sizeof(char) << "\n"
            << "sizeof(int) = " << sizeof(int) << "\n"
            << "sizeof(long) = " << sizeof(long) << "\n"
            //<< "sizeof(long int) = " << sizeof(long long) << "\n"
            << "sizeof(float) = " << sizeof(float) << "\n"
            << "sizeof(double) = " << sizeof(double) << "\n"
            << "sizeof(long double) = " << sizeof(long double) << "\n\n"
            << "std::numeric_limits<long>::digits = "
            << std::numeric_limits<long>::digits << "\n"
            << "std::numeric_limits<unsigned long>::digits = "
            << std::numeric_limits<unsigned long>::digits << "\n"
            << "\nstd::pair:\n"
            << "sizeof(std::pair<float, unsigned>) = "
            << sizeof(std::pair<float, unsigned>) << '\n'
            << "sizeof(std::pair<float, long unsigned>) = "
            << sizeof(std::pair<float, long unsigned>) << '\n'
            << "sizeof(std::pair<double, unsigned>) = "
            << sizeof(std::pair<double, unsigned>) << '\n'
            << "sizeof(std::pair<double, long unsigned>) = "
            << sizeof(std::pair<double, long unsigned>) << '\n'
            << "\nPair:\n"
            << "sizeof(Pair<char, char>) = "
            << sizeof(Pair<char, char>) << '\n'
            << "sizeof(Pair<float, unsigned>) = "
            << sizeof(Pair<float, unsigned>) << '\n'
            << "sizeof(Pair<float, long unsigned>) = "
            << sizeof(Pair<float, long unsigned>) << '\n'
            << "sizeof(Pair<double, unsigned>) = "
            << sizeof(Pair<double, unsigned>) << '\n'
            << "sizeof(Pair<double, long unsigned>) = "
            << sizeof(Pair<double, long unsigned>) << '\n';

  return 0;
}
