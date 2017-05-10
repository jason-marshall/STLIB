// -*- C++ -*-

#include <limits>
#include <iostream>

int
main()
{

  std::cout << "sizeof(bool) = " << sizeof(bool) << "\n"
            << "std::numeric_limits<bool>::min() = "
            << std::numeric_limits<bool>::min() << "\n"
            << "std::numeric_limits<bool>::max() = "
            << std::numeric_limits<bool>::max() << "\n\n"

            << "sizeof(char) = " << sizeof(char) << "\n"
            << "std::numeric_limits<char>::min() = "
            << int(std::numeric_limits<char>::min()) << "\n"
            << "std::numeric_limits<char>::max() = "
            << int(std::numeric_limits<char>::max()) << "\n\n"

            << "sizeof(signed char) = " << sizeof(signed char) << "\n"
            << "std::numeric_limits<signed char>::min() = "
            << int(std::numeric_limits<signed char>::min()) << "\n"
            << "std::numeric_limits<signed char>::max() = "
            << int(std::numeric_limits<signed char>::max()) << "\n\n"

            << "sizeof(unsigned char) = " << sizeof(unsigned char) << "\n"
            << "std::numeric_limits<unsigned char>::min() = "
            << int(std::numeric_limits<unsigned char>::min()) << "\n"
            << "std::numeric_limits<unsigned char>::max() = "
            << int(std::numeric_limits<unsigned char>::max()) << "\n\n"

            << "sizeof(short) = " << sizeof(short) << "\n"
            << "std::numeric_limits<short>::min() = "
            << std::numeric_limits<short>::min() << "\n"
            << "std::numeric_limits<short>::max() = "
            << std::numeric_limits<short>::max() << "\n\n"

            << "sizeof(unsigned short) = " << sizeof(unsigned short) << "\n"
            << "std::numeric_limits<unsigned short>::min() = "
            << std::numeric_limits<unsigned short>::min() << "\n"
            << "std::numeric_limits<unsigned short>::max() = "
            << std::numeric_limits<unsigned short>::max() << "\n\n"

            << "sizeof(int) = " << sizeof(int) << "\n"
            << "std::numeric_limits<int>::min() = "
            << std::numeric_limits<int>::min() << "\n"
            << "std::numeric_limits<int>::max() = "
            << std::numeric_limits<int>::max() << "\n\n"

            << "sizeof(unsigned int) = " << sizeof(unsigned int) << "\n"
            << "std::numeric_limits<unsigned int>::min() = "
            << std::numeric_limits<unsigned int>::min() << "\n"
            << "std::numeric_limits<unsigned int>::max() = "
            << std::numeric_limits<unsigned int>::max() << "\n\n"

            << "sizeof(long) = " << sizeof(long) << "\n"
            << "std::numeric_limits<long>::min() = "
            << std::numeric_limits<long>::min() << "\n"
            << "std::numeric_limits<long>::max() = "
            << std::numeric_limits<long>::max() << "\n\n"

            << "sizeof(unsigned long) = " << sizeof(unsigned long) << "\n"
            << "std::numeric_limits<unsigned long>::min() = "
            << std::numeric_limits<unsigned long>::min() << "\n"
            << "std::numeric_limits<unsigned long>::max() = "
            << std::numeric_limits<unsigned long>::max() << "\n\n"

            << "sizeof(std::size_t) = " << sizeof(std::size_t) << "\n"
            << "std::numeric_limits<std::size_t>::min() = "
            << std::numeric_limits<std::size_t>::min() << "\n"
            << "std::numeric_limits<std::size_t>::max() = "
            << std::numeric_limits<std::size_t>::max() << "\n\n"

            << "sizeof(std::ptrdiff_t) = " << sizeof(std::ptrdiff_t) << "\n"
            << "std::numeric_limits<std::ptrdiff_t>::min() = "
            << std::numeric_limits<std::ptrdiff_t>::min() << "\n"
            << "std::numeric_limits<std::ptrdiff_t>::max() = "
            << std::numeric_limits<std::ptrdiff_t>::max() << "\n\n"

            << "sizeof(float) = " << sizeof(float) << "\n"
            << "std::numeric_limits<float>::min() = "
            << std::numeric_limits<float>::min() << "\n"
            << "std::numeric_limits<float>::max() = "
            << std::numeric_limits<float>::max() << "\n"
            << "std::numeric_limits<float>::epsilon() = "
            << std::numeric_limits<float>::epsilon() << "\n"
            << "std::numeric_limits<float>::digits = "
            << std::numeric_limits<float>::digits << "\n\n"

            << "sizeof(double) = " << sizeof(double) << "\n"
            << "std::numeric_limits<double>::min() = "
            << std::numeric_limits<double>::min() << "\n"
            << "std::numeric_limits<double>::max() = "
            << std::numeric_limits<double>::max() << "\n"
            << "std::numeric_limits<double>::epsilon() = "
            << std::numeric_limits<double>::epsilon() << "\n"
            << "std::numeric_limits<double>::digits = "
            << std::numeric_limits<double>::digits << "\n\n"

            << "sizeof(long double) = " << sizeof(long double) << "\n"
            << "std::numeric_limits<long double>::min() = "
            << std::numeric_limits<long double>::min() << "\n"
            << "std::numeric_limits<long double>::max() = "
            << std::numeric_limits<long double>::max() << "\n"
            << "std::numeric_limits<long double>::epsilon() = "
            << std::numeric_limits<long double>::epsilon() << "\n"
            << "std::numeric_limits<long double>::digits = "
            << std::numeric_limits<long double>::digits << "\n\n";

  return 0;
}
