// -*- C++ -*-

// CONTINUE: Improve the type checking tests.

#include "stlib/ads/array/ArrayTypes.h"

#include <iostream>
#include <typeinfo>

#include <cassert>

using namespace stlib;

int
main()
{
  using namespace ads;

  std::cout << "\nArray of double\n";
  {
    typedef ArrayTypes<double> AT;
    {
      typedef AT::value_type type;
      std::cout << "value_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::parameter_type type;
      std::cout << "parameter_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::unqualified_value_type type;
      std::cout << "unqualified_value_type = " << typeid(type).name()
                << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::pointer type;
      std::cout << "pointer = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double*));
    }
    {
      typedef AT::const_pointer type;
      std::cout << "const_pointer = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::iterator type;
      std::cout << "iterator = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double*));
    }
    {
      typedef AT::const_iterator type;
      std::cout << "const_iterator = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::reference type;
      std::cout << "reference = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::const_reference type;
      std::cout << "const_reference = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double));
    }
    {
      typedef AT::size_type type;
      std::cout << "size_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(int));
    }
    {
      typedef AT::difference_type type;
      std::cout << "difference_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(std::ptrdiff_t));
    }
  }


  std::cout << "\nArray of const double\n";
  {
    typedef ArrayTypes<const double> AT;
    {
      typedef AT::value_type type;
      std::cout << "value_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::parameter_type type;
      std::cout << "parameter_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::unqualified_value_type type;
      std::cout << "unqualified_value_type = " << typeid(type).name()
                << '\n';
      assert(typeid(type) == typeid(double));
    }
    {
      typedef AT::pointer type;
      std::cout << "pointer = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::const_pointer type;
      std::cout << "const_pointer = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::iterator type;
      std::cout << "iterator = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::const_iterator type;
      std::cout << "const_iterator = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double*));
    }
    {
      typedef AT::reference type;
      std::cout << "reference = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double));
    }
    {
      typedef AT::const_reference type;
      std::cout << "const_reference = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(const double));
    }
    {
      typedef AT::size_type type;
      std::cout << "size_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(int));
    }
    {
      typedef AT::difference_type type;
      std::cout << "difference_type = " << typeid(type).name() << '\n';
      assert(typeid(type) == typeid(std::ptrdiff_t));
    }
  }

  return 0;
}
