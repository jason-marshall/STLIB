// -*- C++ -*-

#include <functional>
#include <iostream>

#include <cmath>

struct Void {
  typedef void result_type;

  result_type
  operator()() const
  {
    std::cout << "void()\n\n";
  }
};

struct Int {
  typedef int result_type;

  result_type
  operator()() const
  {
    std::cout << "int()\n\n";
    return 1;
  }
};

struct B {
  typedef int argument_type;
  typedef int result_type;

  result_type
  operator()(argument_type x) const
  {
    std::cout << "Argument = " << x << ".\n\n";
    return x;
  }
};

struct C {
  typedef float first_argument_type;
  typedef double second_argument_type;
  typedef double result_type;

  result_type
  operator()(first_argument_type x, second_argument_type y) const
  {
    std::cout << "First argument = " << x
              << ", second argument = " << y << ".\n\n";
    return x + y;
  }
};


struct AssignInt {
  int
  operator()() const
  {
    return 7;
  }
};

struct AssignConstIntRef {
  int _data;

  AssignConstIntRef() :
    _data(7)
  {
  }

  const int&
  operator()() const
  {
    return _data;
  }
};

struct AssignVoid {
  void
  operator()(int* x) const
  {
    *x = 7;
  }
};


class Caller
{
public:

  template<typename _F>
  typename _F::result_type
  call(_F f)
  {
    return call<_F>(&_F::operator(), f);
  }

private:

  template<typename _F>
  typename _F::result_type
  call(typename _F::result_type(_F::* /*dummy*/)() const, _F f)
  {
    std::cout << "result_type ()\n";
    return f();
  }

#if 0
  template<typename _F>
  typename _F::result_type
  call(void (_F::* /*dummy*/)() const, _F f)
  {
    std::cout << "void ()\n";
    return f();
  }
#endif

  template<typename _F>
  typename _F::result_type
  call(typename _F::result_type(_F::* /*dummy*/)(typename _F::argument_type)
       const, _F f)
  {
    return f(typename _F::argument_type());
  }

  template<typename _F>
  typename _F::result_type
  call(typename _F::result_type(_F::* /*dummy*/)(typename _F::first_argument_type,
       typename _F::second_argument_type) const, _F f)
  {
    return f(typename _F::first_argument_type(),
             typename _F::second_argument_type());
  }


public:

  template<typename _F>
  void
  assign(_F f)
  {
    assign(&_F::operator(), f);
  }

private:

  template<typename _F>
  void
  assign(int (_F::* /*dummy*/)() const, _F f)
  {
    int x = f();
    std::cout << "assign int f(), x = " << x << "\n";
  }

  template<typename _F>
  void
  assign(const int& (_F::* /*dummy*/)() const, _F f)
  {
    int x = f();
    std::cout << "assign const int& f(), x = " << x << "\n";
  }

  template<typename _F>
  void
  assign(void (_F::* /*dummy*/)(int*) const, _F f)
  {
    int x;
    f(&x);
    std::cout << "assign void f(int*), x = " << x << "\n";
  }

};

int
f(int x)
{
  std::cout << "f int\n";
  return 2 * x;
}

template<typename _T>
inline
_T
g(_T x)
{
  std::cout << "g\n";
  return 2 * x;
}

int
main()
{
  Caller caller;
  {
    Void x;
    caller.call(x);
  }
  {
    Int x;
    caller.call(x);
  }
  {
    B b;
    caller.call(b);
  }
  {
    C c;
    caller.call(c);
  }

  caller.call(std::ptr_fun(f));
  caller.call(std::ptr_fun(g<double>));
  // exp is overloaded.
  double(*fp)(double) = std::exp;
  caller.call(std::ptr_fun(fp));

  {
    AssignInt x;
    caller.assign(x);
  }
  {
    AssignConstIntRef x;
    caller.assign(x);
  }
  {
    AssignVoid x;
    caller.assign(x);
  }

  return 0;
}
