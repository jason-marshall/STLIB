// -*- C++ -*-

#if !defined(__ext_functional_h__)
#define __ext_functional_h__

#include <functional>

namespace stlib
{
namespace ext
{


template<class _T>
inline
_T
identity_element(std::plus<_T>)
{
  return _T(0);
}

template<class _T>
inline
_T
identity_element(std::multiplies<_T>)
{
  return _T(1);
}

template<class _Operation1, class _Operation2>
class unary_compose
  : public std::unary_function<typename _Operation2::argument_type,
    typename _Operation1::result_type>
{
protected:
  _Operation1 _f1;
  _Operation2 _f2;

public:
  unary_compose(const _Operation1& x, const _Operation2& y)
    : _f1(x), _f2(y)
  {
  }

  typename _Operation1::result_type
  operator()(const typename _Operation2::argument_type& x) const
  {
    return _f1(_f2(x));
  }
};

template<class _Operation1, class _Operation2>
inline
unary_compose<_Operation1, _Operation2>
compose1(const _Operation1& f1, const _Operation2& f2)
{
  return unary_compose<_Operation1, _Operation2>(f1, f2);
}

template<class _Operation1, class _Operation2, class _Operation3>
class binary_compose
  : public std::unary_function<typename _Operation2::argument_type,
    typename _Operation1::result_type>
{
protected:
  _Operation1 _f1;
  _Operation2 _f2;
  _Operation3 _f3;

public:
  binary_compose(const _Operation1& x, const _Operation2& y,
                 const _Operation3& z)
    : _f1(x), _f2(y), _f3(z)
  {
  }

  typename _Operation1::result_type
  operator()(const typename _Operation2::argument_type& x) const
  {
    return _f1(_f2(x), _f3(x));
  }
};

template<class _Operation1, class _Operation2, class _Operation3>
inline binary_compose<_Operation1, _Operation2, _Operation3>
compose2(const _Operation1& f1, const _Operation2& f2,
         const _Operation3& f3)
{
  return binary_compose<_Operation1, _Operation2, _Operation3>
         (f1, f2, f3);
}

template<class _Arg1, class _Arg2>
struct _Project1st : public std::binary_function<_Arg1, _Arg2, _Arg1> {
  _Arg1
  operator()(const _Arg1& x, const _Arg2&) const
  {
    return x;
  }
};

template<class _Arg1, class _Arg2>
struct _Project2nd : public std::binary_function<_Arg1, _Arg2, _Arg2> {
  _Arg2
  operator()(const _Arg1&, const _Arg2& y) const
  {
    return y;
  }
};

template<class _Arg1, class _Arg2>
struct project1st : public _Project1st<_Arg1, _Arg2> {
};

template<class _Arg1, class _Arg2>
struct project2nd : public _Project2nd<_Arg1, _Arg2> {
};

template<class _Result>
struct _Constant_void_fun {
  typedef _Result result_type;
  result_type _val;

  _Constant_void_fun(const result_type& v) : _val(v)
  {
  }

  const result_type&
  operator()() const
  {
    return _val;
  }
};

template<class _Result, class _Argument>
struct _Constant_unary_fun {
  typedef _Argument argument_type;
  typedef  _Result  result_type;
  result_type _val;

  _Constant_unary_fun(const result_type& v) : _val(v)
  {
  }

  const result_type&
  operator()(const _Argument&) const
  {
    return _val;
  }
};

template<class _Result, class _Arg1, class _Arg2>
struct _Constant_binary_fun {
  typedef  _Arg1   first_argument_type;
  typedef  _Arg2   second_argument_type;
  typedef  _Result result_type;
  _Result _val;

  _Constant_binary_fun(const _Result& v) : _val(v)
  {
  }

  const result_type&
  operator()(const _Arg1&, const _Arg2&) const
  {
    return _val;
  }
};

template<class _Result>
struct constant_void_fun
    : public _Constant_void_fun<_Result> {

  constant_void_fun(const _Result& v)
    : _Constant_void_fun<_Result>(v)
  {
  }
};

template<class _Result, class _Argument = _Result>
struct constant_unary_fun : public _Constant_unary_fun<_Result, _Argument> {
  constant_unary_fun(const _Result& v)
    : _Constant_unary_fun<_Result, _Argument>(v)
  {
  }
};

template<class _Result, class _Arg1 = _Result, class _Arg2 = _Arg1>
struct constant_binary_fun
    : public _Constant_binary_fun<_Result, _Arg1, _Arg2> {
  constant_binary_fun(const _Result& v)
    : _Constant_binary_fun<_Result, _Arg1, _Arg2>(v)
  {
  }
};

template<class _Result>
inline
constant_void_fun<_Result>
constant0(const _Result& val)
{
  return constant_void_fun<_Result>(val);
}

template<class _Result>
inline
constant_unary_fun<_Result, _Result>
constant1(const _Result& val)
{
  return constant_unary_fun<_Result, _Result>(val);
}

template<class _Result>
inline
constant_binary_fun<_Result, _Result, _Result>
constant2(const _Result& val)
{
  return constant_binary_fun<_Result, _Result, _Result>(val);
}

} // namespace ext
}

#endif

