// -*- C++ -*-

#if !defined(__numerical_constants_Logarithm_ipp__)
#error This file is an implementation detail of Logarithm.
#endif

namespace stlib
{
namespace numerical
{


// We must specialize for each integer type because a specialization may not
// depend on a template parameter.

// The logarithm base 0 is indeterminate, so we do not provide a result.
// Trying to compute a logarithm base 0 will cause a compilation error.
template<char Argument>
class Logarithm<char, 0, Argument>
{
};
template<signed char Argument>
class Logarithm<signed char, 0, Argument>
{
};
template<unsigned char Argument>
class Logarithm<unsigned char, 0, Argument>
{
};
template<short Argument>
class Logarithm<short, 0, Argument>
{
};
template<unsigned short Argument>
class Logarithm<unsigned short, 0, Argument>
{
};
template<int Argument>
class Logarithm<int, 0, Argument>
{
};
template<unsigned int Argument>
class Logarithm<unsigned int, 0, Argument>
{
};
template<long Argument>
class Logarithm<long, 0, Argument>
{
};
template<unsigned long Argument>
class Logarithm<unsigned long, 0, Argument>
{
};

// The logarithm base 1 is indeterminate, so we do not provide a result.
// Trying to compute a logarithm base 1 will cause a compilation error.
template<char Argument>
class Logarithm<char, 1, Argument>
{
};
template<signed char Argument>
class Logarithm<signed char, 1, Argument>
{
};
template<unsigned char Argument>
class Logarithm<unsigned char, 1, Argument>
{
};
template<short Argument>
class Logarithm<short, 1, Argument>
{
};
template<unsigned short Argument>
class Logarithm<unsigned short, 1, Argument>
{
};
template<int Argument>
class Logarithm<int, 1, Argument>
{
};
template<unsigned int Argument>
class Logarithm<unsigned int, 1, Argument>
{
};
template<long Argument>
class Logarithm<long, 1, Argument>
{
};
template<unsigned long Argument>
class Logarithm<unsigned long, 1, Argument>
{
};

// The logarithm of 0 is indeterminate, so we do not provide a result.
//  Trying to compute the logarithm of 0 will cause a compilation error.
template<char Base>
class Logarithm<char, Base, 0>
{
};
template<signed char Base>
class Logarithm<signed char, Base, 0>
{
};
template<unsigned char Base>
class Logarithm<unsigned char, Base, 0>
{
};
template<short Base>
class Logarithm<short, Base, 0>
{
};
template<unsigned short Base>
class Logarithm<unsigned short, Base, 0>
{
};
template<int Base>
class Logarithm<int, Base, 0>
{
};
template<unsigned int Base>
class Logarithm<unsigned int, Base, 0>
{
};
template<long Base>
class Logarithm<long, Base, 0>
{
};
template<unsigned long Base>
class Logarithm<unsigned long, Base, 0>
{
};

// log_B(1) = 0 for nonzero B.
template<char Base>
class Logarithm<char, Base, 1>
{
public:
  enum {Result = 0};
};
template<signed char Base>
class Logarithm<signed char, Base, 1>
{
public:
  enum {Result = 0};
};
template<unsigned char Base>
class Logarithm<unsigned char, Base, 1>
{
public:
  enum {Result = 0};
};
template<short Base>
class Logarithm<short, Base, 1>
{
public:
  enum {Result = 0};
};
template<unsigned short Base>
class Logarithm<unsigned short, Base, 1>
{
public:
  enum {Result = 0};
};
template<int Base>
class Logarithm<int, Base, 1>
{
public:
  enum {Result = 0};
};
template<unsigned int Base>
class Logarithm<unsigned int, Base, 1>
{
public:
  enum {Result = 0};
};
template<long Base>
class Logarithm<long, Base, 1>
{
public:
  enum {Result = 0};
};
template<unsigned long Base>
class Logarithm<unsigned long, Base, 1>
{
public:
  enum {Result = 0};
};


} // namespace numerical
}
