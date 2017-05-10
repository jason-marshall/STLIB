// -*- C++ -*-

//
// Specializations for 0-D.
//

#if !defined(__FixedArray0_ipp__)
#error This file is an implementation detail of the class FixedArray.
#endif

namespace stlib
{
namespace ads
{

//
// Math operators.
//

template<typename T>
inline
T
computeSum(const FixedArray<0, T>& /*x*/)
{
  return 0;
}

template<typename T>
inline
T
computeProduct(const FixedArray<0, T>& /*x*/)
{
  return 1;
}

// min() and max() are intentionally not implemented.

//
// Equality
//

template<typename T1, typename T2>
inline
bool
operator==(const FixedArray<0, T1>& /*a*/, const FixedArray<0, T2>& /*b*/)
{
  return true;
}

//
// Comparison
//

template<typename T1, typename T2>
inline
bool
operator<(const FixedArray<0, T1>& /*a*/, const FixedArray<0, T2>& /*b*/)
{
  return false;
}

} // namespace ads
} // namespace stlib
