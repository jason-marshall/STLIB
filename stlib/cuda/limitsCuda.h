// -*- C++ -*-

#if !defined(__cuda_limitsCuda_h__)
#define __cuda_limitsCuda_h__

namespace std
{

template<typename _T>
struct numeric_limits;

template<>
struct numeric_limits<float> {
  static const bool is_specialized = true;

  static
  __device__
  __host__
  float
  min()
  {
    return __FLT_MIN__;
  }

  static
  __device__
  __host__
  float
  max()
  {
    return __FLT_MAX__;
  }

  static const int digits = __FLT_MANT_DIG__;
  static const int digits10 = __FLT_DIG__;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const int radix = __FLT_RADIX__;

  static
  __device__
  __host__
  float
  epsilon()
  {
    return __FLT_EPSILON__;
  }

  static
  __device__
  __host__
  float
  round_error()
  {
    return 0.5F;
  }

  static const int min_exponent = __FLT_MIN_EXP__;
  static const int min_exponent10 = __FLT_MIN_10_EXP__;
  static const int max_exponent = __FLT_MAX_EXP__;
  static const int max_exponent10 = __FLT_MAX_10_EXP__;

  static const bool has_infinity = __FLT_HAS_INFINITY__;
  static const bool has_quiet_NaN = __FLT_HAS_QUIET_NAN__;
  static const bool has_signaling_NaN = has_quiet_NaN;

  static
  __device__
  __host__
  float
  infinity()
  {
    const unsigned x = 0x7f800000;
    return *reinterpret_cast<const float*>(&x);
  }

  static
  __device__
  __host__
  float
  quiet_NaN()
  {
    return nan(0);
  }

  static const bool is_bounded = true;
  static const bool is_modulo = false;
};

template<>
struct numeric_limits<double> {
  static const bool is_specialized = true;

  static
  __device__
  __host__
  double
  min()
  {
    return __DBL_MIN__;
  }

  static
  __device__
  __host__
  double
  max()
  {
    return __DBL_MAX__;
  }

  static const int digits = __DBL_MANT_DIG__;
  static const int digits10 = __DBL_DIG__;
  static const bool is_signed = true;
  static const bool is_integer = false;
  static const bool is_exact = false;
  static const int radix = __FLT_RADIX__;

  static
  __device__
  __host__
  double
  epsilon()
  {
    return __DBL_EPSILON__;
  }

  static
  __device__
  __host__
  double
  round_error()
  {
    return 0.5F;
  }

  static const int min_exponent = __DBL_MIN_EXP__;
  static const int min_exponent10 = __DBL_MIN_10_EXP__;
  static const int max_exponent = __DBL_MAX_EXP__;
  static const int max_exponent10 = __DBL_MAX_10_EXP__;

  static const bool has_infinity = __DBL_HAS_INFINITY__;
  static const bool has_quiet_NaN = __DBL_HAS_QUIET_NAN__;
  static const bool has_signaling_NaN = has_quiet_NaN;

  static
  __device__
  __host__
  double
  infinity()
  {
    const unsigned long long x = 0x7ff0000000000000;
    return *reinterpret_cast<const double*>(&x);
  }

  static
  __device__
  __host__
  double
  quiet_NaN()
  {
    return nan(0);
  }

  static const bool is_bounded = true;
  static const bool is_modulo = false;
};

} // namespace std

#endif
