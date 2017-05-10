// -*- C++ -*-

#if !defined(__numerical_interpolation_simplex_ipp__)
#error This file is an implementation detail of numerical/interpolation/simplex.
#endif

namespace stlib
{
namespace numerical
{

//
// 1-D space.
//

// 1-D simplex. 1-D space. Reduced problem.
namespace internal
{

template<typename T, typename F>
inline
void
lin_interp_reduced(const T b, const F& beta, const T x, F& f)
{
  assert(b != 0);
  // f = (x / b) * beta;
  f = beta;
  f *= (x / b);
}

}

// 1-D simplex. 1-D space.
template<typename T, typename F>
inline
F
linear_interpolation(const T a, T b,
                     const F& alpha, F beta,
                     T x)
{
  // Translate the nodes and x by -a.
  b -= a;
  x -= a;
  // Translate the values by -alpha.
  beta -= alpha;
  // Interpolate
  F field;
  internal::lin_interp_reduced(b, beta, x, field);
  field += alpha;
  return field;
}

// 1-D simplex. 1-D space.
// Implementation of general interface.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<std::array<T, 1>, 2>&
                     positions,
                     const std::array<F, 2>& values,
                     const std::array<T, 1>& location)
{
  return linear_interpolation(positions[0][0], positions[1][0],
                              values[0], values[1], location[0]);
}

// 1-D simplex. 2-D space.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<T, 2>& a,
                     const std::array<T, 2>& b,
                     const F& alpha, const F& beta,
                     const std::array<T, 2>& x)
{
  std::array<T, 2> c;
  c[0] = a[0] + a[1] - b[1];
  c[1] = a[1] + b[0] - a[0];
  return linear_interpolation(a, b, c, alpha, beta, alpha, x);
}

// 1-D simplex. 2-D space.
// Implementation of general interface.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<std::array<T, 2>, 2>& positions,
                     const std::array<F, 2>& values,
                     const std::array<T, 2>& location)
{
  return linear_interpolation(positions[0], positions[1],
                              values[0], values[1], location);
}


//
// 2-D.
//

// 2-D simplex. 2-D space. Reduced problem.
namespace internal
{

template<typename T, typename F>
inline
void
lin_interp_reduced(const std::array<T, 2>& b,
                   const std::array<T, 2>& c,
                   const F& beta, const F& gamma,
                   const std::array<T, 2>& x,
                   F& field)
{
  const T den = b[0] * c[1] - b[1] * c[0];
  assert(den != 0);

  /*
  field = ((c[1] * beta - b[1] * gamma) * x[0] +
        (- c[0] * beta + b[0] * gamma) * x[1]) / den;
  */
  field = beta;
  field *= c[1];
  F s = gamma;
  s *= b[1];
  field -= s;
  field *= x[0];

  s = beta;
  s *= - c[0];
  F t = gamma;
  t *= b[0];
  s += t;
  s *= x[1];

  field += s;
  field /= den;
}

}


// 2-D simplex. 2-D space.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<T, 2>& a,
                     std::array<T, 2> b,
                     std::array<T, 2> c,
                     const F& alpha,
                     F beta,
                     F gamma,
                     std::array<T, 2> x)
{
  // Translate the nodes and x by -a.
  b -= a;
  c -= a;
  x -= a;
  // Translate the values by -alpha.
  beta -= alpha;
  gamma -= alpha;
  // Interpolate.
  F field;
  internal::lin_interp_reduced(b, c, beta, gamma, x, field);
  field += alpha;
  return field;
}

// 2-D simplex. 2-D space.
// Implementation of general interface.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<std::array<T, 2>, 3>& positions,
                     const std::array<F, 3>& values,
                     const std::array<T, 2>& location)
{
  return linear_interpolation(positions[0], positions[1], positions[2],
                              values[0], values[1], values[2], location);
}

//
// 3-D.
//

// 3-D simplex. 3-D space. Reduced problem.
namespace internal
{

template<typename T, typename F>
inline
void
lin_interp_reduced(const std::array<T, 3>& b,
                   const std::array<T, 3>& c,
                   const std::array<T, 3>& d,
                   const F& beta, const F& gamma, const F& delta,
                   const std::array<T, 3>& x,
                   F& field)
{
  const T den = ((c[2] * d[1] - c[1] * d[2]) * b[0] +
                 (b[1] * d[2] - b[2] * d[1]) * c[0] +
                 (b[2] * c[1] - b[1] * c[2]) * d[0]);
  assert(den != 0);

  /*
  const F p0 = ((c[2]*d[1] - c[1]*d[2]) * beta +
  	   (b[1]*d[2] - b[2]*d[1]) * gamma +
  	   (b[2]*c[1] - b[1]*c[2]) * delta);
  const F p1 = ((c[0]*d[2] - c[2]*d[0]) * beta +
  	   (b[2]*d[0] - b[0]*d[2]) * gamma +
  	   (b[0]*c[2] - b[2]*c[0]) * delta);
  const F p2 = ((c[1]*d[0] - c[0]*d[1]) * beta +
  	   (b[0]*d[1] - b[1]*d[0]) * gamma +
  	   (b[1]*c[0] - b[0]*c[1]) * delta);
  field = (p0 * x[0] + p1 * x[1] + p2 * x[2]) / den;
  */

  F s = beta;
  s *= c[2] * d[1] - c[1] * d[2];
  F t = gamma;
  t *= b[1] * d[2] - b[2] * d[1];
  s += t;
  t = delta;
  t *= b[2] * c[1] - b[1] * c[2];
  s += t;
  s *= x[0];
  field = s;

  s = beta;
  s *= c[0] * d[2] - c[2] * d[0];
  t = gamma;
  t *= b[2] * d[0] - b[0] * d[2];
  s += t;
  t = delta;
  t *= b[0] * c[2] - b[2] * c[0];
  s += t;
  s *= x[1];
  field += s;

  s = beta;
  s *= c[1] * d[0] - c[0] * d[1];
  t = gamma;
  t *= b[0] * d[1] - b[1] * d[0];
  s += t;
  t = delta;
  t *= b[1] * c[0] - b[0] * c[1];
  s += t;
  s *= x[2];
  field += s;

  field /= den;
}

}

// 3-D simplex. 3-D space.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<T, 3>& a,
                     std::array<T, 3> b,
                     std::array<T, 3> c,
                     std::array<T, 3> d,
                     const F& alpha,
                     F beta,
                     F gamma,
                     F delta,
                     std::array<T, 3> x)
{
  // Translate the nodes and x by -a.
  b -= a;
  c -= a;
  d -= a;
  x -= a;
  // Translate the values by -alpha.
  beta -= alpha;
  gamma -= alpha;
  delta -= alpha;
  // Interpolate.
  F field;
  internal::lin_interp_reduced(b, c, d, beta, gamma, delta, x, field);
  field += alpha;
  return field;
}

// 3-D simplex. 3-D space.
// Implementation of general interface.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<std::array<T, 3>, 4>& positions,
                     const std::array<F, 4>& values,
                     const std::array<T, 3>& location)
{
  return linear_interpolation(positions[0], positions[1],
                              positions[2], positions[3],
                              values[0], values[1], values[2], values[3],
                              location);
}

// 2-D simplex. 3-D space.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<T, 3>& a,
                     std::array<T, 3> b,
                     std::array<T, 3> c,
                     const F& alpha,
                     F beta,
                     F gamma,
                     std::array<T, 3> x)
{
  const F zero = F();

  // Translate the nodes and x by -a.
  b -= a;
  c -= a;
  x -= a;
  // Translate the values by -alpha.
  beta -= alpha;
  gamma -= alpha;
  // Normal to the triangle.
  std::array<T, 3> n = {{
      -b[2]* c[1] + b[1]* c[2],
      b[2]* c[0] - b[0]* c[2],
      -b[1]* c[0] + b[0]* c[1]
    }
  };
  // Interpolate.
  F field;
  internal::lin_interp_reduced(b, c, n, beta, gamma, zero, x, field);
  field += alpha;
  return field;
}

// 2-D simplex. 3-D space.
// Implementation of general interface.
template<typename T, typename F>
inline
F
linear_interpolation(const std::array<std::array<T, 3>, 3>& positions,
                     const std::array<F, 3>& values,
                     const std::array<T, 3>& location)
{
  return linear_interpolation(positions[0], positions[1], positions[2],
                              values[0], values[1], values[2], location);
}

} // namespace numerical
}
