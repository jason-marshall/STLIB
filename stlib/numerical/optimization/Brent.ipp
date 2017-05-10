// -*- C++ -*-

#if !defined(__numerical_optimization_Brent_ipp__)
#error This file is an implementation detail of the class Brent.
#endif

namespace stlib
{
namespace numerical
{


template<class _Function>
inline
double
Brent<_Function>::
minimize(const double initial1, const double initial2, double* minPoint)
{
  const std::size_t ITMAX = 100;
  const double CGOLD = 0.3819660;
  const double ZEPS = std::numeric_limits<double>::epsilon() * 1.0e-3;
  double a, b, d = 0., etemp, fu, fv, fw, fx;
  double p, q, r, tol1, tol2, u, v, w, x, xm;
  double e = 0.;

  Base::bracket(initial1, initial2);
  a = Base::points()[0];
  b = Base::points()[2];
  x = w = v = Base::points()[1];
  fw = fv = fx = Base::values()[1];
  for (std::size_t iter = 0; iter != ITMAX; ++iter) {
    xm = 0.5 * (a + b);
    tol2 = 2.0 * (tol1 = _tolerance * std::abs(x) + ZEPS);
    if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
      *minPoint = x;
      return fx;
    }
    if (std::abs(e) > tol1) {
      r = (x - w) * (fx - fv);
      q = (x - v) * (fx - fw);
      p = (x - v) * q - (x - w) * r;
      q = 2.0 * (q - r);
      if (q > 0.0) {
        p = -p;
      }
      q = std::abs(q);
      etemp = e;
      e = d;
      if (std::abs(p) >= std::abs(0.5 * q * etemp) || p <= q * (a - x)
          || p >= q * (b - x)) {
        d = CGOLD * (e = (x >= xm ? a - x : b - x));
      }
      else {
        d = p / q;
        u = x + d;
        if (u - a < tol2 || b - u < tol2) {
          if (xm - x >= 0) {
            d = tol1;
          }
          else {
            d = -tol1;
          }
        }
      }
    }
    else {
      d = CGOLD * (e = (x >= xm ? a - x : b - x));
    }

    u = (std::abs(d) >= tol1 ? x + d : x + (d >= 0 ? tol1 : - tol1));
    fu = _function(u);
    if (fu <= fx) {
      if (u >= x) {
        a = x;
      }
      else {
        b = x;
      }
      shift(&v, &w, &x, u);
      shift(&fv, &fw, &fx, fu);
    }
    else {
      if (u < x) {
        a = u;
      }
      else {
        b = u;
      }
      if (fu <= fw || w == x) {
        v = w;
        w = u;
        fv = fw;
        fw = fu;
      }
      else if (fu <= fv || v == x || v == w) {
        v = u;
        fv = fu;
      }
    }
  }
  throw std::runtime_error("Too many iterations in Brent::minimize().");
}


} // namespace numerical
}
