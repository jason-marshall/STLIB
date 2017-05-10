// -*- C++ -*-

#if !defined(__numerical_optimization_DBrent_ipp__)
#error This file is an implementation detail of the class DBrent.
#endif

namespace stlib
{
namespace numerical
{


template<class _Function>
inline
double
DBrent<_Function>::
minimize(const double initial1, const double initial2, double* minPoint)
{
  const std::size_t ITMAX = 100;
  const double ZEPS = std::numeric_limits<double>::epsilon() * 1.0e-3;
  bool ok1, ok2;
  double a, b, d = 0, d1, d2, du, dv, dw, dx, e = 0;
  double fu, fv, fw, fx, olde, tol1, tol2, u, u1, u2, v, w, x, xm;

  Base::bracket(initial1, initial2);
  a = Base::points()[0];
  b = Base::points()[2];
  x = w = v = Base::points()[1];
  fw = fv = fx = Base::values()[1];
  dw = dv = dx = _function.derivative(x);
  for (std::size_t iter = 0; iter != ITMAX; ++iter) {
    xm = 0.5 * (a + b);
    tol1 = _tolerance * std::abs(x) + ZEPS;
    tol2 = 2.0 * tol1;
    if (std::abs(x - xm) <= (tol2 - 0.5 * (b - a))) {
      *minPoint = x;
      return fx;
    }
    if (std::abs(e) > tol1) {
      d1 = 2.0 * (b - a);
      d2 = d1;
      if (dw != dx) {
        d1 = (w - x) * dx / (dx - dw);
      }
      if (dv != dx) {
        d2 = (v - x) * dx / (dx - dv);
      }
      u1 = x + d1;
      u2 = x + d2;
      ok1 = (a - u1) * (u1 - b) > 0.0 && dx * d1 <= 0.0;
      ok2 = (a - u2) * (u2 - b) > 0.0 && dx * d2 <= 0.0;
      olde = e;
      e = d;
      if (ok1 || ok2) {
        if (ok1 && ok2) {
          d = (std::abs(d1) < std::abs(d2) ? d1 : d2);
        }
        else if (ok1) {
          d = d1;
        }
        else {
          d = d2;
        }
        if (std::abs(d) <= std::abs(0.5 * olde)) {
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
        else {
          e = dx >= 0.0 ? a - x : b - x;
          d = 0.5 * e;
        }
      }
      else {
        e = dx >= 0.0 ? a - x : b - x;
        d = 0.5 * e;
      }
    }
    else {
      e = dx >= 0.0 ? a - x : b - x;
      d = 0.5 * e;
    }
    if (std::abs(d) >= tol1) {
      u = x + d;
      fu = _function(u);
    }
    else {
      u = x + (d >= 0 ? tol1 : -tol1);
      fu = _function(u);
      if (fu > fx) {
        *minPoint = x;
        return fx;
      }
    }
    du = _function.derivative(u);
    if (fu <= fx) {
      if (u >= x) {
        a = x;
      }
      else {
        b = x;
      }
      move(&v, &fv, &dv, w, fw, dw);
      move(&w, &fw, &dw, x, fx, dx);
      move(&x, &fx, &dx, u, fu, du);
    }
    else {
      if (u < x) {
        a = u;
      }
      else {
        b = u;
      }
      if (fu <= fw || w == x) {
        move(&v, &fv, &dv, w, fw, dw);
        move(&w, &fw, &dw, u, fu, du);
      }
      else if (fu < fv || v == x || v == w) {
        move(&v, &fv, &dv, u, fu, du);
      }
    }
  }
  throw std::runtime_error("Too many iterations in DBrent::minimize().");
}


} // namespace numerical
}
