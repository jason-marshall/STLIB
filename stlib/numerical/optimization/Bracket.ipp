// -*- C++ -*-

#if !defined(__numerical_optimization_Bracket_ipp__)
#error This file is an implementation detail of the class Bracket.
#endif

namespace stlib
{
namespace numerical
{


template<class _Function>
inline
void
Bracket<_Function>::
bracket(const double a, const double b)
{
  const double Gold = 1.618034, GLimit = 100.0, Tiny = 1.0e-20;
  _x[0] = a;
  _x[1] = b;
  double fu;
  _f[0] = _function(_x[0]);
  _f[1] = _function(_x[1]);
  if (_f[1] > _f[0]) {
    std::swap(_x[0], _x[1]);
    std::swap(_f[1], _f[0]);
  }
  _x[2] = _x[1] + Gold * (_x[1] - _x[0]);
  _f[2] = _function(_x[2]);
  while (_f[1] > _f[2]) {
    double r = (_x[1] - _x[0]) * (_f[1] - _f[2]);
    double q = (_x[1] - _x[2]) * (_f[1] - _f[0]);
    double den = 2.0 * std::max(std::abs(q - r), Tiny);
    if (q - r < 0) {
      den = - den;
    }
    double u = _x[1] - ((_x[1] - _x[2]) * q - (_x[1] - _x[0]) * r) / den;
    double ulim = _x[1] + GLimit * (_x[2] - _x[1]);
    if ((_x[1] - u) * (u - _x[2]) > 0) {
      fu = _function(u);
      if (fu < _f[2]) {
        _x[0] = _x[1];
        _x[1] = u;
        _f[0] = _f[1];
        _f[1] = fu;
        order();
        return;
      }
      else if (fu > _f[1]) {
        _x[2] = u;
        _f[2] = fu;
        order();
        return;
      }
      u = _x[2] + Gold * (_x[2] - _x[1]);
      fu = _function(u);
    }
    else if ((_x[2] - u) * (u - ulim) > 0.0) {
      fu = _function(u);
      if (fu < _f[2]) {
        shift(&_x[1], &_x[2], &u, u + Gold * (u - _x[2]));
        shift(&_f[1], &_f[2], &fu, _function(u));
      }
    }
    else if ((u - ulim) * (ulim - _x[2]) >= 0) {
      u = ulim;
      fu = _function(u);
    }
    else {
      u = _x[2] + Gold * (_x[2] - _x[1]);
      fu = _function(u);
    }
    shift(&_x[0], &_x[1], &_x[2], u);
    shift(&_f[0], &_f[1], &_f[2], fu);
  }
  order();
}


} // namespace numerical
}
