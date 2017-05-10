// -*- C++ -*-

/*
 *      C library of Limited memory BFGS (L-BFGS).
 *
 * Copyright (c) 1990, Jorge Nocedal
 * Copyright (c) 2007,2008,2009 Naoaki Okazaki
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#if !defined(__QuasiNewtonLBFGS_ipp__)
#error This file is an implementation detail of the class QuasiNewtonLBFGS.
#endif

namespace stlib
{
namespace numerical
{

template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
minimize(Vector* x)
{
  if (_areThrowingExceptions) {
    if (_areThrowingMaxComputationExceptions) {
      return _minimize(x);
    }
    else {
      try {
        return _minimize(x);
      }
      catch (OptMaxComputationError&) {
        _function.resetNumFunctionCalls();
        Vector g(x->size());
        return _function(*x, &g);
      }
    }
  }
  else {
    try {
      return _minimize(x);
    }
    catch (OptError&) {
      _function.resetNumFunctionCalls();
      Vector g(x->size());
      return _function(*x, &g);
    }
  }
}

template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
_minimize(Vector* x)
{
  // Start the timer.
  performance::Timer timer;

  const std::size_t n = x->size();
  Number ys, yy;
  Number beta;
  Number fx = 0.;
  Number rate = 0.;

  /* Check the input parameters for errors. */
  assert(n != 0);

  _function.resetNumFunctionCalls();

  /* Allocate working space. */
  Vector xp(n, 0.), g(n, 0.), gp(n, 0.), d(n, 0.);

  /* Allocate limited memory storage. */
  std::vector<IterationData> lm(_m, IterationData(n));

  /* Allocate an array for storing previous values of the objective function. */
  Vector pf(_past, 0.);

  /* Evaluate the function value and its gradient. */
  fx = _function(*x, &g);

  /* Store the initial value of the objective function. */
  if (! pf.empty()) {
    pf[0] = fx;
  }

  /*
    Compute the direction;
    we assume the initial hessian matrix H_0 is the identity matrix.
  */
  for (std::size_t i = 0; i != d.size(); ++i) {
    d[i] = -g[i];
  }

  // Make sure that the initial variables are not a minimizer.
  if (hasGradientConverged(*x, g)) {
    return fx;
  }

  // Compute the initial step.
  Number step = 1. / std::sqrt(ext::dot(d, d));

  std::size_t j;
  std::size_t k = 1;
  std::size_t end = 0;
  for (;;) {
    /* Store the current position and gradient vectors. */
    std::copy(x->begin(), x->end(), xp.begin());
    std::copy(g.begin(), g.end(), gp.begin());

    /* Search for an optimal step. */
    try {
      lineSearch(x, &fx, &g, d, &step, xp);
    }
    catch (OptError&) {
      /* Revert to the previous point. */
      std::copy(xp.begin(), xp.end(), x->begin());
      std::copy(gp.begin(), gp.end(), g.begin());
      throw;
    }

    // Check for gradient convergence.
    if (hasGradientConverged(*x, g)) {
      break;
    }

    /*
      Test for stopping criterion.
      The criterion is given by the following formula:
      (f(past_x) - f(x)) / f(x) < \delta
    */
    if (! pf.empty()) {
      /* We don't test the stopping criterion while k < past. */
      if (_past <= k) {
        /* Compute the relative improvement from the past. */
        rate = (pf[k % _past] - fx) / fx;

        /* The stopping criterion. */
        if (rate < _delta) {
          break;
        }
      }

      /* Store the current value of the objective function. */
      pf[k % _past] = fx;
    }

    if (_maxIterations != 0 && _maxIterations < k + 1) {
      throw OptMaxIterationsError("In QuasiNewtonLBFGS::_minimize(): Maximum number of iterations reached.");
    }

    // Check if we have exceeded the maximum allowed time.
    if (timer.elapsed() > _maxTime) {
      throw OptMaxTimeError("In QuasiNewtonLBFGS::_minimize(): The maximum allowed time has been exceeded.");
    }

    {
      /*
        Update vectors s and y:
        s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        y_{k+1} = g_{k+1} - g_{k}.
      */
      IterationData& it = lm[end];
      for (std::size_t i = 0; i != it.s.size(); ++i) {
        it.s[i] = (*x)[i] - xp[i];
        it.y[i] = g[i] - gp[i];
      }

      /*
        Compute scalars ys and yy:
        ys = y^t \cdot s = 1 / \rho.
        yy = y^t \cdot y.
        Notice that yy is used for scaling the hessian matrix H_0
        (Cholesky factor).
      */
      ys = ext::dot(it.y, it.s);
      yy = ext::dot(it.y, it.y);
      it.ys = ys;
    }

    /*
      Recursive formula to compute dir = -(H \cdot g).
      This is described in page 779 of:
      Jorge Nocedal.
      Updating Quasi-Newton Matrices with Limited Storage.
      Mathematics of Computation, Vol. 35, No. 151,
      pp. 773--782, 1980.
    */
    const std::size_t bound = (_m <= k) ? _m : k;
    ++k;
    end = (end + 1) % _m;

    /* Compute the steepest direction. */
    /* Compute the negative of gradients. */
    for (std::size_t i = 0; i != d.size(); ++i) {
      d[i] = -g[i];
    }

    j = end;
    for (std::size_t i = 0; i != bound; ++i) {
      j = (j + _m - 1) % _m;    /* if (--j == -1) j = _m-1; */
      IterationData& it = lm[j];
      /* \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}. */
      it.alpha = ext::dot(it.s, d);
      it.alpha /= it.ys;
      /* q_{i} = q_{i+1} - \alpha_{i} y_{i}. */
      for (std::size_t i = 0; i != d.size(); ++i) {
        d[i] -= it.alpha * it.y[i];
      }
    }

    d *= ys / yy;

    for (std::size_t i = 0; i != bound; ++i) {
      IterationData& it = lm[j];
      /* \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}. */
      beta = ext::dot(it.y, d);
      beta /= it.ys;
      /* \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}. */
      for (std::size_t i = 0; i != d.size(); ++i) {
        d[i] += (it.alpha - beta) * it.s[i];
      }
      j = (j + 1) % _m;        /* if (++j == _m) j = 0; */
    }

    /*
      Now the search direction d is ready. We try step = 1 first.
    */
    step = 1.0;
  }

  return fx;
}


template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
minimizeIllConditioned(Vector* x, const std::size_t groupSize)
{
  if (_areThrowingExceptions) {
    if (_areThrowingMaxComputationExceptions) {
      return _minimizeIllConditioned(x, groupSize);
    }
    else {
      try {
        return _minimizeIllConditioned(x, groupSize);
      }
      catch (OptMaxComputationError&) {
        _function.resetNumFunctionCalls();
        Vector g(x->size());
        return _function(*x, &g);
      }
    }
  }
  else {
    try {
      return _minimizeIllConditioned(x, groupSize);
    }
    catch (OptError&) {
      _function.resetNumFunctionCalls();
      Vector g(x->size());
      return _function(*x, &g);
    }
  }
}

template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
_minimizeIllConditioned(Vector* x, const std::size_t groupSize)
{
  typedef FunctionOfSelectedCoordinates<ObjectiveFunction<Function> > Fosc;
  assert(groupSize > 0 && x->size() % groupSize == 0);
  std::size_t oldGroup = x->size() / groupSize;
  for (std::size_t i = 0; i != _maxResetIllConditioned; ++i) {
    try {
      // Return the minima if we can calculate it.
      return _minimize(x);
    }
    catch (OptError&) {
      // Determine the group with the maximum gradient.
      std::size_t group = findGroupWithMaximumGradient(*x, groupSize);
      // If it is the same group as last time then the last attempt to
      // improve the condition was not sufficient.
      if (group == oldGroup) {
        throw OptError("In QuasiNewtonLBFGS::minimizeIllConditioned(): Unable to substantially improve an ill-conditioned problem.");
      }
      oldGroup = group;
      // Select the coordinates in the group.
      std::vector<std::size_t> indices(groupSize);
      Vector y(groupSize);
      for (std::size_t j = 0; j != groupSize; ++j) {
        indices[j] = group * groupSize + j;
        y[j] = (*x)[indices[j]];
      }
      // Perform a minimization on those coordinates.
      Fosc f(_function, *x, indices);
      ConjugateGradient<Fosc> opt(f);
      try {
        // Minimize on the selected coordinates.
        opt.minimize(&y);
        // Update the selected coordinates.
        for (std::size_t j = 0; j != groupSize; ++j) {
          (*x)[indices[j]] = y[j];
        }
      }
      catch (OptError& error) {
        throw OptError(
          std::string("In QuasiNewtonLBFGS::minimizeIllConditioned(): Failed minimization on selected coordinates of an ill-conditioned problem. ")
          + error.what());
      }
    }
  }
  throw OptMaxIterationsError("In QuasiNewtonLBFGS::minimizeIllConditioned(): Maximum number of resets for an ill-conditioned problem has been exceeded.");
  return 0;
}


template<class _Function>
inline
bool
QuasiNewtonLBFGS<_Function>::
hasGradientConverged(const Vector& x, const Vector& g) const
{
  /* Compute x and g norms. */
  const Number xnorm = std::max(std::sqrt(ext::dot(x, x)), 1.);
  const Number gnorm = std::sqrt(ext::dot(g, g));

  /*
    Relative gradient convergence test.
    The criterion is given by the following formula:
    |g(x)| / \max(1, |x|) < \epsilon
  */
  if (gnorm <= _relativeGradientTolerance * xnorm) {
    /* Convergence. */
    return true;
  }
  // RMS gradient convergence test.
  // ||g|| / sqrt(n) < epsilon
  // Formulate to avoid the square root and division.
  if (gnorm * gnorm <
      _rmsGradientTolerance * _rmsGradientTolerance * g.size()) {
    return true;
  }
  // If we are checking the maximum component of the gradient.
  if (_maxGradientTolerance != 0) {
    Number maximum = 0;
    Number magnitude;
    for (std::size_t i = 0; i != g.size(); ++i) {
      magnitude = std::abs(g[i]);
      if (magnitude > maximum) {
        maximum = magnitude;
      }
    }
    if (maximum < _maxGradientTolerance) {
      return true;
    }
  }
  return false;
}


template<class _Function>
inline
std::size_t
QuasiNewtonLBFGS<_Function>::
findGroupWithMaximumGradient(const Vector& x, const std::size_t groupSize)
{
  Vector gradient(x.size());
  _function(x, &gradient);
  const std::size_t numGroups = x.size() / groupSize;
  std::size_t maxGroup = 0;
  Number maxSquaredGradient = 0;
  for (std::size_t g = 0; g != numGroups; ++g) {
    Number s = 0;
    for (std::size_t i = 0; i != groupSize; ++i) {
      s += gradient[g * groupSize + i] * gradient[g * groupSize + i];
    }
    if (s > maxSquaredGradient) {
      maxGroup = g;
      maxSquaredGradient = s;
    }
  }
  return maxGroup;
}


template<class _Function>
inline
void
QuasiNewtonLBFGS<_Function>::
lineSearch(Vector* x,
           Number* f,
           Vector* g,
           const Vector& s, // The search direction.
           Number* stp,
           const Vector& xp)
{
  std::size_t count = 0;
  Number dg;
  Number stx, fx, dgx;
  Number sty, fy, dgy;
  Number fxm, dgxm, fym, dgym, fm, dgm;
  Number finit, ftest1, dgtest;
  Number width, previousWidth;
  Number stmin, stmax;

  /* Check the input parameters for errors. */
  if (*stp <= 0.) {
    throw OptStepError("In QuasiNewtonLBFGS::lineSearch(): Non-positive step size.");
  }

  /* Compute the initial gradient in the search direction. */
  Number dginit = ext::dot(*g, s);

  /* Make sure that s points to a descent direction. */
  if (0 < dginit) {
    throw OptStepError("In QuasiNewtonLBFGS::lineSearch(): Bad descent direction.");
  }

  /* Initialize local variables. */
  bool isBracketed = false;
  bool stage1 = true;
  finit = *f;
  dgtest = _ftol * dginit;
  width = _maxStep - _minStep;
  previousWidth = 2.0 * width;

  /*
    The variables stx, fx, dgx contain the values of the step,
    function, and directional derivative at the best step.
    The variables sty, fy, dgy contain the value of the step,
    function, and derivative at the other endpoint of
    the interval of uncertainty.
    The variables stp, f, dg contain the values of the step,
    function, and derivative at the current step.
  */
  stx = sty = 0.;
  fx = fy = finit;
  dgx = dgy = dginit;

  for (;;) {
    /*
      Set the minimum and maximum steps to correspond to the
      present interval of uncertainty.
    */
    if (isBracketed) {
      stmin = std::min(stx, sty);
      stmax = std::max(stx, sty);
    }
    else {
      stmin = stx;
      stmax = *stp + 4.0 * (*stp - stx);
    }

    /* Clip the step in the range of [stpmin, stpmax]. */
    if (*stp < _minStep) {
      *stp = _minStep;
    }
    if (_maxStep < *stp) {
      *stp = _maxStep;
    }

    /*
      If an unusual termination is to occur then let
      stp be the lowest point obtained so far.
    */
    if ((isBracketed && ((*stp <= stmin || stmax <= *stp)
                         || _maxLinesearch <= count + 1)) || (isBracketed
                             && (stmax - stmin <= _xtol * stmax))) {
      *stp = stx;
    }

    /*
      Compute the current value of x:
      x <- x + (*stp) * s.
    */
    for (std::size_t i = 0; i != x->size(); ++i) {
      (*x)[i] = xp[i] + *stp * s[i];
    }

    /* Evaluate the function and gradient values. */
    *f = _function(*x, g);
    dg = ext::dot(*g, s);

    ftest1 = finit + *stp * dgtest;
    ++count;

    /* Test for errors and convergence. */
    if (isBracketed && (*stp <= stmin || stmax <= *stp)) {
      throw OptError("In QuasiNewtonLBFGS::lineSearch(): Rounding errors prevent further progress.");
    }
    if (*stp == _maxStep && *f <= ftest1 && dg <= dgtest) {
      throw OptStepError("In QuasiNewtonLBFGS::lineSearch(): The step is the maximum value.");
    }
    if (*stp == _minStep && (ftest1 < *f || dgtest <= dg)) {
      std::ostringstream message;
      message << "In QuasiNewtonLBFGS::lineSearch():\n"
              << " The step is the minimum value.\n"
              << " *stp = " << *stp << '\n'
              << " _minStep = " << _minStep << '\n';
      throw OptStepError(message.str());
    }
    if (isBracketed && (stmax - stmin) <= _xtol * stmax) {
      throw OptError("In QuasiNewtonLBFGS::lineSearch(): Relative width of the interval of uncertainty is too small.");
    }
    if (_maxLinesearch <= count) {
      throw OptMaxIterationsError("In QuasiNewtonLBFGS::lineSearch(): Maximum number of iterations.");
    }
    if (*f <= ftest1 && std::abs(dg) <= _gtol * (-dginit)) {
      /* The sufficient decrease condition and the directional derivative condition hold. */
      break;
    }

    /*
      In the first stage we seek a step for which the modified
      function has a nonpositive value and nonnegative derivative.
    */
    if (stage1 && *f <= ftest1 && std::min(_ftol, _gtol) * dginit <= dg) {
      stage1 = false;
    }

    /*
      A modified function is used to predict the step only if
      we have not obtained a step for which the modified
      function has a nonpositive function value and nonnegative
      derivative, and if a lower function value has been
      obtained but the decrease is not sufficient.
    */
    if (stage1 && ftest1 < *f && *f <= fx) {
      /* Define the modified function and derivative values. */
      fm = *f - *stp * dgtest;
      fxm = fx - stx * dgtest;
      fym = fy - sty * dgtest;
      dgm = dg - dgtest;
      dgxm = dgx - dgtest;
      dgym = dgy - dgtest;

      /*
        Call updateTrialInterval() to update the interval of
        uncertainty and to compute the new step.
      */
      updateTrialInterval(&stx, &fxm, &dgxm,
                          &sty, &fym, &dgym,
                          stp, &fm, &dgm,
                          stmin, stmax, &isBracketed);

      /* Reset the function and gradient values for f. */
      fx = fxm + stx * dgtest;
      fy = fym + sty * dgtest;
      dgx = dgxm + dgtest;
      dgy = dgym + dgtest;
    }
    else {
      /*
        Call update_trial_interval() to update the interval of
        uncertainty and to compute the new step.
      */
      updateTrialInterval(&stx, &fx, &dgx,
                          &sty, &fy, &dgy,
                          stp, f, &dg,
                          stmin, stmax, &isBracketed);
    }

    /*
      Force a sufficient decrease in the interval of uncertainty.
    */
    if (isBracketed) {
      if (0.66 * previousWidth <= std::abs(sty - stx)) {
        *stp = stx + 0.5 * (sty - stx);
      }
      previousWidth = width;
      width = std::abs(sty - stx);
    }
  }
}


template<class _Function>
inline
void
QuasiNewtonLBFGS<_Function>::
updateTrialInterval(Number* x, Number* fx, Number* dx,
                    Number* y, Number* fy, Number* dy,
                    Number* t, Number* ft, Number* dt,
                    const Number tmin, const Number tmax, bool* isBracketed)
{
  const bool dsign = *dt * (*dx / std::abs(*dx)) < 0.;

  /* Check the input parameters for errors. */
  if (*isBracketed) {
    if (*t <= std::min(*x, *y) || std::max(*x, *y) <= *t) {
      throw OptError("In QuasiNewtonLBFGS::updateTrialInterval: The trival value t is out of the interval.");
    }
    if (0. <= *dx * (*t - *x)) {
      throw OptError("In QuasiNewtonLBFGS::updateTrialInterval: The function must decrease from x.");
    }
    if (tmax < tmin) {
      throw OptError("In QuasiNewtonLBFGS::updateTrialInterval: Incorrect tmin and tmax specified.");
    }
  }

  bool bound;
  Number mc; /* minimizer of an interpolated cubic. */
  Number mq; /* minimizer of an interpolated quadratic. */
  Number newt;   /* new trial value. */
  /*
    Trial value selection.
  */
  if (*fx < *ft) {
    /*
      Case 1: a higher function value.
      The minimum is bracketed. If the cubic minimizer is closer
      to x than the quadratic one, the cubic one is taken, else
      the average of the minimizers is taken.
    */
    *isBracketed = true;
    bound = true;
    mc = cubicMinimizer(*x, *fx, *dx, *t, *ft, *dt);
    mq = quadraticMinimizer(*x, *fx, *dx, *t, *ft);
    // Note: Difference from libSBFGS. Correct for the case that mc is NaN.
    if (mc != mc) {
      mc = mq;
    }
    if (std::abs(mc - *x) < std::abs(mq - *x)) {
      newt = mc;
    }
    else {
      newt = mc + 0.5 * (mq - mc);
    }
  }
  else if (dsign) {
    /*
      Case 2: a lower function value and derivatives of
      opposite sign. The minimum is bracketed. If the cubic
      minimizer is closer to x than the quadratic (secant) one,
      the cubic one is taken, else the quadratic one is taken.
    */
    *isBracketed = true;
    bound = false;
    mc = cubicMinimizer(*x, *fx, *dx, *t, *ft, *dt);
    mq = quadraticMinimizer(*x, *dx, *t, *dt);
    if (mc != mc) {
      mc = mq;
    }
    if (std::abs(mc - *t) > std::abs(mq - *t)) {
      newt = mc;
    }
    else {
      newt = mq;
    }
  }
  else if (std::abs(*dt) < std::abs(*dx)) {
    /*
      Case 3: a lower function value, derivatives of the
      same sign, and the magnitude of the derivative decreases.
      The cubic minimizer is only used if the cubic tends to
      infinity in the direction of the minimizer or if the minimum
      of the cubic is beyond t. Otherwise the cubic minimizer is
      defined to be either tmin or tmax. The quadratic (secant)
      minimizer is also computed and if the minimum is bracketed
      then the the minimizer closest to x is taken, else the one
      farthest away is taken.
    */
    bound = true;
    mc = cubicMinimizer(*x, *fx, *dx, *t, *ft, *dt, tmin, tmax);
    mq = quadraticMinimizer(*x, *dx, *t, *dt);
    if (mc != mc) {
      mc = mq;
    }
    if (*isBracketed) {
      if (std::abs(*t - mc) < std::abs(*t - mq)) {
        newt = mc;
      }
      else {
        newt = mq;
      }
    }
    else {
      if (std::abs(*t - mc) > std::abs(*t - mq)) {
        newt = mc;
      }
      else {
        newt = mq;
      }
    }
  }
  else {
    /*
      Case 4: a lower function value, derivatives of the
      same sign, and the magnitude of the derivative does
      not decrease. If the minimum is not bracketed, the step
      is either tmin or tmax, else the cubic minimizer is taken.
    */
    bound = false;
    if (*isBracketed) {
      newt = cubicMinimizer(*t, *ft, *dt, *y, *fy, *dy);
      if (newt != newt) {
        newt = tmax;
      }
    }
    else if (*x < *t) {
      newt = tmax;
    }
    else {
      newt = tmin;
    }
  }

  /*
    Update the interval of uncertainty. This update does not
    depend on the new step or the case analysis above.

    - Case a: if f(x) < f(t),
    x <- x, y <- t.
    - Case b: if f(t) <= f(x) && f'(t)*f'(x) > 0,
    x <- t, y <- y.
    - Case c: if f(t) <= f(x) && f'(t)*f'(x) < 0,
    x <- t, y <- x.
  */
  if (*fx < *ft) {
    /* Case a */
    *y = *t;
    *fy = *ft;
    *dy = *dt;
  }
  else {
    /* Case c */
    if (dsign) {
      *y = *x;
      *fy = *fx;
      *dy = *dx;
    }
    /* Cases b and c */
    *x = *t;
    *fx = *ft;
    *dx = *dt;
  }

  /* Clip the new trial value in [tmin, tmax]. */
  if (tmax < newt) {
    newt = tmax;
  }
  if (newt < tmin) {
    newt = tmin;
  }

  /*
    Redefine the new trial value if it is close to the upper bound
    of the interval.
  */
  if (*isBracketed && bound) {
    mq = *x + 0.66 * (*y - *x);
    if (*x < *y) {
      if (mq < newt) {
        newt = mq;
      }
    }
    else {
      if (newt < mq) {
        newt = mq;
      }
    }
  }

  // Record the new trial value.
  *t = newt;
}


template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
quadraticMinimizer(const Number u, const Number fu, const Number du,
                   const Number v, const Number fv) const
{
  const Number a = v - u;
  return u + du / ((fu - fv) / a + du) / 2 * a;
}


/**
 * Find a minimizer of an interpolated quadratic function.
 *  @param  u       The value of one point, u.
 *  @param  du      The value of f'(u).
 *  @param  v       The value of another point, v.
 *  @param  dv      The value of f'(v).

 *  @return The minimizer of the interpolated quadratic.
 */
template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
quadraticMinimizer(const Number u, const Number du, const Number v,
                   const Number dv) const
{
  const Number a = u - v;
  return v + dv / (dv - du) * a;
}


/**
 * Find a minimizer of an interpolated cubic function.
 *  @param u The value of one point, u.
 *  @param fu The value of f(u).
 *  @param du The value of f'(u).
 *  @param v The value of another point, v.
 *  @param fv The value of f(v).
 *  @param du The value of f'(v).

 *  @return The minimizer of the interpolated cubic.
 */
template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
cubicMinimizer(const Number u, const Number fu, const Number du,
               const Number v, const Number fv, const Number dv) const
{
  const Number d = v - u;
  const Number theta = (fu - fv) * 3 / d + du + dv;
  Number p = std::abs(theta);
  Number q = std::abs(du);
  Number r = std::abs(dv);
  const Number s = std::max(std::max(p, q), r);
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
  const Number a = theta / s;
  Number gamma = s * std::sqrt(a * a - (du / s) * (dv / s));
  if (v < u) {
    gamma = -gamma;
  }
  p = gamma - du + theta;
  q = gamma - du + gamma + dv;
  r = p / q;
  return u + r * d;
}

/**
 * Find a minimizer of an interpolated cubic function.
 *  @param u The value of one point, u.
 *  @param fu The value of f(u).
 *  @param du The value of f'(u).
 *  @param v The value of another point, v.
 *  @param fv The value of f(v).
 *  @param du The value of f'(v).
 *  @param xmin The minimum value.
 *  @param xmax The maximum value.

 *  @return The minimizer of the interpolated cubic.
 */
template<class _Function>
inline
typename QuasiNewtonLBFGS<_Function>::Number
QuasiNewtonLBFGS<_Function>::
cubicMinimizer(const Number u, const Number fu, const Number du,
               const Number v, const Number fv, const Number dv,
               const Number xmin, const Number xmax) const
{
  const Number d = v - u;
  const Number theta = (fu - fv) * 3 / d + du + dv;
  Number p = std::abs(theta);
  Number q = std::abs(du);
  Number r = std::abs(dv);
  const Number s = std::max(std::max(p, q), r);
  /* gamma = s*sqrt((theta/s)**2 - (du/s) * (dv/s)) */
  const Number a = theta / s;
  Number gamma = s * std::sqrt(std::max(0., a * a - (du / s) * (dv / s)));
  if (u < v) {
    gamma = -gamma;
  }
  p = gamma - dv + theta;
  q = gamma - dv + gamma + du;
  r = p / q;

  Number result;
  if (r < 0. && gamma != 0.) {
    result = v - r * d;
  }
  else if (a < 0) {
    result = xmax;
  }
  else {
    result = xmin;
  }
  return result;
}

} // namespace numerical
}
