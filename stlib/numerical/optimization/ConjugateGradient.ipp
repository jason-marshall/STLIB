// -*- C++ -*-

#if !defined(__ConjugateGradient_ipp__)
#error This file is an implementation detail of the class ConjugateGradient.
#endif

namespace stlib
{
namespace numerical
{


template<class _Function, template<class> class _Minimizer1D>
inline
typename ConjugateGradient<_Function, _Minimizer1D>::Number
ConjugateGradient<_Function, _Minimizer1D>::
minimize(Vector* x)
{
  // A small value to rectify the special case that the minimum function value
  // is zero.
  const Number Eps = 1.0e-18;
  const Number GradientTolerance = 1.0e-8;
  const std::size_t size = x->size();
  Vector g(size), h(size), direction(size);
  _function.resetNumFunctionCalls();
  Number previous = _function(*x, &direction);
  for (std::size_t i = 0; i != size; ++i) {
    direction[i] = -direction[i];
    g[i] = direction[i];
    h[i] = direction[i];
  }
  Number gg, dgg, gamma, current, maxScaledComponent;
  // Loop until convergence or until the maximum number of function evaluations
  // is exceeded.
  while (true) {
    current = lineMinimization(x, &direction);
    if (2.0 * std::abs(current - previous) <= _fractionalTolerance
        * (std::abs(current) + std::abs(previous) + Eps)) {
      return current;
    }
    previous = current;

    // Test for convergence to zero gradient.
    _function(*x, &direction);
    maxScaledComponent = 0;
    for (std::size_t i = 0; i != size; ++i) {
      maxScaledComponent = std::max(maxScaledComponent,
                                    std::abs(direction[i])
                                    * std::max(std::abs((*x)[i]), 1.));
    }
    maxScaledComponent /= std::max(std::abs(previous), 1.);
    if (maxScaledComponent < GradientTolerance) {
      return current;
    }

    gg = 0.;
    dgg = 0.;
    for (std::size_t i = 0; i != size; ++i) {
      gg += g[i] * g[i];
      dgg += (direction[i] + g[i]) * direction[i];
    }
    // If the gradient is zero, we are done.
    if (gg == 0.0) {
      return current;
    }
    gamma = dgg / gg;
    for (std::size_t i = 0; i != size; ++i) {
      g[i] = -direction[i];
      h[i] = g[i] + gamma * h[i];
      direction[i] = h[i];
    }
  }
}


template<class _Function, template<class> class _Minimizer1D>
inline
typename ConjugateGradient<_Function, _Minimizer1D>::Number
ConjugateGradient<_Function, _Minimizer1D>::
lineMinimization(Vector* x, Vector* direction)
{
  typedef FunctionOnLine<ObjectiveFunction<Function> > F1D;
  // Define the 1-D objective function with the multi-dimensional function
  // and the line.
  F1D f(_function, *x, *direction);
  _Minimizer1D<F1D> minimizer(f);
  // The parametrized line is (*x) + t * (*direction).
  Number t;
  const Number value = minimizer.minimize(0., 1., &t);
  // The step along the line.
  *direction *= t;
  // Move the current minimum point.
  *x += *direction;
  // Return the function value.
  return value;
}

} // namespace numerical
}
