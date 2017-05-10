// -*- C++ -*-

#if !defined(__CoordinateDescentHookeJeeves_ipp__)
#error This file is an implementation detail of the class CoordinateDescentHookeJeeves.
#endif

namespace stlib
{
namespace numerical
{

//
// Constructors
//

template<class _Function>
inline
CoordinateDescentHookeJeeves<_Function>::
CoordinateDescentHookeJeeves(Function& function,
                             const Number initialStepSize,
                             const Number finalStepSize) :
  _function(function, std::numeric_limits<std::size_t>::max()),
  _initialStepSize(initialStepSize),
  _finalStepSize(finalStepSize),
  _stepSizeReductionFactor(0.5),
  _stepLimit(2 * std::size_t(std::ceil(1 / _stepSizeReductionFactor)) + 1)
{
}

template<class _Function>
inline
typename CoordinateDescentHookeJeeves<_Function>::Number
CoordinateDescentHookeJeeves<_Function>::
minimize(Vector* x)
{
  Number value;
  std::size_t numSteps;
  Vector delta(x->size(), Number(0));
  Number trialValue;

  numSteps = 0;
  _function.resetNumFunctionCalls();

  _stepSize = _initialStepSize;
  value = _function(*x);

  while (_stepSize >= _finalStepSize) {
    // Find a descent direction by searching in each coordinate direction.
    trialValue = value;
    if (descentDirection(x, &value, &delta)) {
      ++numSteps;
      // Since we made some improvement, pursue that direction.
      trialValue = value;
      std::size_t accelerationStepCount = 0;
      do {
        // If we have taken too many steps.
        if (accelerationStepCount != 0 &&
            accelerationStepCount % _stepLimit == 0) {
          // Increase the step size;
          delta /= _stepSizeReductionFactor;
        }
        ++accelerationStepCount;

        // Move further in the descent direction.
        value = trialValue;
        *x += delta;
        trialValue = _function(*x);
      }
      while (trialValue < value);
      // Undo the last bad step.
      *x -= delta;
    }
    else {
      // We failed to find a descent direction.  Reduce the step size.
      _stepSize *= _stepSizeReductionFactor;
    }
  }
  return value;
}


template<class _Function>
inline
bool
CoordinateDescentHookeJeeves<_Function>::
descentDirection(Vector* x, Number* value, Vector* delta)
{
  bool result = false;
  for (std::size_t i = 0; i != x->size(); ++i) {
    // Try searching in the positive direction.
    (*delta)[i] = 0;
    if (coordinateSearch(x, value, delta, i, 1)) {
      result = true;
    }
    else {
      // If that didn't work, search in the negative direction.
      (*delta)[i] = 0;
      if (coordinateSearch(x, value, delta, i, -1)) {
        result = true;
      }
    }
  }
  return result;
}


template<class _Function>
inline
bool
CoordinateDescentHookeJeeves<_Function>::
coordinateSearch(Vector* x, Number* value, Vector* delta,
                 const std::size_t i, const int sign)
{
#ifdef STLIB_DEBUG
  assert((*delta)[i] == 0);
  assert(sign == 1 || sign == -1);
#endif

  Number trialValue = *value;
  std::size_t stepCount = 0;
  do {
    // If we have taken too many steps.
    if (stepCount != 0 && stepCount % _stepLimit == 0) {
      // Increase the step size.
      _stepSize /= _stepSizeReductionFactor;
    }
    ++stepCount;

    *value = trialValue;
    (*x)[i] += sign * _stepSize;
    (*delta)[i] += sign * _stepSize;
    trialValue = _function(*x);
  }
  while (trialValue < *value);
  // Undo the last bad step.
  --stepCount;
  (*x)[i] -= sign * _stepSize;
  (*delta)[i] -= sign * _stepSize;

  return stepCount != 0;
}


} // namespace numerical
}
