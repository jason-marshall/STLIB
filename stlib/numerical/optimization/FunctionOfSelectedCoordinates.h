// -*- C++ -*-

/*!
  \file numerical/optimization/FunctionOfSelectedCoordinates.h
  \brief Restrict a function to the selected coordinates.
*/

#if !defined(__numerical_optimization_FunctionOfSelectedCoordinates_h__)
#define __numerical_optimization_FunctionOfSelectedCoordinates_h__

#include <vector>

#include <cassert>

namespace stlib
{
namespace numerical
{

//! Restrict a function to the selected coordinates.
/*!
  This class stores a reference to the supplied function Thus the user is
  responsible for making sure that this function is valid as long as this
  class is used.
*/
template<class _Function>
class FunctionOfSelectedCoordinates
{
  //
  // Types.
  //
public:
  //! The function to evaluate.
  typedef _Function Function;
  //! A vector in N-D space.
  typedef typename Function::argument_type Vector;
  //! The argument type is vector of the selected coordinates.
  typedef typename Function::argument_type argument_type;
  //! The result type.
  typedef typename Function::result_type result_type;

  //
  // Member data.
  //
private:

  //! Reference to the function defined on a vector space.
  Function& _function;
  //! The complete argument vector.
  Vector _completeX;
  //! The complete gradient vector.
  Vector _completeGradient;
  //! The selected coordinates.
  std::vector<std::size_t> _coordinateIndices;

  //
  // Not implemented.
  //
private:

  // Default constructor not implemented.
  FunctionOfSelectedCoordinates();

  // Copy constructor not implemented.
  FunctionOfSelectedCoordinates(const FunctionOfSelectedCoordinates&);

  // Assignment operator not implemented.
  FunctionOfSelectedCoordinates&
  operator=(const FunctionOfSelectedCoordinates&);

  //--------------------------------------------------------------------------
  /*! \name Constructors etc.
    The default constructor, the copy constructor and the assignment
    operator are not implemented. We use the default destructor.
  */
  // @{
public:

  //! Construct from the function, the full argument vector, and the selected coordinates.
  FunctionOfSelectedCoordinates(Function& function,
                                const Vector& x,
                                const std::vector<std::size_t>&
                                coordinateIndices) :
    _function(function),
    _completeX(x),
    _completeGradient(x.size()),
    _coordinateIndices(coordinateIndices)
  {
    assert(! _completeX.empty());
    assert(! _coordinateIndices.empty());
    assert(_coordinateIndices.size() <= _completeX.size());
    for (std::size_t i = 0; i != _coordinateIndices.size(); ++i) {
      assert(_coordinateIndices[i] < _completeX.size());
    }
  }

  // @}
  //--------------------------------------------------------------------------
  //! \name Function evaluation.
  // @{
public:

  //! Evaluate the gradient and return the objective function value.
  result_type
  operator()(const argument_type& x, argument_type* gradient)
  {
    // Copy the selected coordinates into the argument.
    for (std::size_t i = 0; i != _coordinateIndices.size(); ++i) {
      _completeX[_coordinateIndices[i]] = x[i];
    }
    // Evaluate the function.
    const result_type value = _function(_completeX, &_completeGradient);
    // Copy the selected gradient values.
    for (std::size_t i = 0; i != _coordinateIndices.size(); ++i) {
      (*gradient)[i] = _completeGradient[_coordinateIndices[i]];
    }
    // Return the function value.
    return value;
  }

  //! Return the objective function value.
  result_type
  operator()(const argument_type& x)
  {
    // Copy the selected coordinates into the argument.
    for (std::size_t i = 0; i != _coordinateIndices.size(); ++i) {
      _completeX[_coordinateIndices[i]] = x[i];
    }
    // Evaluate the function.
    return _function(_completeX);
  }

  // @}
};

} // namespace numerical
}

#endif
