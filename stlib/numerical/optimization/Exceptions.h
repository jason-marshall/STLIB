// -*- C++ -*-

/*!
  \file numerical/optimization/Exceptions.h
  \brief Exceptions for optimizations.
*/

#if !defined(__numerical_optimization_Exceptions_h__)
#define __numerical_optimization_Exceptions_h__

#include <stdexcept>

namespace stlib
{
namespace numerical
{

//! An exception that occurs during optimization.
/*! All optimization exception inherit from this class. */
class OptError : public std::runtime_error
{
public:
  //! Construct from the error message.
  explicit
  OptError(const std::string& errorMessage) :
    std::runtime_error(errorMessage)
  {
  }
};

//! Exception that is thrown when the maximum amount of computation exceeded.
class OptMaxComputationError : public OptError
{
public:
  //! Construct from the error message.
  explicit
  OptMaxComputationError(const std::string& errorMessage) :
    OptError(errorMessage)
  {
  }
};

//! Exception that is thrown when the maximum number of iterations is exceeded.
class OptMaxIterationsError : public OptMaxComputationError
{
public:
  //! Construct from the error message.
  explicit
  OptMaxIterationsError(const std::string& errorMessage) :
    OptMaxComputationError(errorMessage)
  {
  }
};

//! Exception that is thrown when the maximum number function evaluations is exceeded.
class OptMaxObjFuncCallsError : public OptMaxComputationError
{
public:
  //! Construct from the error message.
  explicit
  OptMaxObjFuncCallsError(const std::string& errorMessage) :
    OptMaxComputationError(errorMessage)
  {
  }
};

//! Exception that is thrown when the maximum time is exceeded.
class OptMaxTimeError : public OptMaxComputationError
{
public:
  //! Construct from the error message.
  explicit
  OptMaxTimeError(const std::string& errorMessage) :
    OptMaxComputationError(errorMessage)
  {
  }
};

//! Exception that is thrown when there is an error in taking a step.
class OptStepError : public OptError
{
public:
  //! Construct from the error message.
  explicit
  OptStepError(const std::string& errorMessage) :
    OptError(errorMessage)
  {
  }
};

} // namespace numerical
}

#endif
