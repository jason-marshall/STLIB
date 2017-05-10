// -*- C++ -*-

#ifndef __performance_numerical_optimization_leonardJones_main_ipp__
#error This file is an implementation detail.
#endif

#include "stlib/ads/timer/Timer.h"
#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ext/array.h"
#include "stlib/numerical/random/uniform/ContinuousUniformGenerator.h"

#include <iostream>

using namespace stlib;

//! Lennard-Jones potential.
/*!
  See http://en.wikipedia.org/wiki/Lennard-Jones_potential

  We set epsilon = 1/4 and sigma = 1. V(r) = r<sup>-12</sup> - r<sup>-6</sup>.
  The potential attains its minimum value of -1/4 at r = 2<sup>1/6</sup>.
*/
class LennardJones :
   public std::unary_function<std::vector<double>, double> {
public:

   result_type
   operator()(const argument_type& x) const {
      assert(x.size() % 3 == 0);
      const std::size_t numAtoms = x.size() / 3;
      result_type f = 0;
      for (std::size_t i = 0; i != numAtoms; ++i) {
         for (std::size_t j = i + 1; j != numAtoms; ++j) {
            f += value(i, j, x);
         }
      }
      return f;
   }

   result_type
   operator()(const argument_type& x, argument_type* gradient) const {
      assert(x.size() == gradient->size());
      assert(x.size() % 3 == 0);
      const std::size_t numAtoms = x.size() / 3;
      result_type f = 0;
      std::fill(gradient->begin(), gradient->end(), 0.);
      for (std::size_t i = 0; i != numAtoms; ++i) {
         for (std::size_t j = i + 1; j != numAtoms; ++j) {
            f += evaluate(i, j, x, gradient);
         }
      }
      return f;
   }

protected:

   result_type
   evaluate(const std::size_t i, const std::size_t j, const argument_type& x,
            argument_type* gradient) const {
      // r = sqrt((a0-b0)^2 + (a1-b1)^2 + (a2-b2)^2)
      // dr/dx0 = (a0 - b0) / r
      // r = sqrt(x*x + y*y + z*z)
      // dr/dx = (1/2) / r * 2 * x = x / r
      // r^(-12) - r^(-6)
      // df/dx = -12 * r^(-13) * x / r - (-6) * r^(-7) * x / r
      // df/dx = -12 * x * r^(-14) + 6 * x * r^(-8)
      // df/dx = 6 * x * (r^(-8) - 2 * r^(-14))
      const std::array<result_type, 3> p =
      {{x[3*i] - x[3*j], x[3*i+1] - x[3*j+1],  x[3*i+2] - x[3*j+2]}};
      const result_type r2 = stlib::ext::dot(p, p);
      const result_type i2 = 1. / r2;
      const result_type i4 = i2 * i2;
      const result_type i6 = i4 * i2;
      const result_type i8 = i4 * i4;
      const result_type t = 6. * (i8 - 2. * i8 * i4 * i2);
      for (std::size_t n = 0; n != 3; ++n) {
         (*gradient)[3*i+n] += p[n] * t;
         (*gradient)[3*j+n] -= p[n] * t;
      }
      return i6 * i6 - i6;
   }

   result_type
   value(const std::size_t i, const std::size_t j, const argument_type& x)
   const {
      const std::array<result_type, 3> p =
      {{x[3*i] - x[3*j], x[3*i+1] - x[3*j+1],  x[3*i+2] - x[3*j+2]}};
      const result_type i2 = 1. / stlib::ext::dot(p, p);
      const result_type i6 = i2 * i2 * i2;
      return i6 * i6 - i6;
   }
};

//! Lennard-Jones potential that is linearized for r < 1/2.
class LennardJonesLinearized :
   public LennardJones {
private:

   typedef LennardJones Base;

public:

   result_type
   operator()(const argument_type& x) const {
      assert(x.size() % 3 == 0);
      const std::size_t numAtoms = x.size() / 3;
      result_type f = 0;
      for (std::size_t i = 0; i != numAtoms; ++i) {
         for (std::size_t j = i + 1; j != numAtoms; ++j) {
            f += value(i, j, x);
         }
      }
      return f;
   }

   result_type
   operator()(const argument_type& x, argument_type* gradient) const {
      assert(x.size() == gradient->size());
      assert(x.size() % 3 == 0);
      const std::size_t numAtoms = x.size() / 3;
      result_type f = 0;
      std::fill(gradient->begin(), gradient->end(), 0.);
      for (std::size_t i = 0; i != numAtoms; ++i) {
         for (std::size_t j = i + 1; j != numAtoms; ++j) {
            f += evaluate(i, j, x, gradient);
         }
      }
      return f;
   }

private:

   result_type
   evaluate(const std::size_t i, const std::size_t j, const argument_type& x,
   argument_type* gradient) const {
      const std::array<result_type, 3> p =
      {{x[3*i] - x[3*j], x[3*i+1] - x[3*j+1],  x[3*i+2] - x[3*j+2]}};
      const result_type r2 = stlib::ext::dot(p, p);
      if (r2 < 0.25) {
         result_type g;
         for (std::size_t n = 0; n != 3; ++n) {
            if (p[n] >= 0) {
               g = -97536.;
            }
            else {
               g = 97536.;
            }
            (*gradient)[3*i+n] += g;
            (*gradient)[3*j+n] -= g;
         }
         const result_type r = std::sqrt(r2);
         return -97536. * r + 52800.;
      }
      // Else
      return Base::evaluate(i, j, x, gradient);
   }

   result_type
   value(const std::size_t i, const std::size_t j, const argument_type& x)
   const {
      const std::array<result_type, 3> p =
      {{x[3*i] - x[3*j], x[3*i+1] - x[3*j+1],  x[3*i+2] - x[3*j+2]}};
      const result_type r2 = stlib::ext::dot(p, p);
      if (r2 < 0.25) {
         const result_type r = std::sqrt(r2);
         return -97536. * r + 52800.;
      }
      // Else
      return Base::value(i, j, x);
   }
};

std::string programName;

void
exitOnError() {
   std::cerr << "Usage:\n"
   << programName << " numAtoms\n";
}

int
main(int argc, char* argv[]) {
   ads::ParseOptionsArguments parser(argc, argv);
   programName = parser.getProgramName();

   if (parser.getNumberOfArguments() != 1) {
      std::cerr << "Wrong number of arguments.\n";
      exitOnError();
   }
   const bool isLinearized = parser.getOption('l');
#ifdef QUASI_NEWTON
   double maxTime = 0;
   const bool useMaxTime = parser.getOption('t', &maxTime);
   std::size_t maxObjFuncCalls = 0;
   const bool useMaxObjFuncCalls = parser.getOption('f', &maxObjFuncCalls);
#endif
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Unparsed options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   std::size_t numAtoms = 0;
   if (! parser.getArgument(&numAtoms)) {
      std::cerr << "Unable to parse the number of atoms.\n";
      exitOnError();
   }
   if (numAtoms == 0) {
      std::cerr << "The number of atoms is not allowed to be zero.\n";
      exitOnError();
   }

   // The initial positions.
   std::vector<double> x(3 * numAtoms, 0.);
   numerical::ContinuousUniformGeneratorOpen<>::DiscreteUniformGenerator
   generator;
   numerical::ContinuousUniformGeneratorOpen<> random(&generator);
   const double length = std::pow(double(numAtoms), 1./3);
   for (std::size_t i = 0; i != x.size(); ++i) {
      x[i] = length * random();
   }

   // Time the minimization.
   double value = 0;
   std::size_t numFunctionCalls = 0;
   ads::Timer timer;
   timer.tic();
   if (isLinearized) {
      LennardJonesLinearized function;
#ifdef QUASI_NEWTON
      numerical::QuasiNewtonLBFGS<LennardJonesLinearized>
         minimizer(function);
      if (useMaxTime) {
         minimizer.setMaxTime(maxTime);
      }
      if (useMaxObjFuncCalls) {
         minimizer.setMaxObjFuncCalls(maxObjFuncCalls);
      }
#elif defined(CONJUGATE_GRADIENT)
      numerical::ConjugateGradient<LennardJonesLinearized>
         minimizer(function);
#elif defined(COORDINATE_DESCENT)
      numerical::CoordinateDescentHookeJeeves<LennardJonesLinearized>
         minimizer(function);
#else
#error Undefined optimization method.
#endif
      try {
         value = minimizer.minimize(&x);
      }
      catch (std::runtime_error error) {
         std::cerr << error.what() << '\n';
         value = function(x);
      }
      numFunctionCalls = minimizer.numFunctionCalls();
   }
   else {
      LennardJones function;
#ifdef QUASI_NEWTON
      numerical::QuasiNewtonLBFGS<LennardJones> minimizer(function);
      if (useMaxTime) {
         minimizer.setMaxTime(maxTime);
      }
      if (useMaxObjFuncCalls) {
         minimizer.setMaxObjFuncCalls(maxObjFuncCalls);
      }
#elif defined(CONJUGATE_GRADIENT)
      numerical::ConjugateGradient<LennardJones> minimizer(function);
#elif defined(COORDINATE_DESCENT)
      numerical::CoordinateDescentHookeJeeves<LennardJones>
         minimizer(function);
#else
#error Undefined optimization method.
#endif
      try {
         value = minimizer.minimize(&x);
      }
      catch (std::runtime_error error) {
         std::cerr << error.what() << '\n';
         value = function(x);
      }
      numFunctionCalls = minimizer.numFunctionCalls();
   }
   double elapsedTime = timer.toc();
   std::cout << "Completed optimization in " << elapsedTime << " seconds.\n"
   << "Value = " << value << ".\n"
   << "Number of function calls = " << numFunctionCalls << '\n'
   << "Positions = \n";
   for (std::size_t i = 0; i < std::min(x.size() / 3, std::size_t(10)); ++i) {
      std::cout << x[3*i] << ' ' << x[3*i+1] << ' ' << x[3*i+2] << '\n';
   }
   if (x.size() / 3 > 10) {
      std::cout << "...\n";
   }

   return 0;
}
