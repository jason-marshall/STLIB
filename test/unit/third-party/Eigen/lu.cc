// -*- C++ -*-

#include <Eigen/LU>
#include <Eigen/Core>

#include <iostream>

#include <cassert>

int
main()
{
  // Solve A x = b
  const std::size_t rows = 2;
  const std::size_t cols = 2;
  Eigen::MatrixXd a;
  a.setRandom(rows, cols);
  Eigen::VectorXd b;
  b.setRandom(cols);
  Eigen::VectorXd x(cols);
  Eigen::FullPivLU<Eigen::MatrixXd> lu(a);
  x = lu.solve(b);
  // Old method.
  //a.lu().solve(b, &x);

  Eigen::VectorXd error = a * x - b;
  assert(error.cwiseAbs().maxCoeff() <
         10. * std::numeric_limits<double>::epsilon());

  return 0;
}
