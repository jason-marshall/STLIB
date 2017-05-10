// -*- C++ -*-

#include "stlib/geom/mesh/simplex/ComplexWithFreeVertex.h"

#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include "stlib/numerical/optimization/staticDimension/QuasiNewton.h"
#include "stlib/numerical/optimization/staticDimension/Simplex.h"

#include <iostream>
#include <vector>

#include <cassert>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{
  //-------------------------------------------------------------------------
  // 2-D
  //-------------------------------------------------------------------------
  {
    typedef geom::ComplexWithFreeVertex<geom::SimplexMeanRatio, 2> CMR;
    typedef geom::ComplexWithFreeVertex<geom::SimplexModMeanRatio, 2> CMMR;
    typedef geom::ComplexWithFreeVertex<geom::SimplexCondNum, 2> CCN;
    typedef geom::ComplexWithFreeVertex<geom::SimplexModCondNum, 2> CMCN;
    typedef CMR::Vertex Vertex;
    typedef CMR::Face Face;

    CMR cmr;
    CMMR cmmr;
    CCN ccn;
    CMCN cmcn;

    Vertex px = {{1, 0}}, py = {{0, 1}}, mx = {{ -1, 0}}, my = {{0, -1}};
    std::vector<Face> faces;

    faces.push_back(Face{{px, py}});
    faces.push_back(Face{{py, mx}});
    faces.push_back(Face{{mx, my}});
    faces.push_back(Face{{my, px}});

    cmr.set(faces.begin(), faces.end());
    cmmr.set(faces.begin(), faces.end());
    ccn.set(faces.begin(), faces.end());
    cmcn.set(faces.begin(), faces.end());

    std::cout << "Vertex surrounded by 4 reference triangles:\n\n"
              << "Data for the vertex at the center:\n"
              << "2 norm of mean ratio at (0, 0) = "
              << cmr.computeNorm2(Vertex{{0., 0.}}) << "\n"
              << "2 norm of modified mean ratio at (0, 0) = "
              << cmmr.computeNorm2Modified(Vertex{{0., 0.}}) << "\n"
              << "2 norm of condition number at (0, 0) = "
              << ccn.computeNorm2(Vertex{{0., 0.}}) << "\n"
              << "2 norm of modified condition number at (0, 0) = "
              << cmcn.computeNorm2Modified(Vertex{{0., 0.}}) << "\n\n"
              << "Data for the inverted triangles:\n"
              << "2 norm of modified mean ratio at (1, 1) = "
              << cmmr.computeNorm2Modified(Vertex{{1., 1.}}) << "\n"
              << "2 norm of modified condition number at (1, 1) = "
              << cmcn.computeNorm2Modified(Vertex{{1., 1.}}) << "\n\n";

    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::QuasiNewton<2, geom::ComplexNorm2Mod<CMMR> > qn(function);
      Vertex x = {{0.1, 0.1}};
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      double value;
      std::size_t numberOfIterations;
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 0.3);
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 1);
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      // Since the gradient is large, I need to specify the maximum allowed
      // step.
      qn.find_minimum(x, value, numberOfIterations, 1.0);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMCN> function(cmcn);
      numerical::QuasiNewton<2, geom::ComplexNorm2Mod<CMCN> > qn(function);
      Vertex x = {{0.1, 0.1}};
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      double value;
      std::size_t numberOfIterations;
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 0.3);
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 1);
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      // Since the gradient is large, I need to specify the maximum allowed
      // step.
      qn.find_minimum(x, value, numberOfIterations, 1.0);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2<CMR> function(cmr);
      numerical::Simplex<2, geom::ComplexNorm2<CMR> > simplex(function);
      Vertex freeVertex = {{0.3, 0.3}};
      std::cout << "Optimize the mean ratio with downhill simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{0.3, 0.3}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{1, 1}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{1, 0}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
  }

  //-------------------------------------------------------------------------
  // 3-D
  //-------------------------------------------------------------------------
  {
    typedef geom::ComplexWithFreeVertex<geom::SimplexMeanRatio, 3> CMR;
    typedef geom::ComplexWithFreeVertex<geom::SimplexModMeanRatio, 3> CMMR;
    typedef geom::ComplexWithFreeVertex<geom::SimplexCondNum, 3> CCN;
    typedef geom::ComplexWithFreeVertex<geom::SimplexModCondNum, 3> CMCN;
    typedef CMR::Vertex Vertex;
    typedef CMR::Face Face;

    CMR cmr;
    CMMR cmmr;
    CCN ccn;
    CMCN cmcn;

    Vertex px = {{1, 0, 0}}, py = {{0, 1, 0}}, pz = {{0, 0, 1}},
    mx = {{ -1, 0, 0}}, my = {{0, -1, 0}}, mz = {{0, 0, -1}};
    std::vector<Face> faces;

    faces.push_back(Face{{px, py, pz}});
    faces.push_back(Face{{py, mx, pz}});
    faces.push_back(Face{{mx, my, pz}});
    faces.push_back(Face{{my, px, pz}});
    faces.push_back(Face{{py, px, mz}});
    faces.push_back(Face{{px, my, mz}});
    faces.push_back(Face{{my, mx, mz}});
    faces.push_back(Face{{mx, py, mz}});

    cmr.set(faces.begin(), faces.end());
    cmmr.set(faces.begin(), faces.end());
    ccn.set(faces.begin(), faces.end());
    cmcn.set(faces.begin(), faces.end());

    std::cout << "Vertex surrounded by 8 reference tets:\n\n"
              << "Data for the vertex at the center:\n"
              << "2 norm of mean ratio at (0, 0, 0) = "
              << cmr.computeNorm2(Vertex{{0., 0., 0.}}) << "\n"
              << "2 norm of modified mean ratio at (0, 0, 0) = "
              << cmmr.computeNorm2Modified(Vertex{{0., 0., 0.}}) << "\n"
              << "2 norm of condition number at (0, 0, 0) = "
              << ccn.computeNorm2(Vertex{{0., 0., 0.}}) << "\n"
              << "2 norm of modified condition number at (0, 0, 0) = "
              << cmcn.computeNorm2Modified(Vertex{{0., 0., 0.}}) << "\n\n"
              << "Data for the inverted tets:\n"
              << "2 norm of modified mean ratio at (1, 1, 1) = "
              << cmmr.computeNorm2Modified(Vertex{{1., 1., 1.}}) << "\n"
              << "2 norm of modified condition number at (1, 1, 1) = "
              << cmcn.computeNorm2Modified(Vertex{{1., 1., 1.}}) << "\n\n";

    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::QuasiNewton<3, geom::ComplexNorm2Mod<CMMR> > qn(function);
      Vertex x = {{0.1, 0.1, 0.1}};
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      double value;
      std::size_t numberOfIterations;
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 0.3);
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
      std::fill(x.begin(), x.end(), 1);
      std::cout << "Optimize the modified mean ratio with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      // Since the gradient is large, I need to specify the maximum allowed
      // step.
      qn.find_minimum(x, value, numberOfIterations, 1.0);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMCN> function(cmcn);
      numerical::QuasiNewton<3, geom::ComplexNorm2Mod<CMCN> > qn(function);
      Vertex x = {{0.1, 0.1, 0.1}};
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      double value;
      std::size_t numberOfIterations;
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";

      std::fill(x.begin(), x.end(), 0.3);
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      qn.find_minimum(x, value, numberOfIterations);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
      std::fill(x.begin(), x.end(), 1);
      std::cout << "Optimize the modified condition number with a "
                << "quasi-Newton method.\n"
                << "Initial position = " << x << "\n"
                << "x tolerance  = " << qn.x_tolerance() << "\n"
                << "gradient tolerance  = " << qn.gradient_tolerance() << "\n";
      // Since the gradient is large, I need to specify the maximum allowed
      // step.
      qn.find_minimum(x, value, numberOfIterations, 1.0);
      std::cout << "Minimum point = " << x << "\n"
                << "Minimum value = " << value << "\n"
                << "Number of iterations = " << numberOfIterations << "\n"
                << "Number of function evaluations = "
                << qn.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2<CMR> function(cmr);
      numerical::Simplex<3, geom::ComplexNorm2<CMR> > simplex(function);
      Vertex freeVertex = {{0.3, 0.3, 0.3}};
      std::cout << "Optimize the mean ratio with downhill simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<3, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{0.3, 0.3, 0.3}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<3, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{1, 1, 1}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
    {
      geom::ComplexNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<3, geom::ComplexNorm2Mod<CMMR> > simplex(function);
      Vertex freeVertex = {{1, 0, 0}};
      std::cout << "Optimize the modified mean ratio with the downhill "
                << "simplex method.\n"
                << "Initial position = " << freeVertex << "\n"
                << "tolerance  = default\n"
                << "offset = default\n";
      bool result = simplex.find_minimum(freeVertex);
      std::cout << "Result = " << result << "\n"
                << "Minimum point = " << simplex.minimum_point() << "\n"
                << "Minimum value = " << simplex.minimum_value() << "\n"
                << "Number of function calls = "
                << simplex.num_function_calls() << "\n\n";
    }
  }

  return 0;
}
