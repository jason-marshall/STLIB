// -*- C++ -*-

#include "stlib/geom/mesh/simplex/ComplexWithFreeVertexOnManifold.h"
#include "stlib/geom/mesh/simplex/SimplexModMeanRatio.h"
#include "stlib/geom/mesh/simplex/SimplexModCondNum.h"

#include "stlib/geom/kernel/ParametrizedLine.h"
#include "stlib/geom/kernel/ParametrizedPlane.h"

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
    typedef geom::ParametrizedLine<2> Manifold;
    typedef Manifold::Point Point;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexMeanRatio,
            2, 1, Manifold > CMR;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexModMeanRatio,
            2, 1, Manifold > CMMR;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexCondNum,
            2, 1, Manifold > CCN;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexModCondNum,
            2, 1, Manifold > CMCN;
    typedef CMR::Vertex Vertex;
    typedef CMR::Face Face;
    typedef CMR::ManifoldPoint ManifoldPoint;

    CMR cmr;
    CMMR cmmr;
    CCN ccn;
    CMCN cmcn;

    Manifold manifold(Point{{0., 0.}}, Point{{1., 0.}});
    cmr.setManifold(&manifold);
    cmmr.setManifold(&manifold);
    ccn.setManifold(&manifold);
    cmcn.setManifold(&manifold);

    Vertex px = {{1, 0}}, py = {{0, 1}}, mx = {{ -1, 0}};
    std::vector<Face> faces;

    faces.push_back(Face{{px, py}});
    faces.push_back(Face{{py, mx}});

    cmr.set(faces.begin(), faces.end());
    cmmr.set(faces.begin(), faces.end());
    ccn.set(faces.begin(), faces.end());
    cmcn.set(faces.begin(), faces.end());

    std::cout << "Vertex incident to 2 reference triangles:\n\n"
              << "Data for the vertex at the origin:\n"
              << "2 norm of mean ratio at (0, 0) = "
              << cmr.computeNorm2(ManifoldPoint{{0.}}) << "\n"
              << "2 norm of modified mean ratio at (0, 0) = "
              << cmmr.computeNorm2Modified(ManifoldPoint{{0.}}) << "\n"
              << "2 norm of condition number at (0, 0) = "
              << ccn.computeNorm2(ManifoldPoint{{0.}}) << "\n"
              << "2 norm of modified condition number at (0, 0) = "
              << cmcn.computeNorm2Modified(ManifoldPoint{{0.}}) << "\n\n"
              << "Data for the inverted triangles:\n"
              << "2 norm of modified mean ratio at (2, 0) = "
              << cmmr.computeNorm2Modified(ManifoldPoint{{2.}}) << "\n"
              << "2 norm of modified condition number at (2, 0) = "
              << cmcn.computeNorm2Modified(ManifoldPoint{{2.}}) << "\n\n";

    {
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::QuasiNewton<1, geom::ComplexManifoldNorm2Mod<CMMR> >
      qn(function);
      ManifoldPoint x = {{0.1}};
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
      // CONTINUE: x = 1 causes a runtime error.
      //x = 1;
      std::fill(x.begin(), x.end(), 0.9);
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
      geom::ComplexManifoldNorm2Mod<CMCN> function(cmcn);
      numerical::QuasiNewton<1, geom::ComplexManifoldNorm2Mod<CMCN> >
      qn(function);
      ManifoldPoint x = {{0.1}};
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
      // CONTINUE: x = 1 causes a runtime error.
      //x = 1;
      std::fill(x.begin(), x.end(), 0.9);
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
      geom::ComplexManifoldNorm2<CMR> function(cmr);
      numerical::Simplex<1, geom::ComplexManifoldNorm2<CMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{0.3}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<1, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{0.3}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<1, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{2}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<1, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{2}};
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
    typedef geom::ParametrizedPlane<3> Manifold;
    typedef Manifold::Point Point;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexMeanRatio,
            3, 2, Manifold > CMR;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexModMeanRatio,
            3, 2, Manifold > CMMR;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexCondNum,
            3, 2, Manifold > CCN;
    typedef geom::ComplexWithFreeVertexOnManifold < geom::SimplexModCondNum,
            3, 2, Manifold > CMCN;
    typedef CMR::Vertex Vertex;
    typedef CMR::Face Face;
    typedef CMR::ManifoldPoint ManifoldPoint;

    CMR cmr;
    CMMR cmmr;
    CCN ccn;
    CMCN cmcn;

    Manifold manifold(Point{{0., 0., 0.}},
                      Point{{1., 0., 0.}},
                      Point{{0., 1., 0.}});
    cmr.setManifold(&manifold);
    cmmr.setManifold(&manifold);
    ccn.setManifold(&manifold);
    cmcn.setManifold(&manifold);

    Vertex px = {{1, 0, 0}},
    py = {{0, 1, 0}},
    pz = {{0, 0, 1}},
    mx = {{ -1, 0, 0}},
    my = {{0, -1, 0}};
    std::vector<Face> faces;

    faces.push_back(Face{{px, py, pz}});
    faces.push_back(Face{{py, mx, pz}});
    faces.push_back(Face{{mx, my, pz}});
    faces.push_back(Face{{my, px, pz}});

    cmr.set(faces.begin(), faces.end());
    cmmr.set(faces.begin(), faces.end());
    ccn.set(faces.begin(), faces.end());
    cmcn.set(faces.begin(), faces.end());

    std::cout << "Vertex incident to 4 reference tets:\n\n"
              << "Data for the vertex at the center:\n"
              << "2 norm of mean ratio at (0, 0, 0) = "
              << cmr.computeNorm2(ManifoldPoint{{0., 0.}}) << "\n"
              << "2 norm of modified mean ratio at (0, 0, 0) = "
              << cmmr.computeNorm2Modified(ManifoldPoint{{0., 0.}}) << "\n"
              << "2 norm of condition number at (0, 0, 0) = "
              << ccn.computeNorm2(ManifoldPoint{{0., 0.}}) << "\n"
              << "2 norm of modified condition number at (0, 0, 0) = "
              << cmcn.computeNorm2Modified(ManifoldPoint{{0., 0.}}) << "\n\n"
              << "Data for the inverted tets:\n"
              << "2 norm of modified mean ratio at (1, 1, 0) = "
              << cmmr.computeNorm2Modified(ManifoldPoint{{1., 1.}}) << "\n"
              << "2 norm of modified condition number at (1, 1, 0) = "
              << cmcn.computeNorm2Modified(ManifoldPoint{{1., 1.}}) << "\n\n";

    {
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::QuasiNewton<2, geom::ComplexManifoldNorm2Mod<CMMR> >
      qn(function);
      ManifoldPoint x = {{0.1, 0.1}};
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
      geom::ComplexManifoldNorm2Mod<CMCN> function(cmcn);
      numerical::QuasiNewton<2, geom::ComplexManifoldNorm2Mod<CMCN> >
      qn(function);
      ManifoldPoint x = {{0.1, 0.1}};
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
      geom::ComplexManifoldNorm2<CMR> function(cmr);
      numerical::Simplex<2, geom::ComplexManifoldNorm2<CMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{0.3, 0.3}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{0.3, 0.3}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{1, 1}};
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
      geom::ComplexManifoldNorm2Mod<CMMR> function(cmmr);
      numerical::Simplex<2, geom::ComplexManifoldNorm2Mod<CMMR> >
      simplex(function);
      ManifoldPoint freeVertex = {{1, 0}};
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
