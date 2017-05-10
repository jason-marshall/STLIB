// -*- C++ -*-

#include "stlib/geom/mesh/simplicial/tile.h"

#include "stlib/geom/mesh/simplicial/file_io.h"
#include "stlib/geom/mesh/simplicial/flip.h"
#include "stlib/geom/mesh/simplicial/manipulators.h"
#include "stlib/geom/mesh/simplicial/quality.h"
#include "stlib/geom/mesh/simplicial/set.h"
#include "stlib/geom/mesh/simplicial/transform.h"

#include "stlib/geom/mesh/iss/optimize.h"

#include <iostream>

#include <cassert>


USING_STLIB_EXT_ARRAY_MATH_OPERATORS;
USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

// Distance function for a unit circle.
class UnitCircle :
  public std::unary_function<const std::array<double, 2>&, double>
{
private:
  typedef std::unary_function<const std::array<double, 2>&, double> Base;

public:
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    return stlib::ext::magnitude(x) - 1;
  }
};


// Closest point function for a unit circle.
class UnitCircleClosestPoint :
  public std::unary_function < const std::array<double, 2>&,
  const std::array<double, 2>& >
{
private:
  typedef std::unary_function < const std::array<double, 2>&,
          const std::array<double, 2>& > Base;

  mutable std::array<double, 2> _p;

public:
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    double mag = stlib::ext::magnitude(x);
    if (mag != 0) {
      _p = x;
      _p /= mag;
    }
    else {
      _p[0] = 1;
      _p[1] = 0;
    }
    return _p;
  }
};


// Distance function for a unit sphere.
class UnitSphere :
  public std::unary_function<const std::array<double, 3>&, double>
{
private:
  typedef std::unary_function<const std::array<double, 3>&, double> Base;

public:
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    return stlib::ext::magnitude(x) - 1;
  }
};


// Closest point function for a unit sphere.
class UnitSphereClosestPoint :
  public std::unary_function < const std::array<double, 3>&,
  const std::array<double, 3>& >
{
private:
  typedef std::unary_function < const std::array<double, 3>&,
          const std::array<double, 3>& > Base;

  mutable std::array<double, 3> _p;

public:
  typedef Base::argument_type argument_type;
  typedef Base::result_type result_type;

  result_type
  operator()(argument_type x) const
  {
    double mag = stlib::ext::magnitude(x);
    if (mag != 0) {
      _p = x;
      _p /= mag;
    }
    else {
      _p[0] = 1;
      _p[1] = 0;
      _p[2] = 0;
    }
    return _p;
  }
};




int
main()
{

  //
  // 2-D, Analytic boundary.
  //

  {
    typedef geom::SimpMeshRed<2, 2> Mesh;
    typedef Mesh::Node Node;

    typedef geom::IndSimpSetIncAdj<2, 2> ISS;

    geom::BBox<double, 2> domain = {{{ -2., -2.}}, {{2., 2.}}};
    double length = 0.1;
    Mesh square;

    // Tile the square.
    geom::tile(domain, length, &square);
    {
      std::cout << "square.vtu\n";
      geom::writeVtkXml(std::cout, square);
    }

    // Mesh the circle via simplex centroid positions.
    Mesh x;
    geom::tile(domain, length, UnitCircle(), &x);
    {
      // Write the initial mesh.
      std::cout << "circ_init.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    // Check the adjacency counts.
    std::array<std::size_t, 4> adjacencyCounts;
    geom::countAdjacencies(x, &adjacencyCounts);
    std::cout << "Circle: initial adjacency counts = " << adjacencyCounts << "\n";

    // Get rid of the simplices with low adjacency counts.
    geom::eraseCellsWithLowAdjacencies(&x, 2);
    geom::countAdjacencies(x, &adjacencyCounts);
    std::cout << "Circle: after removing low adjacency, counts = "
              << adjacencyCounts << "\n";
    {
      // Write the mesh with low adjacency simplices removed.
      std::cout << "circ_min_adj.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    // Move the boundary vertices to the boundary surface.
    std::vector<Node*> bv;
    geom::determineBoundaryNodes(x, std::back_inserter(bv));
    geom::transformNodes<Mesh>(bv.begin(), bv.end(), UnitCircleClosestPoint());
    {
      // Write the mesh with transformed boundary.
      std::cout << "circ_bdry.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    //
    // Optimize the node positions.
    //

    // Make the indexed simplex set.
    ISS iss;
    geom::buildIndSimpSetFromSimpMeshRed(x, &iss);
    // The boundary set and interior set of nodes.
    std::vector<std::size_t> bs, is;
    geom::determineBoundaryVertices(iss, std::back_inserter(bs));
    geom::determineComplementSetOfIndices(iss.vertices.size(), bs.begin(),
                                          bs.end(), std::back_inserter(is));
    for (std::size_t count = 0; count != 5; ++ count) {
      // Optimize the interior node positions.
      geom::geometricOptimizeUsingMeanRatio(&iss, is.begin(), is.end());
      // Optimize the boundary node positions.
      //geom::geometricOptimizeUsingMeanRatio(iss, bs.begin(), bs.end());
      geom::geometricOptimizeConstrainedUsingMeanRatio
      (&iss, bs.begin(), bs.end(), 0.01);
      if (count == 5 - 1) {
        std::cout << "circ_opt_bdry.vtu\n";
        geom::writeVtkXml(std::cout, iss);
      }
      geom::transform(&iss, bs.begin(), bs.end(), UnitCircleClosestPoint());
    }
    x.setVertices(iss.vertices.begin(), iss.vertices.end());

    geom::printQualityStatistics(std::cout, x);

    std::cout << geom::flipUsingModifiedMeanRatio(&x) << " flips.\n";

    geom::printQualityStatistics(std::cout, x);

    {
      // Write the final mesh.
      std::cout << "circ.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }
    std::cout << "\n\n";
  }






  //---------------------------------------------------------------------------
  // 3-D, Analytic boundary.
  //---------------------------------------------------------------------------

  {
    typedef geom::SimpMeshRed<3, 3> Mesh;
    typedef Mesh::Node Node;
    typedef geom::IndSimpSetIncAdj<3, 3> ISS;

    geom::BBox<double, 3> domain = {{{ -2., -2., -2.}}, {{2., 2., 2.}}};
    double length = 0.25;
    Mesh cube;

    // Tile the cube.
    geom::tile(domain, length, &cube);
    {
      std::cout << "cube.vtu\n";
      geom::writeVtkXml(std::cout, cube);
    }

    // Mesh the sphere via simplex centroid positions.
    Mesh x;
    geom::tile(domain, length, UnitSphere(), &x);
    {
      // Write the initial mesh.
      std::cout << "sphere_init.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    // Check the adjacency counts.
    std::array<std::size_t, 5> adjacencyCounts;
    geom::countAdjacencies(x, &adjacencyCounts);
    std::cout << "Sphere: initial adjacency counts = " << adjacencyCounts
              << "\n";

    // Get rid of the simplices with low adjacency counts.
    geom::eraseCellsWithLowAdjacencies(&x, 3);
    geom::countAdjacencies(x, &adjacencyCounts);
    std::cout << "Sphere: after removing low adjacency, counts = "
              << adjacencyCounts << "\n";
    {
      // Write the mesh with low adjacency simplices removed.
      std::cout << "sphere_min_adj.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    // Move the boundary vertices to the boundary surface.
    std::vector<Node*> bv;
    geom::determineBoundaryNodes(x, std::back_inserter(bv));
    geom::transformNodes<Mesh>(bv.begin(), bv.end(), UnitSphereClosestPoint());
    {
      // Write the mesh with transformed boundary.
      std::cout << "sphere_bdry.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }

    //
    // Optimize the node positions.
    //

    // Make the indexed simplex set.
    ISS iss;
    geom::buildIndSimpSetFromSimpMeshRed(x, &iss);
    // The boundary set and interior set of nodes.
    std::vector<std::size_t> bs, is;
    geom::determineBoundaryVertices(iss, std::back_inserter(bs));
    geom::determineComplementSetOfIndices(iss.vertices.size(),
                                          bs.begin(), bs.end(),
                                          std::back_inserter(is));
    // CONTINUE: Fix failed assertion.
#if 0
    for (std::size_t count = 0; count != 5; ++ count) {
      // Optimize the interior node positions.
      // CONTINUE: Assertion failure here.
      geom::geometricOptimizeUsingMeanRatio(&iss, is.begin(), is.end());
      // Optimize the boundary node positions.
      geom::geometricOptimizeUsingMeanRatio(&iss, bs.begin(), bs.end());
      if (count == 5 - 1) {
        std::cout << "sphere_opt_bdry.vtu\n";
        geom::writeVtkXml(std::cout, iss);
      }
      geom::transform(&iss, bs.begin(), bs.end(), UnitSphereClosestPoint());
    }
    x.setVertices(iss.getVerticesBeginning(), iss.getVerticesEnd());

    geom::printQualityStatistics(std::cout, x);

    {
      // Write the final mesh.
      std::cout << "sphere.vtu\n";
      geom::writeVtkXml(std::cout, x);
    }
#endif
  }

  return 0;
}
