// -*- C++ -*-

#include "stlib/levelSet/MolecularSurfaces.h"
#include "stlib/ads/iterator/TrivialOutputIterator.h"
#include "stlib/numerical/constants.h"
#include "stlib/numerical/equality.h"

using namespace stlib;

int
main()
{
  typedef float T;
  const T Pi = numerical::Constants<T>::Pi();
  const T ProbeRadius = 1.4;
  const T Dx = 0.1;
  ads::TrivialOutputIterator vertices;

  // 1-D
  {
    const std::size_t D = 1;
    const std::size_t N = 2;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> atoms;
    atoms.push_back(Ball{{{0}}, T(1)});
    levelSet::MolecularSurfaces<T, D, N> ms(atoms, ProbeRadius, Dx);
    // van der Waals.
    {
      std::pair<T, T> content = ms.vanDerWaals(vertices);
      const T R = 1;
      const T Content = 2 * R;
      const T Boundary = 1;
      assert(std::abs(content.first - Content) < 2 * Dx * Dx);
      assert(content.second == Boundary);
    }
    // Solvent accessible.
    {
      std::pair<T, T> content = ms.solventAccessible(vertices);
      const T R = atoms[0].radius + ProbeRadius;
      const T Content = 2 * R;
      const T Boundary = 1;
      assert(std::abs(content.first - Content) < 2 * Dx * Dx);
      assert(content.second == Boundary);
    }
  }
  // 2-D
  {
    const std::size_t D = 2;
    const std::size_t N = 8;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> atoms;
    atoms.push_back(Ball{{{0, 0}}, T(1)});
    levelSet::MolecularSurfaces<T, D, N> ms(atoms, ProbeRadius, Dx);
    // van der Waals.
    {
      std::pair<T, T> content = ms.vanDerWaals(vertices);
      const T R = 1;
      const T Content = Pi * R * R;
      const T Boundary = 2 * Pi * R;
      assert(std::abs(content.first - Content) < Boundary * Dx * Dx);
      assert(std::abs(content.second - Boundary) < Boundary * Dx * Dx);
    }
    // Solvent accessible.
    {
      std::pair<T, T> content = ms.solventAccessible(vertices);
      const T R = atoms[0].radius + ProbeRadius;
      const T Content = Pi * R * R;
      const T Boundary = 2 * Pi * R;
      assert(std::abs(content.first - Content) < Boundary * Dx * Dx);
      assert(std::abs(content.second - Boundary) < Boundary * Dx * Dx);
    }
  }
  // 3-D
  {
    const std::size_t D = 3;
    const std::size_t N = 8;
    typedef geom::Ball<T, D> Ball;

    std::vector<Ball> atoms;
    atoms.push_back(Ball{{{0, 0, 0}}, T(1)});
    levelSet::MolecularSurfaces<T, D, N> ms(atoms, ProbeRadius, Dx);
    // van der Waals.
    {
      std::pair<T, T> content = ms.vanDerWaals(vertices);
      const T R = 1;
      const T Content = 4 * Pi * R * R * R / 3;
      const T Boundary = 4 * Pi * R * R;
      assert(std::abs(content.first - Content) < Boundary * Dx * Dx);
      assert(std::abs(content.second - Boundary) < Boundary * Dx * Dx);
    }
    // Solvent accessible.
    {
      std::pair<T, T> content = ms.solventAccessible(vertices);
      const T R = atoms[0].radius + ProbeRadius;
      const T Content = 4 * Pi * R * R * R / 3;
      const T Boundary = 4 * Pi * R * R;
      assert(std::abs(content.first - Content) < Boundary * Dx * Dx);
      assert(std::abs(content.second - Boundary) < Boundary * Dx * Dx);
    }
  }

  return 0;
}
