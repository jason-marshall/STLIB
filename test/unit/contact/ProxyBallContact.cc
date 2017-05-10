// -*- C++ -*-

#include "stlib/contact/ProxyBallContact.h"

#include <iostream>
#include <sstream>

using namespace stlib;

// Point.
#define PT ext::make_array<double>
// A simplex of identifiers or indices.
#define IS ext::make_array<std::size_t>

int
main()
{
  const double Eps = 10 * std::numeric_limits<double>::epsilon();

  // Use primes for the node identifiers.
  const std::size_t p[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 47};
  //
  // 1-D space.
  //
  {
    const std::size_t N = 1;
    typedef contact::ProxyBallContact<N> ProxyBallContact;
    typedef ProxyBallContact::Point Point;
    typedef std::tuple<std::size_t, Point> Force;

    {
      // 0 components. Empty mesh.
      const double maximumRelativePenetration = 0.1;
      ProxyBallContact f(0, 0, 0, 0, 0, 0, maximumRelativePenetration);
      std::vector<Force> forces;
      const double dt = f(0, 0, 0, 0, std::back_inserter(forces));
      assert(dt == std::numeric_limits<double>::max());
      assert(forces.empty());
      std::cout << "0 components. Empty mesh.\n" << f << '\n';
    }
    {
      // 1 component. 2 simplices. No contact.
      const std::size_t numberOfComponents = 1;
      const double nodeCoordinates[] = {0, 1, 2};
      const std::size_t numberOfNodes = sizeof(nodeCoordinates) /
                                        sizeof(std::size_t) / N;
      const std::size_t* nodeIdentifiers = p;
      const std::size_t identifierSimplices[] = {p[0], p[1],
                                                 p[1], p[2]
                                                };
      const std::size_t numberOfElements = sizeof(identifierSimplices) /
                                           sizeof(std::size_t) / (N + 1);
      const std::size_t components[] = {0, 0};
      const double maximumRelativePenetration = 0.1;
      ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                         nodeIdentifiers, numberOfElements, identifierSimplices,
                         maximumRelativePenetration);
      assert(f == f);
      const double velocityCoordinates[] = {1, 0, -1};
      const double masses[] = {1, 1};
      std::vector<Force> forces;
      const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                          components, std::back_inserter(forces));
      assert(std::abs(dt - 0.05) < 0.05 * Eps);
      assert(forces.empty());
    }
    {
      // 1 component. 2 simplices. Overlap. No contact.
      const std::size_t numberOfComponents = 1;
      const double nodeCoordinates[] = {0, 1, 0.5, 1.5};
      const std::size_t numberOfNodes = sizeof(nodeCoordinates) /
                                        sizeof(std::size_t) / N;
      const std::size_t* nodeIdentifiers = p;
      const std::size_t identifierSimplices[] = {p[0], p[1],
                                                 p[2], p[3]
                                                };
      const std::size_t numberOfElements = sizeof(identifierSimplices) /
                                           sizeof(std::size_t) / (N + 1);
      const std::size_t components[] = {0, 0};
      const double maximumRelativePenetration = 0.1;
      ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                         nodeIdentifiers, numberOfElements, identifierSimplices,
                         maximumRelativePenetration);

      const double velocityCoordinates[] = {1, 1, -1, -1};
      const double masses[] = {1, 1};
      std::vector<Force> forces;
      const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                          components, std::back_inserter(forces));
      assert(std::abs(dt - 0.025) < 0.025 * Eps);
      assert(forces.empty());
    }
    {
      // 2 components. 2 simplices. Overlap. 1 contact.
      const std::size_t numberOfComponents = 2;
      const double nodeCoordinates[] = {0, 1, 0.5, 1.5};
      const std::size_t numberOfNodes = sizeof(nodeCoordinates) /
                                        sizeof(std::size_t) / N;
      const std::size_t* nodeIdentifiers = p;
      const std::size_t identifierSimplices[] = {p[0], p[1],
                                                 p[2], p[3]
                                                };
      const std::size_t numberOfElements = sizeof(identifierSimplices) /
                                           sizeof(std::size_t) / (N + 1);
      const std::size_t components[] = {0, 1};
      const double maximumRelativePenetration = 0.1;

      {
        // Compression.
        const double velocityCoordinates[] = {1, 1, -1, -1};
        const double masses[] = {1, 1};

        {
          // Spring force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] + 100) < 100 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] - 100) < 100 * Eps);
          // Check the restart capability.
          // Write to a file.
          std::ostringstream out;
          out << f;
          ProxyBallContact g(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration);
          g(nodeCoordinates, velocityCoordinates, masses,
            components, std::back_inserter(forces));
          // Read from a file.
          std::istringstream in(out.str().c_str());
          in >> g;
          std::cout << f << '\n' << g << '\n';
          assert(f == g);
        }
        {
          // Damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             0, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] + 40) < 40 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] - 40) < 40 * Eps);
        }
        {
          // Spring and damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             1, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] + 140) < 140 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] - 140) < 140 * Eps);
        }
      }
      {
        // Expansion.
        const double velocityCoordinates[] = { -1, -1, 1, 1};
        const double masses[] = {1, 1};

        {
          // Spring force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] + 100) < 100 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] - 100) < 100 * Eps);
        }
        {
          // Damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             0, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] - 40) < 40 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] + 40) < 40 * Eps);
        }
        {
          // Spring and damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             1, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(std::abs(dt - 0.025) < 0.025 * Eps);
          assert(forces.size() == 2);
          assert(std::get<0>(forces[0]) == 0);
          assert(std::abs(std::get<1>(forces[0])[0] + 60) < 60 * Eps);
          assert(std::get<0>(forces[1]) == 1);
          assert(std::abs(std::get<1>(forces[1])[0] - 60) < 60 * Eps);
        }
      }
      {
        // Zero velocities.
        const double velocityCoordinates[] = {0, 0, 0, 0};
        const double masses[] = {1, 1};

        {
          // Spring force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(dt > 1e10);
          assert(forces.size() == 0);
        }
        {
          // Damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             0, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(dt > 1e10);
          assert(forces.size() == 0);
        }
        {
          // Spring and damping force.
          ProxyBallContact f(numberOfComponents, numberOfNodes, nodeCoordinates,
                             nodeIdentifiers, numberOfElements,
                             identifierSimplices, maximumRelativePenetration,
                             1, 1);
          std::vector<Force> forces;
          const double dt = f(nodeCoordinates, velocityCoordinates, masses,
                              components, std::back_inserter(forces));
          assert(dt > 1e10);
          assert(forces.size() == 0);
        }
      }
    }
  }
  return 0;
}
