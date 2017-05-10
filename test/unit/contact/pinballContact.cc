// -*- C++ -*-

#include "stlib/contact/pinballContact.h"
#include "stlib/geom/mesh/iss/build.h"

#include <iostream>
#include <sstream>

USING_STLIB_EXT_ARRAY_IO_OPERATORS;
using namespace stlib;

int
main()
{
  const double Eps = 10 * std::numeric_limits<double>::epsilon();

  std::vector<std::size_t> primes;
  primes.push_back(2);
  primes.push_back(3);
  primes.push_back(5);
  primes.push_back(7);
  primes.push_back(11);
  primes.push_back(13);
  primes.push_back(17);
  primes.push_back(19);
  primes.push_back(23);
  primes.push_back(29);
  primes.push_back(31);
  primes.push_back(37);
  primes.push_back(41);
  primes.push_back(47);
  //
  // 1-D space.
  //
  {
    const std::size_t N = 1;
    typedef geom::IndSimpSetIncAdj<N, N> Iss;
    typedef std::array<double, N> Point;
    typedef std::array < std::size_t, N + 1 > IdentifierSimplex;
    typedef std::tuple<std::size_t, std::size_t, Point, double> Contact;
    typedef std::tuple<std::size_t, Point> Force;

    {
      // Empty mesh. Mesh interface.
      Iss mesh;
      std::vector<Point> velocities;
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // Empty mesh. Vector interface.
      std::vector<Point> vertices;
      std::vector<std::size_t> vertexIdentifiers;
      std::vector<IdentifierSimplex> simplices;
      {
        std::vector<Point> velocities;
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(vertices, vertexIdentifiers, velocities, simplices,
                                maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities;
        std::vector<double> masses;
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(vertices, vertexIdentifiers, velocities,
                               simplices, masses, std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. No contact. Mesh interface.
      double vertices[] = {0, 1, 2};
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1,
                                 1, 2
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(0));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // 2 simplices. No contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0.}});
      v.push_back(Point{{1.}});
      v.push_back(Point{{2.}});
      // Vertex identifiers.
      std::vector<std::size_t> vi;
      vi.push_back(2);
      vi.push_back(3);
      vi.push_back(5);
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1]}});
      s.push_back(IdentifierSimplex{{vi[1], vi[2]}});

      std::vector<Point> velocities(v.size(), ext::filled_array<Point>(0));
      const double maximumRelativePenetration = 0.1;
      {
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(0));
        std::vector<double> masses(s.size());
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. 1 contact. Mesh interface.
      double vertices[] = {0, 1, 0.5, 1.5};
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1,
                                 2, 3
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(0));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.size() == 1);
      assert(std::get<0>(contacts[0]) == 0);
      assert(std::get<1>(contacts[0]) == 1);
      assert(std::abs(std::get<2>(contacts[0])[0] + 0.5) < Eps);
    }
    {
      // 2 simplices. 1 contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0.}});
      v.push_back(Point{{1.}});
      v.push_back(Point{{0.5}});
      v.push_back(Point{{1.5}});
      // Vertex identifiers.
      std::vector<std::size_t> vi;
      vi.push_back(2);
      vi.push_back(3);
      vi.push_back(5);
      vi.push_back(7);
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1]}});
      s.push_back(IdentifierSimplex{{vi[2], vi[3]}});

      {
        std::vector<Point> velocities(v.size(), Point{{1.}});
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.size() == 1);
        assert(std::get<0>(contacts[0]) == 0);
        assert(std::get<1>(contacts[0]) == 1);
        assert(std::abs(std::get<2>(contacts[0])[0] + 0.5) < Eps);
      }
      {
        // With zero velocities, there will be no forces.
        std::vector<Point> velocities(v.size(), Point{{0.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 0.5;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.empty());
      }
      {
        // Velocities = 1.
        std::vector<Point> velocities(v.size(), Point{{1.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 0.5;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.size() == s.size());
        std::vector<Point> expectedForces;
        expectedForces.push_back(Point{{-2.}});
        expectedForces.push_back(Point{{-2.}});
        expectedForces.push_back(Point{{2.}});
        expectedForces.push_back(Point{{2.}});
        for (std::size_t i = 0; i != forces.size(); ++i) {
          assert(std::get<0>(forces[i]) == i);
          std::cout << std::get<1>(forces[i]) << '\n';
          //assert(std::get<1>(forces[i]) == expectedForces[i]);
        }
        // Use the saved spring constants.
        std::vector<Force> sameForces;
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(sameForces));
        assert(sameForces.size() == s.size());
        assert(std::equal(forces.begin(), forces.end(), sameForces.begin()));
        // Write to a file.
        std::ostringstream out;
        out << pinballRestoringForces;
        contact::PinballRestoringForces<N>::HashTable
        hashTable(pinballRestoringForces.getSpringAndDampingConstants().begin(),
                  pinballRestoringForces.getSpringAndDampingConstants().end());
        assert(hashTable.size() ==
               pinballRestoringForces.getSpringAndDampingConstants().size());
        // Read from a file.
        std::istringstream in(out.str().c_str());
        in >> pinballRestoringForces;
        assert(hashTable.size() ==
               pinballRestoringForces.getSpringAndDampingConstants().size());
        assert(std::equal(hashTable.begin(), hashTable.end(),
                          pinballRestoringForces.getSpringAndDampingConstants().begin()));
      }
      {
        // Velocities = 1.
        std::vector<Point> velocities(v.size(), Point{{1.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 0.5;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.size() == s.size());
        std::vector<Point> expectedForces;
        expectedForces.push_back(Point{{-2.}});
        expectedForces.push_back(Point{{-2.}});
        expectedForces.push_back(Point{{2.}});
        expectedForces.push_back(Point{{2.}});
        for (std::size_t i = 0; i != forces.size(); ++i) {
          assert(std::get<0>(forces[i]) == i);
          std::cout << std::get<1>(forces[i]) << '\n';
          //assert(std::get<1>(forces[i]) == expectedForces[i]);
        }
      }
    }
    {
      // 2 simplices. 1 contact, full overlap. Mesh interface.
      double vertices[] = {0, 1, 0, 1};
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1,
                                 2, 3
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(0));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.size() == 1);
      assert(std::get<0>(contacts[0]) == 0);
      assert(std::get<1>(contacts[0]) == 1);
      assert(std::abs(std::get<2>(contacts[0])[0] - 1.) < Eps);
    }
    {
      // 2 simplices. 1 contact, full overlap. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0.}});
      v.push_back(Point{{1.}});
      v.push_back(Point{{0.}});
      v.push_back(Point{{1.}});
      // Vertex identifiers.
      std::vector<std::size_t> vi;
      vi.push_back(2);
      vi.push_back(3);
      vi.push_back(5);
      vi.push_back(7);
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1]}});
      s.push_back(IdentifierSimplex{{vi[2], vi[3]}});

      {
        std::vector<Point> velocities(v.size(), Point{{1.}});
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.size() == 1);
        assert(std::get<0>(contacts[0]) == 0);
        assert(std::get<1>(contacts[0]) == 1);
        assert(std::abs(std::get<2>(contacts[0])[0] - 1.) < Eps);
      }
      {
        // Velocities = 1.
        std::vector<Point> velocities(v.size(), Point{{1.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 1.;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.size() == s.size());
        std::vector<Point> expectedForces;
        expectedForces.push_back(Point{{1.}});
        expectedForces.push_back(Point{{1.}});
        expectedForces.push_back(Point{{-1.}});
        expectedForces.push_back(Point{{-1.}});
        for (std::size_t i = 0; i != forces.size(); ++i) {
          assert(std::get<0>(forces[i]) == i);
          std::cout << std::get<1>(forces[i]) << '\n';
          //assert(std::get<1>(forces[i]) == expectedForces[i]);
        }
      }
    }
  }

  //
  // 2-D space.
  //
  {
    const std::size_t N = 2;
    typedef geom::IndSimpSetIncAdj<N, N> Iss;
    typedef std::array<double, N> Point;
    typedef std::array < std::size_t, N + 1 > IdentifierSimplex;
    typedef std::tuple<std::size_t, std::size_t, Point, double> Contact;
    typedef std::tuple<std::size_t, Point> Force;

    {
      // Empty mesh. Mesh interface.
      Iss mesh;
      std::vector<Point> velocities;
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // Empty mesh. Vector interface.
      std::vector<Point> vertices;
      std::vector<std::size_t> vertexIdentifiers;
      std::vector<IdentifierSimplex> simplices;
      {
        std::vector<Point> velocities;
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(vertices, vertexIdentifiers, velocities, simplices,
                                maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities;
        std::vector<double> masses;
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(vertices, vertexIdentifiers, velocities,
                               simplices, masses, std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. No contact. Mesh interface.
      double vertices[] = {0, 0,
                           1, 0,
                           1, 1,
                           0, 1
                          };
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1, 2,
                                 2, 3, 0
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(1));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // 2 simplices. No contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0, 0}});
      v.push_back(Point{{1, 0}});
      v.push_back(Point{{1, 1}});
      v.push_back(Point{{0, 1}});
      // Vertex identifiers.
      std::vector<std::size_t> vi;
      vi.push_back(2);
      vi.push_back(3);
      vi.push_back(5);
      vi.push_back(7);
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1], vi[2]}});
      s.push_back(IdentifierSimplex{{vi[2], vi[3], vi[0]}});

      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(1));
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(1));
        std::vector<double> masses(s.size());
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. 1 contact. Mesh interface.
      double vertices[] = {0, 1,
                           0, 0,
                           1, 0,
                           1, 0,
                           1, 1,
                           0, 1
                          };
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1, 2,
                                 3, 4, 5
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);
      // Centroids: (1/3, 1/3), (2/3, 2/3)
      // Distance bewteen = sqrt(2) / 3 = 0.47140452079103173
      // Area of triangle = 1/2
      // Radius = sqrt(5)/6
      // Penetration = sqrt(5)/3 - sqrt(2)/3 = (sqrt(5) - sqrt(2))/3

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(1));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.size() == 1);
      //std::cerr << std::get<0>(contacts[0]) << ' '
      //<< std::get<1>(contacts[0]) << '\n';

      assert(std::get<0>(contacts[0]) +
             std::get<1>(contacts[0]) == 1);
      double pc = (std::sqrt(5.) - std::sqrt(2.)) / 3. / std::sqrt(2.);
      if (std::get<0>(contacts[0]) == 1) {
        pc = - pc;
      }
      //std::cerr << std::get<2>(contacts[0]) << '\n'
      //<< std::get<3>(contacts[0]) << '\n';
      assert(std::abs(std::get<2>(contacts[0])[0] + pc) < Eps);
      assert(std::abs(std::get<2>(contacts[0])[1] + pc) < Eps);
    }
    {
      // 2 simplices. One contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0, 1}});
      v.push_back(Point{{0, 0}});
      v.push_back(Point{{1, 0}});
      v.push_back(Point{{1, 0}});
      v.push_back(Point{{1, 1}});
      v.push_back(Point{{0, 1}});
      // Vertex identifiers.
      std::vector<std::size_t> vi;
      vi.push_back(2);
      vi.push_back(3);
      vi.push_back(5);
      vi.push_back(7);
      vi.push_back(11);
      vi.push_back(13);
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1], vi[2]}});
      s.push_back(IdentifierSimplex{{vi[3], vi[4], vi[5]}});

      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(1));
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.size() == 1);
        assert(std::get<0>(contacts[0]) +
               std::get<1>(contacts[0]) == 1);
        double pc = (std::sqrt(5.) - std::sqrt(2.)) / 3. / std::sqrt(2.);
        if (std::get<0>(contacts[0]) == 1) {
          pc = - pc;
        }
        assert(std::abs(std::get<2>(contacts[0])[0] + pc) < Eps);
        assert(std::abs(std::get<2>(contacts[0])[1] + pc) < Eps);
      }
      {
        // Velocities = 1.
        std::vector<Point> velocities(v.size(), Point{{1., 1.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 0.5;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.size() == s.size());
        for (std::size_t i = 0; i != forces.size(); ++i) {
          assert(std::get<0>(forces[i]) == i);
        }
        std::cout << "2-D contact\n";
        for (std::size_t i = 0; i != forces.size(); ++i) {
          std::cout << std::get<0>(forces[i]) << ' '
                    << std::get<1>(forces[i]) << '\n';
        }
      }
    }
  }

  //
  // 3-D space.
  //
  {
    const std::size_t N = 3;
    typedef geom::IndSimpSetIncAdj<N, N> Iss;
    typedef std::array<double, N> Point;
    typedef std::array < std::size_t, N + 1 > IdentifierSimplex;
    typedef std::tuple<std::size_t, std::size_t, Point, double> Contact;
    typedef std::tuple<std::size_t, Point> Force;

    {
      // Empty mesh. Mesh interface.
      Iss mesh;
      std::vector<Point> velocities;
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // Empty mesh. Vector interface.
      std::vector<std::array<double, N> > vertices;
      std::vector<std::size_t> vertexIdentifiers;
      std::vector < std::array < std::size_t, N + 1 > > simplices;
      {
        std::vector<Point> velocities;
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(vertices, vertexIdentifiers, velocities, simplices,
                                maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities;
        std::vector<double> masses;
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(vertices, vertexIdentifiers, velocities,
                               simplices, masses, std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. No contact. Mesh interface.
      typedef geom::IndSimpSetIncAdj<3, 3> Iss;
      typedef std::array<double, 3> Point;
      typedef std::tuple<std::size_t, std::size_t, Point, double> Contact;

      //
      // Data for an octahedron
      //
      const std::size_t numVertices = 7;
      double vertices[] = {0, 0, 0,
                           1, 0, 0,
                           -1, 0, 0,
                           0, 1, 0,
                           0, -1, 0,
                           0, 0, 1,
                           0, 0, -1
                          };
      const std::size_t num_tets = 8;
      std::size_t tets[] = {0, 1, 3, 5,
                            0, 3, 2, 5,
                            0, 2, 4, 5,
                            0, 4, 1, 5,
                            0, 3, 1, 6,
                            0, 2, 3, 6,
                            0, 4, 2, 6,
                            0, 1, 4, 6
                           };

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, num_tets, tets);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(1));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.empty());
    }
    {
      // 8 simplices. No contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0, 0, 0}});
      v.push_back(Point{{1, 0, 0}});
      v.push_back(Point{{-1, 0, 0}});
      v.push_back(Point{{0, 1, 0}});
      v.push_back(Point{{0, -1, 0}});
      v.push_back(Point{{0, 0, 1}});
      v.push_back(Point{{0, 0, -1}});
      // Vertex identifiers.
      std::vector<std::size_t> vi(primes.begin(), primes.begin() + v.size());
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1], vi[3], vi[5]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[3], vi[2], vi[5]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[2], vi[4], vi[5]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[4], vi[1], vi[5]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[3], vi[1], vi[6]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[2], vi[3], vi[6]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[4], vi[2], vi[6]}});
      s.push_back(IdentifierSimplex{{vi[0], vi[1], vi[4], vi[6]}});

      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(1));
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.empty());
      }
      {
        std::vector<Point> velocities(v.size());
        std::vector<double> masses(s.size());
        const double maximumRelativePenetration = 0.1;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.empty());
      }
    }
    {
      // 2 simplices. 1 contact. Mesh interface.
      double vertices[] = {0, 0, 0,
                           1, 0, 0,
                           0, 1, 0,
                           0, 0, 1,
                           0.5, 0, 0,
                           1.5, 0, 0,
                           0.5, 1, 0,
                           0.5, 0, 1
                          };
      const std::size_t numVertices = sizeof(vertices) / sizeof(double) / N;
      std::size_t simplices[] = {0, 1, 2, 3,
                                 4, 5, 6, 7
                                };
      const std::size_t numSimplices = sizeof(simplices) / sizeof(std::size_t) /
                                       (N + 1);
      // Distance bewteen centroids = 0.5.
      // Volume = 1/6
      // Radius = sqrt(3)/4
      // Penetration = sqrt(3)/2 - 0.5

      // Construct from vertices and tetrahedra.
      Iss mesh;
      geom::build(&mesh, numVertices, vertices, numSimplices, simplices);
      std::vector<Point> velocities(numVertices, ext::filled_array<Point>(1));
      const double maximumRelativePenetration = 0.1;
      std::vector<Contact> contacts;
      contact::pinballContact(mesh, velocities, maximumRelativePenetration,
                              std::back_inserter(contacts));
      assert(contacts.size() == 1);
      assert(std::get<0>(contacts[0]) +
             std::get<1>(contacts[0]) == 1);
      double pc = 0.5 * std::sqrt(3.) - 0.5;
      if (std::get<0>(contacts[0]) == 1) {
        pc = - pc;
      }
      assert(std::abs(std::get<2>(contacts[0])[0] + pc) < Eps);
      assert(std::abs(std::get<2>(contacts[0])[1] - 0) < Eps);
      assert(std::abs(std::get<2>(contacts[0])[2] - 0) < Eps);
    }
    {
      // 2 simplices. 1 contact. Vector interface.
      // Vertices
      std::vector<Point> v;
      v.push_back(Point{{0, 0, 0}});
      v.push_back(Point{{1, 0, 0}});
      v.push_back(Point{{0, 1, 0}});
      v.push_back(Point{{0, 0, 1}});
      v.push_back(Point{{0.5, 0, 0}});
      v.push_back(Point{{1.5, 0, 0}});
      v.push_back(Point{{0.5, 1, 0}});
      v.push_back(Point{{0.5, 0, 1}});
      // Vertex identifiers.
      std::vector<std::size_t> vi(primes.begin(), primes.begin() + v.size());
      // (Vertex identifier) simplices.
      std::vector<IdentifierSimplex> s;
      s.push_back(IdentifierSimplex{{vi[0], vi[1], vi[2], vi[3]}});
      s.push_back(IdentifierSimplex{{vi[4], vi[5], vi[6], vi[7]}});

      {
        std::vector<Point> velocities(v.size(), ext::filled_array<Point>(1));
        const double maximumRelativePenetration = 0.1;
        std::vector<Contact> contacts;
        contact::pinballContact(v, vi, velocities, s, maximumRelativePenetration,
                                std::back_inserter(contacts));
        assert(contacts.size() == 1);
        assert(std::get<0>(contacts[0]) +
               std::get<1>(contacts[0]) == 1);
        double pc = 0.5 * std::sqrt(3.) - 0.5;
        if (std::get<0>(contacts[0]) == 1) {
          pc = - pc;
        }
        assert(std::abs(std::get<2>(contacts[0])[0] + pc) < Eps);
        assert(std::abs(std::get<2>(contacts[0])[1] - 0) < Eps);
        assert(std::abs(std::get<2>(contacts[0])[2] - 0) < Eps);
      }
      {
        // Velocities = 1.
        std::vector<Point> velocities(v.size(), Point{{1., 1., 1.}});
        std::vector<double> masses(s.size(), 1.);
        const double maximumRelativePenetration = 0.5;
        std::vector<Force> forces;
        contact::PinballRestoringForces<N>
        pinballRestoringForces(maximumRelativePenetration);
        pinballRestoringForces(v, vi, velocities, s, masses,
                               std::back_inserter(forces));
        assert(forces.size() == s.size());
        for (std::size_t i = 0; i != forces.size(); ++i) {
          assert(std::get<0>(forces[i]) == i);
        }
        std::cout << "3-D contact\n";
        for (std::size_t i = 0; i != forces.size(); ++i) {
          std::cout << std::get<0>(forces[i]) << ' '
                    << std::get<1>(forces[i]) << '\n';
        }
      }
    }
  }
  return 0;
}
