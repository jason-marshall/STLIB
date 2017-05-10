// -*- C++ -*-

/*!
  \file proxyBall.ipp
*/

#if !defined(__proxyBall_ipp__)
#error This file is an implementation detail.
#endif

#include "contact/ProxyBallContactConcurrent.h"
#include "geom/mesh/iss/file_io.h"
#include "geom/mesh/iss/set.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer/Timer.h"
#include "concurrent/partition/rcb.h"

#include <fstream>
#include <limits>

#include <cassert>

namespace {

// Global variables.

//! The program name.
std::string programName;

// Local functions.

//! Exit with an error message and usage information.
void
exitOnError() {
   int commWorldRank;
   commWorldRank = MPI::COMM_WORLD.Get_rank();

   if (commWorldRank == 0) {
      std::cerr
            << "Bad arguments.  Usage:\n"
            << "mpirun -np n proxyBallN.exe mesh [forces] [-n=numberOfTimes] [-s=sleepTime] [-p]\n"
            << "[-m maximumRelativePenetration]\n"
            << "  n is the number of processors.\n"
            << "  mesh is the input file containing the simplicial mesh.\n"
            << "  forces is the optional output file for the forces.\n"
            << "  numberOfTimes is the number of times to perform the contact detection.\n"
            << "  s is the sleeping option.  If specified, each processor will\n"
            << "    sleep up to sleepTime seconds between communications.\n"
            << "    The sleep time must be specified in whole seconds.\n"
            << "  p turns on printing.\n"
            << "Exiting...\n";
   }
   MPI::Finalize();
   exit(1);
}

}

int
main(int argc, char* argv[]) {
   //
   // Types.
   //
   typedef contact::ProxyBallContactConcurrent<N> ProxyBallContact;
   typedef ProxyBallContact::Number Number;
   typedef ProxyBallContact::Point Point;
   typedef ProxyBallContact::Force Force;
   typedef geom::IndSimpSetIncAdj<N, N> Mesh;

   //
   // Initialize MPI and get the MPI arguments.
   //

   MPI::Init(argc, argv);
   const std::size_t commWorldRank = MPI::COMM_WORLD.Get_rank();
   const std::size_t commWorldSize = MPI::COMM_WORLD.Get_size();

   //
   // Parse the program options and arguments.
   //

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the optional command line arguments
   //

   // Determine the number of times to perform the contact detection.
   std::size_t numberOfTimes = 1;
   parser.getOption("n", &numberOfTimes);
   // Check for bad input.
   if (numberOfTimes < 1) {
      std::cerr << "Bad number of times to perform the contact detection.";
      exitOnError();
   }

   // Get the sleep time.
   double maximumSleepTime = 0;
   parser.getOption("s", &maximumSleepTime);

   // By default, don't print.
   const bool arePrinting = parser.getOption("p");

   double maximumRelativePenetration = 0.1;
   parser.getOption("m", &maximumRelativePenetration);

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   // Seed the random number generator for sleep times.
   srand(commWorldRank);

   //
   // Parse the program arguments.
   //

   if (parser.getNumberOfArguments() < 1) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Read the mesh.
   Mesh mesh;
   {
      std::ifstream meshFile(parser.getArgument().c_str());
      if (! meshFile.good()) {
         std::cerr << "Could not open the mesh file.";
         exitOnError();
      }
      geom::readAscii(meshFile, &mesh);
   }

   // Determine the connected components.
   std::vector<std::size_t> globalComponents;
   std::size_t numberOfComponents =
      geom::labelComponents(mesh, &globalComponents);
   // Compute the element centroids.
   std::vector<Point> centroids(mesh.getSimplicesSize());
   {
      Mesh::Simplex simplex;
      for (std::size_t i = 0; i != mesh.getSimplicesSize(); ++i) {
         mesh.getSimplex(i, &simplex);
         geom::computeCentroid(simplex, &centroids[i]);
      }
   }
   if (MPI::COMM_WORLD.Get_rank() == 0) {
      std::cout << "Global mesh:\n"
                << "  components = " << numberOfComponents << '\n'
                << "  vertices = " << mesh.getVerticesSize() << '\n'
                << "  elements = " << mesh.getSimplicesSize() << '\n'
                << '\n';
   }

   //
   // Partition the mesh.
   //
   std::vector<std::size_t> elementIdentifiers(mesh.getSimplicesSize());
   for (std::size_t i = 0; i != elementIdentifiers.size(); ++i) {
      elementIdentifiers[i] = i;
   }
   std::vector<std::size_t*> idPartition(commWorldSize + 1);
   concurrent::rcb<N>(commWorldSize, &elementIdentifiers, &idPartition,
                      centroids);

   // Construct the contact functor.
   const std::size_t numberOfNodes = mesh.getVerticesSize();
   std::vector<double> nodeCoordinates(N * numberOfNodes);
   for (std::size_t i = 0; i != numberOfNodes; ++i) {
      for (std::size_t n = 0; n != N; ++n) {
         nodeCoordinates[i* N+n] = mesh.getVertex(i)[n];
      }
   }
   std::vector<std::size_t> nodeIdentifiers(numberOfNodes);
   for (std::size_t i = 0; i != nodeIdentifiers.size(); ++i) {
      nodeIdentifiers[i] = i;
   }
   const std::size_t numberOfElements =
      std::distance(idPartition[commWorldRank], idPartition[commWorldRank+1]);
   std::vector<std::size_t> identifierSimplices(numberOfElements *(N + 1));
   const std::size_t offset = std::distance(&elementIdentifiers[0],
                              idPartition[commWorldRank]);
   for (std::size_t i = 0; i != numberOfElements; ++i) {
      for (std::size_t n = 0; n != N + 1; ++n) {
         identifierSimplices[i*(N+1)+n] =
            mesh.getIndexedSimplex(elementIdentifiers[offset+i])[n];
      }
   }
   std::vector<std::size_t> components(numberOfElements);
   for (std::size_t i = 0; i != components.size(); ++i) {
      components[i] = globalComponents[elementIdentifiers[offset+i]];
   }
#if 0
   if (commWorldRank == 0) {
      std::cout << "numberOfNodes = " << numberOfNodes << '\n'
                << "nodeCoordinates = " << nodeCoordinates << '\n'
                << "nodeIdentifiers = " << nodeIdentifiers << '\n'
                << "numberOfElements = " << numberOfElements << '\n'
                << "identifierSimplices = " << identifierSimplices << '\n'
                << "offset = " << offset << '\n'
                << "components = " << components << '\n';
   }
#endif
   ProxyBallContact contact(MPI::COMM_WORLD, numberOfComponents,
                            numberOfNodes, &nodeCoordinates[0],
                            &nodeIdentifiers[0], numberOfElements,
                            &identifierSimplices[0], &components[0],
                            maximumRelativePenetration);

   // Set the velocities to the negative of the node positions.
   std::vector<double> velocityCoordinates(nodeCoordinates);
   for (std::size_t i = 0; i != velocityCoordinates.size(); ++i) {
      velocityCoordinates[i] *= -1.;
   }
   const std::vector<double> masses(components.size(), 1);
   std::vector<Force> elementForces;

   MPI::COMM_WORLD.Barrier();
   // Perform the contact detection the specified number of times.
   ads::Timer timer;
   timer.tic();
   double stableTimeStep = 0;
   for (; numberOfTimes; --numberOfTimes) {
      elementForces.clear();
      stableTimeStep = contact(&nodeCoordinates[0],
                               &velocityCoordinates[0],
                               &masses[0],
                               &components[0],
                               std::back_inserter(elementForces));
      // Wait for a random amount of time.
      sleep(unsigned(rand() * maximumSleepTime / RAND_MAX));
   }
   const double elapsedTime = timer.toc();

   if (commWorldRank == 0) {
      std::cout << "Time to perform contact detection = " << elapsedTime << ".\n";
   }

   // Combine forces.
   const Point zero = ext::filled_array<Point>(0);
   std::vector<Point> forces(numberOfElements, zero);
   for (std::size_t i = 0; i != elementForces.size(); ++i) {
      forces[std::tr1::get<0>(elementForces[i])] +=
         std::tr1::get<1>(elementForces[i]);
   }

   // Describe the communication types.
   enum {TagSize, TagMessage};

   if (arePrinting) {
      //
      // Print information.
      //
      // Collect the information into a string.
      std::ostringstream message;
      message << "In processor " << commWorldRank << "\n"
              << numberOfElements << " elements.\n"
              << "stable time step = " << stableTimeStep << "\n"
              << elementForces.size() << " forces.\n";
      message << "Elements:\n";
      for (std::size_t i = 0; i != numberOfElements; ++i) {
         message << elementIdentifiers[offset + i] << ' ';
      }
      message << '\n';

      std::size_t count = 0;
      for (std::size_t i = 0; i != forces.size() && count < 10; ++i) {
         if (forces[i] != zero) {
            message << "index = " << i
                    << ", force =  " << forces[i] << "\n";
            ++count;
         }
      }
      message << "\n";

      // Send the size of the string and the string data to the rank 0
      // processor.
      std::string messageString(message.str());
      std::size_t messageSize = messageString.size();
      MPI::Request sizeRequest =
         MPI::COMM_WORLD.Isend(&messageSize, sizeof(std::size_t), MPI::CHAR, 0,
                               TagSize);
      MPI::Request messageRequest =
         MPI::COMM_WORLD.Isend(&messageString[0], messageSize, MPI::CHAR, 0,
                               TagMessage);

      //
      // Collect information from each processor and print it.
      //
      if (arePrinting && commWorldRank == 0) {
         std::size_t receiveSize;
         char* messageData;
         // For each processor.
         for (std::size_t r = 0; r != commWorldSize; ++r) {
            // Receive the message size in characters.
            MPI::COMM_WORLD.Recv(&receiveSize, sizeof(std::size_t), MPI::CHAR, r,
                                 TagSize);
            // Allocate memory for the message.
            messageData = new char[receiveSize];
            // Receive the message data.
            MPI::COMM_WORLD.Recv(messageData, receiveSize, MPI::CHAR, r,
                                 TagMessage);
            // Make a string from the data.
            std::string message(messageData, receiveSize);
            // Free the memory for the message data.
            delete[] messageData;
            // Print the message.
            std::cout << message;
         }
      }

      sizeRequest.Wait();
      messageRequest.Wait();
   }

   // If an output file for the forces was specified.
   if (parser.getNumberOfArguments() != 0) {
      // Record the nonzero forces and the associated element identifiers.
      std::vector<Force> nonzeroForces;
      Force f;
      for (std::size_t i = 0; i != forces.size(); ++i) {
         std::tr1::get<0>(f) = elementIdentifiers[offset + i];
         std::tr1::get<1>(f) = forces[i];
         nonzeroForces.push_back(f);
      }

      // Send the nonzero forces to the rank 0 processor.
      std::size_t messageSize = sizeof(Force) * nonzeroForces.size();
      MPI::Request sizeRequest =
         MPI::COMM_WORLD.Isend(&messageSize, sizeof(std::size_t), MPI::CHAR, 0,
                               TagSize);
      MPI::Request messageRequest =
         MPI::COMM_WORLD.Isend(&nonzeroForces[0], messageSize, MPI::CHAR, 0,
                               TagMessage);

      // Collect the nonzero forces from each processor.
      if (commWorldRank == 0) {
         std::vector<Point> globalForces(mesh.getSimplicesSize(), zero);
         std::size_t receiveSize;
         Force* receivedForces;
         // For each processor.
         for (std::size_t r = 0; r != commWorldSize; ++r) {
            // Receive the message size in characters.
            MPI::COMM_WORLD.Recv(&receiveSize, sizeof(std::size_t), MPI::CHAR, r,
                                 TagSize);
            // Allocate memory for the forces.
            const std::size_t numberOfForces = receiveSize / sizeof(Force);
            receivedForces = new Force[numberOfForces];
            // Receive the forces.
            MPI::COMM_WORLD.Recv(receivedForces, receiveSize, MPI::CHAR, r,
                                 TagMessage);
            // Accumulate the forces.
            for (std::size_t i = 0; i != numberOfForces; ++i) {
               globalForces[std::tr1::get<0>(receivedForces[i])] =
                  std::tr1::get<1>(receivedForces[i]);
            }
            // Free the memory for the message data.
            delete[] receivedForces;
         }
         // Write the forces to the output file.
         std::ofstream file(parser.getArgument().c_str());
         for (std::size_t i = 0; i != globalForces.size(); ++i) {
            if (globalForces[i] != zero) {
               file << i << ' ' << globalForces[i] << '\n';
            }
         }
         // There should be no more arguments.
         assert(parser.areArgumentsEmpty());
      }

      sizeRequest.Wait();
      messageRequest.Wait();
   }

   MPI::Finalize();

   return 0;
}
