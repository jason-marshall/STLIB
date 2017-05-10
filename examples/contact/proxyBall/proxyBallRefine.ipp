// -*- C++ -*-

/*!
  \file proxyBallRefine.ipp
*/

#if !defined(__proxyBallRefine_ipp__)
#error This file is an implementation detail.
#endif

#include "contact/ProxyBallContactConcurrent.h"
#include "geom/mesh/iss/file_io.h"
#include "geom/mesh/iss/set.h"
#include "geom/mesh/simplicial/refine.h"
#include "geom/mesh/simplicial/build.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "stlib/ads/timer/Timer.h"
#include "concurrent/partition/rcb.h"

#include <iostream>
#include <fstream>
#include <limits>
#include <map>

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
            << "mpirun -np n proxyBallRefineN.exe mesh [-n=numberOfTimes] [-s=sleepTime] [-p]\n"
            << "[-m=maximumRelativePenetration]\n"
            << "  n is the number of processors.\n"
            << "  mesh is the input file containing the simplicial mesh.\n"
            << "  forces is the optional output file for the forces.\n"
            << "  numberOfTimes is the number of times to perform the contact detection.\n"
            << "  s is the sleeping option.  If specified, each processor will\n"
            << "    sleep up to sleepTime seconds between communications.\n"
            << "    The sleep time must be specified in whole seconds.\n"
            << "  m specifies the maximium allowed relative penetration for contact..\n"
            << "  p turns on printing.\n"
            << "Exiting...\n";
   }
   MPI::Finalize();
   exit(1);
}

}

void
refine(geom::SimpMeshRed<N, N>* mesh,
       std::map<std::size_t, std::size_t>* components,
       const std::size_t targetSize) {
   typedef geom::SimpMeshRed<N, N> SMR;
   typedef SMR::CellIterator CellIterator;
   typedef SMR::CellIteratorSet CellIteratorSet;

   // We don't use a surface manifold in performing refinement.
   geom::PointsOnManifold < N, N - 1, 1 > * manifold = 0;
   // This stores the cells that are split during a single recursive cell
   // splitting operation.
   std::vector<CellIterator> splitCells;
   std::size_t size = mesh->computeCellsSize();
   // Check the trivial case that the mesh is empty.
   if (size == 0) {
      return;
   }

   // Loop until we reach the target number of elements.
   while (size < targetSize) {
      // Make a set of the cell iterators.
      CellIteratorSet cells;
      for (CellIterator i = mesh->getCellsBeginning(); i != mesh->getCellsEnd();
            ++i) {
         cells.insert(i);
      }
      // Iterate until all the original cells have been split or we have reached
      // the target size.
      while (size < targetSize && ! cells.empty()) {
         const std::size_t component =
            (*components)[(*cells.begin())->getIdentifier()];
         // Recursively split a cell.
         const std::size_t splitCount = splitCell(mesh, manifold, *cells.begin(),
                                        std::back_inserter(splitCells));
         // The new cells are in the same component as the split cell.
         for (std::size_t i = 0; i != splitCount; ++i) {
            (*components)[size] = component;
            ++size;
         }
         // Remove the cells that were split from the set.
         for (std::vector<CellIterator>::const_iterator i = splitCells.begin();
               i != splitCells.end(); ++i) {
            cells.erase(*i);
         }
         splitCells.clear();
      }
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
   typedef geom::SimpMeshRed<N, N> SMR;

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

   // The mesh we will use for contact detection.
   Mesh mesh;
   // The number of components is calculated below.
   std::size_t numberOfComponents = 0;
   // The component of each element in the mesh.
   std::vector<std::size_t> components;
   {
      // Read the mesh.
      Mesh inputMesh;
      {
         std::ifstream meshFile(parser.getArgument().c_str());
         if (! meshFile.good()) {
            std::cerr << "Could not open the mesh file.";
            exitOnError();
         }
         geom::readAscii(meshFile, &inputMesh);
      }

      // Determine the connected components.
      std::vector<std::size_t> inputComponents;
      numberOfComponents = geom::labelComponents(inputMesh, &inputComponents);
      // Compute the element centroids.
      std::vector<Point> centroids(inputMesh.getSimplicesSize());
      {
         Mesh::Simplex simplex;
         for (std::size_t i = 0; i != inputMesh.getSimplicesSize(); ++i) {
            inputMesh.getSimplex(i, &simplex);
            geom::computeCentroid(simplex, &centroids[i]);
         }
      }
      if (MPI::COMM_WORLD.Get_rank() == 0) {
         std::cout << "Input mesh:\n"
                   << "  components = " << numberOfComponents << '\n'
                   << "  vertices = " << inputMesh.getVerticesSize() << '\n'
                   << "  elements = " << inputMesh.getSimplicesSize() << '\n'
                   << '\n';
      }

      //
      // Partition the mesh.
      //
      std::vector<std::size_t> elementIdentifiers(inputMesh.getSimplicesSize());
      for (std::size_t i = 0; i != elementIdentifiers.size(); ++i) {
         elementIdentifiers[i] = i;
      }
      std::vector<std::size_t*> idPartition(commWorldSize + 1);
      concurrent::rcb<N>(commWorldSize, &elementIdentifiers, &idPartition,
                         centroids);

      //
      // Build the local portion of the input mesh.
      //
      Mesh localMesh;
      std::vector<std::size_t> localComponents;
      {
         const std::size_t numberOfElements =
            std::distance(idPartition[commWorldRank], idPartition[commWorldRank+1]);
         const std::size_t offset = std::distance(&elementIdentifiers[0],
                                    idPartition[commWorldRank]);
         std::vector<std::size_t> localIdentifiers(numberOfElements);
         for (std::size_t i = 0; i != numberOfElements; ++i) {
            localIdentifiers[i] = elementIdentifiers[offset+i];
         }
         geom::buildFromSubsetSimplices(inputMesh, localIdentifiers.begin(),
                                        localIdentifiers.end(), &localMesh);
         localComponents.resize(numberOfElements);
         for (std::size_t i = 0; i != localComponents.size(); ++i) {
            localComponents[i] = inputComponents[elementIdentifiers[offset+i]];
         }
      }

      //
      // Refine the local portion to obtain the desired number of elements.
      //
      SMR refined(localMesh);
      std::map<std::size_t, std::size_t> componentMap;
      for (std::size_t i = 0; i != localComponents.size(); ++i) {
         componentMap[i] = localComponents[i];
      }
      refine(&refined, &componentMap, inputMesh.getSimplicesSize());

      //
      // Convert to a indexed simplex set.
      //
      geom::buildIndSimpSetFromSimpMeshRed(refined, &mesh);
      components.resize(mesh.getSimplicesSize());
      assert(componentMap.size() == components.size());
      for (std::map<std::size_t, std::size_t>::const_iterator
            i = componentMap.begin(); i != componentMap.end(); ++i) {
         components[i->first] = i->second;
      }
      for (std::size_t i = 0; i != components.size(); ++i) {
         assert(components[i] < numberOfComponents);
      }
   }

   //
   // Construct the contact functor.
   //
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
   const std::size_t numberOfElements = mesh.getSimplicesSize();
   std::vector<std::size_t> identifierSimplices(numberOfElements *(N + 1));
   for (std::size_t i = 0; i != numberOfElements; ++i) {
      for (std::size_t n = 0; n != N + 1; ++n) {
         identifierSimplices[i*(N+1)+n] = mesh.getIndexedSimplex(i)[n];
      }
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
   for (std::size_t n = 0; n != numberOfTimes; ++n) {
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

   // Collect the number of computed element forces to the root processor.
   const std::size_t size = elementForces.size();
   std::vector<std::size_t> sizes(commWorldSize, 0);
   MPI::COMM_WORLD.Gather(&size, sizeof(std::size_t), MPI::BYTE,
                          &sizes[0], sizeof(std::size_t), MPI::BYTE, 0);

   if (commWorldRank == 0) {
      std::cout << "Number of computed forces in a time step:\n"
                << "Total = " << std::accumulate(sizes.begin(), sizes.end(), 0)
                << ", Minimum = " << *std::min_element(sizes.begin(), sizes.end())
                << ", Maximum = " << *std::max_element(sizes.begin(), sizes.end())
                << ".\n"
                << "Average time to perform contact detection = "
                << elapsedTime / numberOfTimes << ".\n";
   }

   MPI::Finalize();

   return 0;
}
