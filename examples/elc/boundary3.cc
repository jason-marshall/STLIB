// -*- C++ -*-

/*!
  \file boundary3.cc

  Eulerian/Lagrangian coupling for the boundary of a solid in 3-D.

  The processors are divided into Eulerian processors and Lagrangian
  processors.  All processors read a mesh (an indexed triangle face set
  in 3-D).  The Eulerian processors put a bounding box around the mesh
  to determine their domains.  The Lagrangian processors partition the mesh
  and use it as the boundary.
*/

#include "partition.h"

#include "stlib/elc/EulerianCommBoundary.h"
#include "stlib/elc/LagrangianComm.h"

#include "stlib/geom/mesh/iss/file_io.h"

#include "stlib/ads/utility/ParseOptionsArguments.h"
#include "numerical/partition.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
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
#ifdef ELC_USE_CPP_INTERFACE
   commWorldRank = MPI::COMM_WORLD.Get_rank();
#else
   MPI_Comm_rank(MPI_COMM_WORLD, &commWorldRank);
#endif

   if (commWorldRank == 0) {
      std::cerr
            << "Bad arguments.  Usage:\n"
            << "mpirun -np n boundary3.exe eul_x eul_y eul_z mesh\n"
            << "       [-n numberOfTimes] [-s sleepTime] [-p] [-c]\n"
            << "  n is the number of processors.\n"
            << "  eul_* is the number of Eulerian processors "
            << "in the * direction.\n"
            << "  mesh is file containing the triangle mesh.\n"
            << "  numberOfTimes is the number of times to perform the exchange.\n"
            << "  s is the sleeping option.  If specified, each processor will\n"
            << "    sleep up to sleepTime seconds between communications.\n"
            << "    The sleep time must be specified in whole seconds.\n"
            << "  p turns on printing.\n"
            << "  c turns on checking of the pressure values.\n"
            << "Exiting...\n";
   }
#ifdef ELC_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif
   exit(1);
}

}

int
main(int argc, char* argv[]) {
   //
   // Types.
   //

   typedef double Number;
   typedef geom::BBox<3> BBox;
   typedef BBox::Point Point;

   //
   // Initialize MPI and get the MPI arguments.
   //

#ifdef ELC_USE_CPP_INTERFACE
   MPI::Init(argc, argv);
   const int commWorldRank = MPI::COMM_WORLD.Get_rank();
   const int commWorldSize = MPI::COMM_WORLD.Get_size();
#else
   MPI_Init(&argc, &argv);
   int commWorldRank, commWorldSize;
   MPI_Comm_rank(MPI_COMM_WORLD, &commWorldRank);
   MPI_Comm_size(MPI_COMM_WORLD, &commWorldSize);
#endif

   //
   // Parse the program options and arguments.
   //

   ads::ParseOptionsArguments parser(argc, argv);

   programName = parser.getProgramName();

   //
   // Get the optional command line arguments
   //

   // Determine the number of times to perform the communication.
   std::size_t numberOfTimes = 1;
   parser.getOption("n", &numberOfTimes);
   // Check for bad input.
   if (numberOfTimes < 1) {
      std::cerr << "Bad number of times to perform the communication.";
      exitOnError();
   }

   // Get the sleep time.
   double maximumSleepTime = 0;
   parser.getOption("s", &maximumSleepTime);

   // By defalt, don't print.
   const bool arePrinting = parser.getOption("p");
   // By defalt, don't check the pressure.
   const bool areCheckingPressure = parser.getOption("c");
   // CONTINUE: Using this option will lead to a failed assertion.
   // By defalt, use the identifiers.
   const bool areUsingIdentifiers = ! parser.getOption("i");

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

   if (parser.getNumberOfArguments() != 4) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   // Get the number of Eulerian processors in each direction.
   std::array<int, 3> extents = {{}};
   for (std::size_t i = 0; i != extents.size(); ++i) {
      parser.getArgument(&extents[i]);
      if (extents[i] <= 0) {
         std::cerr << "Bad extents.";
         exitOnError();
      }
   }

   //
   // Get the mesh file.  We open it here because the Eulerian processors use
   // the bounding box of the mesh to determine their domains.
   //
   std::ifstream meshFile(parser.getArgument().c_str());
   if (! meshFile.good()) {
      std::cerr << "Could not open the mesh file.";
      exitOnError();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   // Describe the communication types.
   enum {TagSize, TagMessage};

   //
   // Divide the processors into two communicators.
   //

   // Compute the number of Eulerian and Lagrangian processors.
   const int eulerianRoot = 0;
   const int numEulerian = extents[0] * extents[1] * extents[2];
   const int numLagrangian = commWorldSize - numEulerian;
   assert(numLagrangian > 0);
   const int lagrangianRoot = numEulerian;

   // Choose Eulerian or Lagrangian
   enum Group {Eulerian, Lagrangian};
   Group group;
   if (commWorldRank < numEulerian) {
      group = Eulerian;
   }
   else {
      group = Lagrangian;
   }

#ifdef ELC_USE_CPP_INTERFACE
   MPI::Intracomm comm = MPI::COMM_WORLD.Split(group == Eulerian, 0);
   const int size = comm.Get_size();
   const int rank = comm.Get_rank();
#else
   MPI_Comm comm;
   MPI_Comm_split(MPI_COMM_WORLD, group == Eulerian, 0, &comm);
   int size, rank;
   MPI_Comm_size(comm, &size);
   MPI_Comm_rank(comm, &rank);
#endif

   //
   // Read the global boundary.
   // This is done here because the Eulerian processors need the mesh in
   // order to determine an appropriate domain.
   //

   // Define some required types.
   typedef geom::IndSimpSet<3, 2> Mesh;
   typedef Mesh::Vertex Vertex;
   static_assert(sizeof(Vertex) == sizeof(Point),
                 "The vertex type must be the same as the point type.");
   //typedef Mesh::IndexedSimplex IndexedFace;
   typedef std::array < int, 2 + 1 > IndexedFace;

   // The mesh holds the positions and connectivities.
   // Read the mesh.
   Mesh mesh;
   geom::readAscii(meshFile, &mesh);

   if (group == Eulerian) {
      //
      // Compute the x,y,z ranks of this processor from the overall rank.
      //

      std::array<int, 3> directionRanks;
      directionRanks[2] = rank / (extents[0] * extents[1]);
      int r = rank % (extents[0] * extents[1]);
      directionRanks[1] = r / extents[0];
      directionRanks[0] = r % extents[0];

      //
      // Compute the domain of this processor.
      //

      // First make a bounding box around the mesh.
      BBox meshDomain;
      meshDomain.bound(mesh.getVerticesBeginning(), mesh.getVerticesEnd());
      Point lo(meshDomain.getLowerCorner());
      Point hi(meshDomain.getUpperCorner());
      for (std::size_t n = 0; n != 3; ++n) {
         // Check for degeneracies in the mesh bounding box.
         if (lo[n] == hi[n]) {
            lo[n] -= 1.0;
            hi[n] += 1.0;
         }
         // Expand the bounding box by 10%.
         double x = 0.1 * (hi[n] - lo[n]);
         lo[n] -= x;
         hi[n] += x;
      }

      const Point dr = ext::convert_array<Number>(directionRanks);
      const Point ex = ext::convert_array<Number>(extents);
      // The domain.
      BBox domain(lo + ((hi - lo) * dr) / ex,
      lo + ((hi - lo) *(dr + 1.0)) / ex);
      // The region of interest. Expand the domain by 10%.
      BBox region(domain.getLowerCorner() -
      0.1 *(domain.getUpperCorner() - domain.getLowerCorner()),
      domain.getUpperCorner() +
      0.1 *(domain.getUpperCorner() - domain.getLowerCorner()));

      // Construct the Eulerian communicator.
      // We use it through a pointer to the base class in order to exercise
      // the virtual function interface.
      elc::EulerianComm<3, Number>* eulerianCommunicator;
#ifdef ELC_USE_CPP_INTERFACE
      eulerianCommunicator = new elc::EulerianCommBoundary<3, Number>
      (MPI::COMM_WORLD, comm, numLagrangian, lagrangianRoot,
      elc::LocalIndices);
#else
      eulerianCommunicator = new elc::EulerianCommBoundary<3, Number>
      (MPI_COMM_WORLD, comm, numLagrangian, lagrangianRoot,
      elc::LocalIndices);
#endif

      // Perform the exchange the specified number of times.
      for (; numberOfTimes; --numberOfTimes) {

         // Get the boundary.
         eulerianCommunicator->receiveMesh(region);
         eulerianCommunicator->waitForMesh();

         if (areCheckingPressure) {
            // Set the pressures.
            for (int i = 0; i != eulerianCommunicator->getNumberOfNodes(); ++i) {
               if (domain.isIn(eulerianCommunicator->getPositions()[i])) {
                  eulerianCommunicator->getPressures()[i] =
                  sum(eulerianCommunicator->getPositions()[i]) +
                  sum(eulerianCommunicator->getVelocities()[i]);
               }
            }
         }

         // Wait for a random amount of time.
         sleep(unsigned(rand() * maximumSleepTime / RAND_MAX));

         // Send the pressures.
         eulerianCommunicator->sendPressure();
         eulerianCommunicator->waitForPressure();
      }

      if (arePrinting) {
         //
         // Print information.
         //

         // Collect the information into a string.
         std::ostringstream message;
         message << "In Eulerian processor " << rank
         << " with domain " << domain
         << "\nregion = " << region << ":\n";
         message << eulerianCommunicator->getPressures().size() << " nodes.\n";
         for (int i = 0; i != eulerianCommunicator->getPressures().size() &&
         i < 10; ++i) {
            message << "pos =  " << eulerianCommunicator->getPositions()[i]
            << ", vel =  " << eulerianCommunicator->getVelocities()[i]
            << ", pres = " << eulerianCommunicator->getPressures()[i]
            << "\n";
         }
         message << '\n';

         // Send the size of the string and the string data to the rank 0
         // processor.
         std::string messageString(message.str());
         int messageSize = messageString.size();
#ifdef ELC_USE_CPP_INTERFACE
         MPI::COMM_WORLD.Send(&messageSize, 1, MPI::INT, 0, TagSize);
         MPI::COMM_WORLD.Send(messageString.data(), messageSize, MPI::CHAR,
         0, TagMessage);
#else
         MPI_Send(&messageSize, 1, MPI_INT, 0, TagSize, MPI_COMM_WORLD);
         MPI_Send(const_cast<char*>(messageString.data()), messageSize,
         MPI_CHAR, 0, TagMessage, MPI_COMM_WORLD);
#endif
      }
      delete eulerianCommunicator;
   }
   else { // source == Lagrangian
      //
      // Make the local portion of the boundary.
      //

      std::vector<int> identifiers;
      std::vector<Vertex> positions;
      std::vector<IndexedFace> connectivities;
      partitionMesh<3, double>(size, rank,
      mesh.getVertices(), mesh.getIndexedSimplices(),
      &identifiers, &positions, &connectivities);

      // Set the velocities to -0.5 times the positions.
      std::vector<Point> velocities(positions);
      for (std::size_t i = 0; i != velocities.size(); ++i) {
         velocities[i] *= -0.5;
      }

      std::vector<Number> pressures(identifiers.size());

      //
      // Construct the Lagrangian communicator.
      //

#ifdef ELC_USE_CPP_INTERFACE
      elc::LagrangianComm<3, Number>
      lagrangianCommunicator(MPI::COMM_WORLD, comm, numEulerian, eulerianRoot,
      elc::LocalIndices);
#else
      elc::LagrangianComm<3, Number>
      lagrangianCommunicator(MPI_COMM_WORLD, comm, numEulerian, eulerianRoot,
      elc::LocalIndices);
#endif

      // Perform the exchange the specified number of times.
      for (; numberOfTimes; --numberOfTimes) {
         //
         // Perform the point-to-point communication.
         //
         if (areUsingIdentifiers) {
            lagrangianCommunicator.sendMesh(identifiers.size(),
            &identifiers[0],
            &positions[0][0], &velocities[0][0],
            connectivities.size(),
            &connectivities[0][0]);
         }
         else {
            lagrangianCommunicator.sendMesh(identifiers.size(), 0,
            &positions[0][0], &velocities[0][0],
            connectivities.size(),
            &connectivities[0][0]);
         }
         lagrangianCommunicator.waitForMesh();

         // Wait for a random amount of time.
         sleep(unsigned(rand() * maximumSleepTime / RAND_MAX));

         lagrangianCommunicator.receivePressure(pressures.size(),
         &pressures[0]);
         lagrangianCommunicator.waitForPressure();
      }

      //
      // Check the pressures.
      //

      if (areCheckingPressure) {
         for (std::size_t i = 0; i != pressures.size(); ++i) {
            assert(std::abs(pressures[i] - (sum(positions[i]) + sum(velocities[i])))
            < 10 * std::numeric_limits<double>::epsilon());
         }
      }

      if (arePrinting) {
         //
         // Print information.
         //

         // Collect the information into a string.
         std::ostringstream message;
         message << "In Lagrangian processor " << rank << "\n";
         message << positions.size() << " nodes.\n";
         std::size_t numberKnown = 0;
         for (std::size_t i = 0; i != pressures.size(); ++i) {
            if (pressures[i] != std::numeric_limits<double>::max()) {
               ++numberKnown;
            }
         }
         message << numberKnown << " pressures are known.\n";
         for (std::size_t i = 0; i != positions.size() && i < 10; ++i) {
            message << "id = " << identifiers[i]
            << ", pos =  " << positions[i]
            << ", pres = " << pressures[i] << "\n";
         }
         message << "\n";

         // Send the size of the string and the string data to the rank 0
         // processor.
         std::string messageString(message.str());
         std::size_t messageSize = messageString.size();
#ifdef ELC_USE_CPP_INTERFACE
         MPI::COMM_WORLD.Send(&messageSize, 1, MPI::INT, 0, TagSize);
         MPI::COMM_WORLD.Send(&messageString[0], messageSize, MPI::CHAR,
         0, TagMessage);
#else
         MPI_Send(&messageSize, 1, MPI_INT, 0, TagSize, MPI_COMM_WORLD);
         MPI_Send(const_cast<char*>(&messageString[0]), messageSize,
         MPI_CHAR, 0, TagMessage, MPI_COMM_WORLD);
#endif
      }
   }

   //
   // Collect information from each processor and print it.
   //
   if (arePrinting && commWorldRank == 0) {
      int messageSize;
      char* messageData;
      // For each processor.
      for (int r = 0; r != commWorldSize; ++r) {
         // Receive the message size in characters.
#ifdef ELC_USE_CPP_INTERFACE
         MPI::COMM_WORLD.Recv(&messageSize, 1, MPI::INT, r, TagSize);
#else
         MPI_Status status;
         MPI_Recv(&messageSize, 1, MPI_INT, r, TagSize, MPI_COMM_WORLD, &status);
#endif

         // Allocate memory for the message.
         messageData = new char[messageSize];

         // Receive the message data.
#ifdef ELC_USE_CPP_INTERFACE
         MPI::COMM_WORLD.Recv(messageData, messageSize, MPI::CHAR, r,
         TagMessage);
#else
         MPI_Recv(messageData, messageSize, MPI_CHAR, r,
         TagMessage, MPI_COMM_WORLD, &status);
#endif

         // Make a string from the data.
         std::string message(messageData, messageSize);

         // Free the memory for the message data.
         delete[] messageData;

         // Print the message.
         std::cout << message;
      }
   }

#ifdef ELC_USE_CPP_INTERFACE
   MPI::Finalize();
#else
   MPI_Finalize();
#endif

   return 0;
}
