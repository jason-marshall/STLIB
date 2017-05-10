// -*- C++ -*-

/*!
  \file tauLeaping.cc

  Concurrent tau-leaping.

  CONTINUE.
*/

#include "stlib/stochastic/tauLeapingConcurrent.h"

#include "stlib/ads/timer.h"
#include "stlib/ads/utility.h"

#include "numerical/partition.h"
#include "numerical/random/UniformRandom.h"
#include "numerical/random/Poisson.h"

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <limits>
#include <vector>

#include <cassert>

using namespace stlib;

void
exitOnError();

//
// Global variables.
//

static std::string programName;

int
main(int argc, char* argv[]) {
   //
   // Types.
   //

   typedef double Number;
   typedef numerical::UniformRandom2<Number> UniformRandom;
   typedef numerical::Poisson<Number, numerical::UniformRandom2> Poisson;

   //
   // Initialize MPI and get the MPI arguments.
   //

   MPI::Init(argc, argv);
   const int commWorldRank = MPI::COMM_WORLD.Get_rank();
   const int commWorldSize = MPI::COMM_WORLD.Get_size();
   MPI::Intracomm comm = MPI::COMM_WORLD;

   // Describe the communication types.
   enum {TagSize, TagMessage};

   ads::ParseOptionsArguments parser(argc, argv);

   // Program name.
   programName = parser.getProgramName();

   // If they did not specify the reactions, input state and output state files.
   if (parser.getNumberOfArguments() != 3) {
      std::cerr << "Bad number of required arguments.\n"
                << "You gave the arguments:\n";
      parser.printArguments(std::cerr);
      exitOnError();
   }

   //
   // Parse the options.
   //

   // The allowed error in the tau-leaping method.
   Number epsilon = 0;
   parser.getOption("epsilon", &epsilon);
   if (epsilon <= 0) {
      std::cerr << "Error: Bad value for epsilon.\n";
      exitOnError();
   }

   // Starting time
   Number startTime = 0;
   parser.getOption("startTime", &startTime);

   // Ending time
   Number endTime = std::numeric_limits<Number>::max();
   parser.getOption("endTime", &endTime);

   // Number of steps
   int numberOfSteps = 0;
   parser.getOption("steps", &numberOfSteps);

   if (endTime == std::numeric_limits<Number>::max() && numberOfSteps == 0) {
      std::cerr << "Error: You must specify either the ending time or the\n"
                << "number of steps.\n";
      exitOnError();
   }

   // Seed for the random number generator.
   int seed = 1;
   parser.getOption("seed", &seed);
   // Add the rank so the seeds for different processes are different.
   seed += commWorldRank;
   if (seed == 0) {
      std::cerr << "Error: The seed must be nonzero.\n";
      exitOnError();
   }

   // Check that we parsed all of the options.
   if (! parser.areOptionsEmpty()) {
      std::cerr << "Error.  Unmatched options:\n";
      parser.printOptions(std::cerr);
      exitOnError();
   }

   //
   // Read in the reactions and the initial state.
   //

   // Construct the state class.
   stochastic::State<Number> state;

   // Read the reactions.
   int numberOfReactions;
   {
      std::ifstream in(parser.getArgument().c_str());
      // Get the number of reactions.
      in >> numberOfReactions;
      // Determine our portion of the reactions.
      int begin, end;
      numerical::partitionRange(numberOfReactions, commWorldSize, commWorldRank,
                                &begin, &end);
      // Skip reactions until we reach our portion.
      stochastic::State<Number>::Reaction reaction;
      for (int i = 0; i != begin; ++i) {
         in >> reaction;
      }
      // Read our portion.
      for (int i = begin; i != end; ++i) {
         in >> reaction;
         state.insertReaction(reaction);
      }
   }

   // Read the initial populations.
   {
      std::ifstream in(parser.getArgument().c_str());
      stochastic::readPopulationsAscii(in, &state);
   }

   // Check the validity of the initial state.
   if (! isValid(state)) {
      std::cerr << "Error: The initial state of the simulation is not valid.\n";
      exitOnError();
   }

   //
   // Run the simulation.
   //

   int simulationSteps = 0;
   ads::Timer timer;
   // Use a barrier to make the timings more consistent across processes.
   comm.Barrier();
   timer.tic();

   // If they specified the ending time.
   if (endTime != std::numeric_limits<Number>::max()) {
      simulationSteps =
         stochastic::computeTauLeapingConcurrentSsa<Poisson>
         (comm, &state, epsilon, endTime, seed);
   }
   // Otherwise they specified the number of steps.
   else {
      stochastic::computeTauLeapingConcurrentSsa<Poisson>
      (comm, &state, epsilon, numberOfSteps, seed);
      simulationSteps = numberOfSteps;
   }

   ads::Timer::number_type elapsedTime = timer.toc();


   // Write the output state.
   if (commWorldRank == 0) {
      std::ofstream out(parser.getArgument().c_str());
      out << state.getPopulations();
   }
   else {
      // Since this is not the root processor, we don't use the final argument.  Here we access
      // it so we can check that we parsed all of the arguments.
      parser.getArgument().c_str();
   }

   // There should be no more arguments.
   assert(parser.areArgumentsEmpty());

   //
   // Print information.
   //
   // Collect the information into a string.
   std::ostringstream message;
   message << "In processor " << commWorldRank << ":\n";
   message << "  There are " << state.getNumberOfReactions() << " reactions.\n"
           << "  Simulation time = " << elapsedTime << "\n";
   // Send the size of the string and the string data to the rank 0
   // processor.
   std::string messageString(message.str());
   int messageSize = messageString.size();
   MPI::COMM_WORLD.Send(&messageSize, 1, MPI::INT, 0, TagSize);
   MPI::COMM_WORLD.Send(messageString.data(), messageSize, MPI::CHAR,
                        0, TagMessage);

   //
   // Collect information from each processor and print it.
   //
   if (commWorldRank == 0) {
      // Print a general message.
      std::cout
            << "The simulation took " << simulationSteps << " steps.\n"
            << "The simulation time interval is [" << startTime
            << " .. " << state.getTime() << "].\n"
            << "There are a total of " << numberOfReactions << " reactions\n\n";

      // Print a message from each processor.
      int messageSize;
      char* messageData;
      for (int r = 0; r != commWorldSize; ++r) {
         // Receive the message size in characters.
         MPI::COMM_WORLD.Recv(&messageSize, 1, MPI::INT, r, TagSize);

         // Allocate memory for the message.
         messageData = new char[messageSize];

         // Receive the message data.
         MPI::COMM_WORLD.Recv(messageData, messageSize, MPI::CHAR, r, TagMessage);

         // Make a string from the data.
         std::string message(messageData, messageSize);

         // Free the memory for the message data.
         delete[] messageData;

         // Print the message.
         std::cout << message;
      }
   }

   MPI::Finalize();

   return 0;
}



// Print usage information and exit.
void
exitOnError() {
   if (MPI::COMM_WORLD.Get_rank() == 0) {
      std::cerr
            << "Bad arguments.  Usage:\n"
            << "mpirun -np n tauLeaping.exe [-epsilon=e] [-startTime=s] [-endTime=e]\n"
            << "  [-steps=s] [-seed=s] reactions inputState outputState\n"
            << "- n is the number of processors.\n"
            //CONTINUE
            << "Exiting...\n";
   }
   MPI::Finalize();
   exit(1);
}
