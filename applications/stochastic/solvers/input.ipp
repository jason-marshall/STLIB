// -*- C++ -*-

#ifndef __input_ipp__
#error This file is an implementation detail.
#endif

/* 
   Common input for all solvers:

   <are printing information>
   <number of species>
   <number of reactions>
   <list of initial amounts>
   <packed reactions>
   <list of propensity factors>
   <number of species to record>
   <list of species to record>
   <number of reactions to record>
   <list of reactions to record>
   <maximum allowed steps>
   <number of solver parameters>
   <list of solver parameters>
   <starting time>

   Each term in brackets occupies a single line.

   A value of zero indicates there is no limit on the maximum allowed steps.
   (More precisely, the limit is std::numeric_limits<std::size_t>::max().)
*/

// Use 16 bits of precision for floating point types.
std::size_t defaultPrecision = std::cout.precision();
std::cout.precision(16);

ads::ParseOptionsArguments parser(argc, argv);

// Program name.
programName = parser.getProgramName();

// There should be no arguments.
if (parser.getNumberOfArguments() != 0) {
  std::cerr << "Bad arguments.\n"
            << "You gave the arguments:\n";
  parser.printArguments(std::cerr);
  exitOnError();
}
  
// Performance information.
const bool arePrintingPerformance = parser.getOption("p") || 
               parser.getOption("performance");

// If we are printing a Python dictionary of information.
bool arePrintingInformation;
std::cin >> arePrintingInformation;

//
// Read the model.
//

// The number of species and reactions.
std::size_t numberOfSpecies = 0, numberOfReactions = 0;
std::cin >> numberOfSpecies >> numberOfReactions;

// The initial populations.
std::vector<double> initialPopulations(numberOfSpecies);
for (std::size_t i = 0; i != numberOfSpecies; ++i) {
  std::cin >> initialPopulations[i];
}

// The reactions.
ReactionSet reactions;
stochastic::readReactantsAndProductsAscii(std::cin, numberOfReactions,
                                          &reactions);

// The propensity factors for the reactions.
{
  double rateConstant;
  for (std::size_t i = 0; i != numberOfReactions; ++i) {
    std::cin >> rateConstant;
    reactions.setRateConstant(i, rateConstant);
  }
}

// The recorded species.
std::vector<std::size_t> recordedSpecies;
std::cin >> recordedSpecies;

// The recorded reactions.
std::vector<std::size_t> recordedReactions;
std::cin >> recordedReactions;

//
// Read the simulation parameters.
//

// Maximum allowed steps.
double maximumAllowedSteps;
std::cin >> maximumAllowedSteps;
if (maximumAllowedSteps == 0) {
  maximumAllowedSteps = std::numeric_limits<double>::max();
}

// The solver parameters.
std::vector<double> solverParameters;
std::cin >> solverParameters;

double startTime;
std::cin >> startTime;
