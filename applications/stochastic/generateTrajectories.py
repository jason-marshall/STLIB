"""Generate trajectories. Read a low-level description of the model and 
simulation parameters from stdin. For each trajectory write the populations,
reaction counts, and random number generator state to stdout."""

import sys
import string

from state.simulationMethods import simulationMethods, simulationOptions,\
    getModule
from modules import arrays
from modules import mt19937

if __name__ == '__main__':
    #
    # Read data from stdin.
    #

    # The model.

    # initialAmounts (populations)
    data = map(int, sys.stdin.readline().rstrip().split())
    # Deduce the number of species from the initial populations.
    numberOfSpecies = len(data)
    initialAmounts = arrays.ArrayInt(numberOfSpecies)
    for n in range(numberOfSpecies):
        initialAmounts[n] = data[n]

    # packedReactions
    data = map(int, sys.stdin.readline().rstrip().split())
    packedReactions = arrays.ArrayInt(len(data))
    for n in range(len(data)):
        packedReactions[n] = data[n]

    # propensityFactors
    data = map(float, sys.stdin.readline().rstrip().split())
    # Deduce the number of reactions.
    numberOfReactions = len(data)
    propensityFactors = arrays.ArrayDouble(numberOfReactions)
    for n in range(numberOfReactions):
        propensityFactors[n] = data[n]

    # RNG state.

    # mt19937State
    data = map(long, sys.stdin.readline().rstrip().split())
    mt19937StateSize = 624
    assert len(data) == mt19937StateSize
    mt19937State = arrays.ArrayUnsigned(mt19937StateSize)
    for n in range(mt19937StateSize):
        mt19937State[n] = data[n]

    # Simulation parameters.

    # method is an integer index.
    method = int(sys.stdin.readline())

    # options is an integer index.
    options = int(sys.stdin.readline())

    # startTime
    startTime = float(sys.stdin.readline())

    # maximumAllowedSteps
    maximumAllowedSteps = int(sys.stdin.readline())

    # frameTimes
    data = map(float, sys.stdin.readline().rstrip().split())
    # Deduce the number of frames.
    numberOfFrames = len(data)
    frameTimes = arrays.ArrayDouble(numberOfFrames)
    for n in range(numberOfFrames):
        frameTimes[n] = data[n]
        
    # numberOfTrajectories
    numberOfTrajectories = int(sys.stdin.readline())

    #
    # Build data structures from the input data.
    #

    # The solver module.
    module = getModule(method, options)
    # Construct the solver.
    solver = module.newSolver(numberOfSpecies, numberOfReactions,
                              packedReactions, propensityFactors)
    # Set the random number generator state.
    module.setMt19937State(solver, mt19937State)

    framePopulations = arrays.ArrayInt(numberOfFrames * numberOfSpecies)
    frameReactionCounts = arrays.ArraySizeType(numberOfFrames * 
                                               numberOfReactions)

    for n in range(numberOfTrajectories):
        module.generateTrajectory(solver, initialAmounts,
                                  startTime, maximumAllowedSteps,
                                  numberOfFrames, frameTimes,
                                  framePopulations, frameReactionCounts)
        # Print the populations.
        reprList = [repr(framePopulations[n]) for n in 
                    range(numberOfFrames * numberOfSpecies)]
        sys.stdout.write('%s\n' % string.join(reprList, ' '))
        # Print the reaction counts.
        reprList = [repr(frameReactionCounts[n]) for n in 
                    range(numberOfFrames * numberOfReactions)]
        sys.stdout.write('%s\n' % string.join(reprList, ' '))
        # Print the RNG state.
        module.getMt19937State(solver, mt19937State)
        reprList = [repr(mt19937State[n]) for n in range(mt19937StateSize)]
        sys.stdout.write('%s\n' % string.join(reprList, ' '))
        sys.stdout.flush()
