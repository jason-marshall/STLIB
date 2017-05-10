"""Implements the TimeSeriesUniform class."""

import numpy

class TimeSeriesUniform:
    r"""
    >>> from TimeSeriesUniform import TimeSeriesUniform
    >>> from Model import Model
    >>> from Species import Species
    >>> from SpeciesReference import SpeciesReference
    >>> from Reaction import Reaction
    >>> from FirstReaction import FirstReaction

    >>> m = Model(0)
    >>> m.species['s1'] = Species(0)
    >>> m.reactions['r1'] = Reaction(m, [], [SpeciesReference('s1', 1)], '1')
    >>> solver = FirstReaction(m, 1000)
    >>> simulation = TimeSeriesUniform(solver, ['s1'], ['r1'], range(11))
    >>> simulation.generateTrajectory()

    The results of the test depend on the random number seed.
    >>> print(simulation.populations) # doctest: +ELLIPSIS
    [[...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]]
    >>> print(simulation.reactionCounts) # doctest: +ELLIPSIS
    [[...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]
     [...]]
    """
    def __init__(self, solver, recordedSpecies, recordedReactions, frameTimes):
        self.solver = solver
        # List of species identifiers.
        self.recordedSpecies = recordedSpecies
        # List of reaction identifiers.
        self.recordedReactions = recordedReactions
        # The frame times at which to record the state.
        self.frameTimes = frameTimes
        # 2-D array of the recorded species populations. The row is the frame
        # index; the column is the recorded species index.
        self.populations = numpy.zeros((len(frameTimes), len(recordedSpecies)))
        # 2-D array of the recorded species populations. The row is the frame
        # index; the column is the recorded species index.
        self.reactionCounts = numpy.zeros((len(frameTimes),
                                           len(recordedReactions)))

    def initialize(self):
        # Initialize the solver.
        self.solver.initialize()
        # Initialize the population and reaction counts arrays.
        self.populations.fill(0)
        self.reactionCounts.fill(0)

    def generateTrajectory(self):
        self.initialize()
        m = self.solver.model
        for i in range(len(self.frameTimes)):
            # Advance the simulation.
            self.solver.simulate(self.frameTimes[i])
            # Record the state.
            for j in range(len(self.recordedSpecies)):
                id = self.recordedSpecies[j]
                self.populations[i, j] = m.species[id].amount
            for j in range(len(self.recordedReactions)):
                id = self.recordedReactions[j]
                self.reactionCounts[i, j] = m.reactions[id].count

if __name__ == '__main__':
    from Model import Model
    from Species import Species
    from SpeciesReference import SpeciesReference
    from Reaction import Reaction
    from FirstReaction import FirstReaction

    # Immigration.
    m = Model(0)
    m.species['s1'] = Species(0)
    m.reactions['r1'] = Reaction(m, [], [SpeciesReference('s1', 1)], '1')
    solver = FirstReaction(m, 1000)
    simulation = TimeSeriesUniform(solver, ['s1'], ['r1'], range(11))
    simulation.generateTrajectory()
    print(simulation.populations)
    print(simulation.reactionCounts)
