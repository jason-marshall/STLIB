"""Implements the TrajectoryCalculator class."""

import numpy

class TrajectoryCalculator:
    """Calculates populations and reaction counts for all reaction style
    trajectories."""
    
    def __init__(self, model):
        """Build the list of state change vectors from the model."""
        # Evaluate the model to make sure the initial amount values are set.
        errors = model.evaluate()
        assert not errors
        # The list of state change vectors.
        self.stateChangeVectors = []
        for reaction in model.reactions:
            # The dense state change vector.
            dense = [0] * len(model.speciesIdentifiers)
            for x in reaction.reactants:
                i = model.speciesIdentifiers.index(x.species)
                dense[i] -= x.stoichiometry
            for x in reaction.products:
                i = model.speciesIdentifiers.index(x.species)
                dense[i] += x.stoichiometry
            # The sparse state change vector.
            sparse = []
            for i in range(len(dense)):
                if dense[i] != 0:
                    sparse.append((i, dense[i]))
            self.stateChangeVectors.append(sparse)

    def makeFramesAtReactionEvents(self, trajectories, index,
                                   includeStart=False, includeEnd=False):
        """Make numpy arrays of the times, populations, and reaction counts
        at each event time. If includeStart is True, the starting time is an
        event. Likewise for incudeEnd. Return a tuple of the arrays."""
        initialPopulations = trajectories.initialPopulations[index]
        indices = trajectories.indices[index]
        times = trajectories.times[index]
        if len(indices) != len(times):
            raise Exception('The number of reaction indices does not match '\
                            'the number of times.')
        # Sizes.
        numberOfFrames = len(indices) + 2
        numberOfSpecies = len(initialPopulations)
        numberOfReactions = len(self.stateChangeVectors)
        # Build the arrays.
        eventTimes = numpy.zeros(numberOfFrames)
        populations = numpy.zeros(numberOfFrames * numberOfSpecies)
        populations.shape = (numberOfFrames, -1)
        reactionCounts = numpy.zeros(numberOfFrames * numberOfReactions)
        reactionCounts.shape = (numberOfFrames, -1)
        # The initial state at the start time.
        eventTimes[0] = trajectories.initialTime
        for i in range(numberOfSpecies):
            populations[0][i] = initialPopulations[i]
        reactionCounts[0] = 0
        # For each reaction.
        for i in range(len(indices)):
            eventTimes[i+1] = times[i]
            # The previous state.
            populations[i+1] = populations[i]
            reactionCounts[i+1] = reactionCounts[i]
            # Fire the reaction.
            for (index, value) in\
                    self.stateChangeVectors[indices[i]]:
                populations[i+1][index] += value
            reactionCounts[i+1][indices[i]] += 1
        # The final state at the end time.
        eventTimes[-1] = trajectories.finalTime
        populations[-1] = populations[-2]
        reactionCounts[-1] = reactionCounts[-2]
        # Exclude the start and end times if necessary.
        if not includeStart:
            eventTimes = eventTimes[1:]
            populations = populations[1:]
            reactionCounts = reactionCounts[1:]
        if not includeEnd:
            eventTimes = eventTimes[:-1]
            populations = populations[:-1]
            reactionCounts = reactionCounts[:-1]
        # Return the tuple of arrays.
        return (eventTimes, populations, reactionCounts)

    def computeFinalReactionCounts(self, trajectories, index):
        indices = trajectories.indices[index]
        numberOfReactions = len(self.stateChangeVectors)
        reactionCounts = numpy.zeros(numberOfReactions)
        for i in indices:
            reactionCounts[i] += 1
        return reactionCounts

    def computeFinalPopulations(self, trajectories, index):
        initialPopulations = trajectories.initialPopulations[index]
        # Compute the final reaction counts.
        reactionCounts = self.computeFinalReactionCounts(trajectories, index)
        # The populations array.
        populations = numpy.array(initialPopulations)
        # For each reaction in the model.
        for i in range(len(self.stateChangeVectors)):
            count = reactionCounts[i]
            # Fire the reaction the cumulative number of times.
            for (index, value) in self.stateChangeVectors[i]:
                populations[index] += count * value
        return populations

def main():
    import sys
    sys.path.insert(1, '..')
    from Model import Model
    from Reaction import Reaction
    from Species import Species
    from SpeciesReference import SpeciesReference
    from TimeSeriesAllReactions import TimeSeriesAllReactions

    model = Model()
    model.id = 'model'
    model.speciesIdentifiers.append('s1')
    model.species['s1'] = Species('C1', 'species 1', '13')
    model.speciesIdentifiers.append('s2')
    model.species['s2'] = Species('C1', 'species 2', '17')
    model.reactions.append(
        Reaction('r1', 'reaction 1', [SpeciesReference('s1')], 
                 [SpeciesReference('s2')], True, '1.5'))
    model.reactions.append(
        Reaction('r2', 'reaction 2', 
                 [SpeciesReference('s1'), SpeciesReference('s2')], 
                 [SpeciesReference('s1', 2)], True, '2.5'))

    x = TrajectoryCalculator(model)
    # Two species, two reactions.
    t = TimeSeriesAllReactions([0, 1], [0, 1], 0., 2.)
    t.appendIndices([0, 1, 1, 0])
    t.appendTimes([0.25, 0.5, 0.75, 1.])
    t.appendInitialPopulations([13, 17])
    times, populations, reactionCounts =\
        x.makeFramesAtReactionEvents(t, 0, True, True)
    print 'times'
    print times
    print 'populations'
    print populations
    print 'reactionCounts'
    print reactionCounts
    print 'Final reaction counts.'
    print x.computeFinalReactionCounts(t, 0)
    print 'Final populations.'
    print x.computeFinalPopulations(t, 0)

if __name__ == '__main__':
    main()
